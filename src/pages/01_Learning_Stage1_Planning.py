"""
Learning Agent - Stage 1: 계획 (Planning)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.learning_agent import LearningAgentRunner
from src.graph.utils import get_phase_messages
from src.graph.sidebars import render_learning_sidebar
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.graph.learning_logging import ensure_learning_conversation

URL_PROF_IMAGE = "./data/teacher.png"

def show_planning_info(planning_data: dict):
    """계획 정보 표시"""
    if not planning_data:
        st.warning("⚠️ 계획 정보가 없습니다.")
        return
    
    st.subheader("📋 수집된 계획 정보")
    
    st.markdown("### 📊 데이터 설명")
    st.info(planning_data.get('dataset_description', 'N/A'))
    
    st.markdown("### 🎯 분석 목표")
    st.info(planning_data.get('data_analysis_objective', 'N/A'))
    
    st.markdown("### 📋 분석 계획 (선택)")
    plan = planning_data.get('data_analysis_plan', '')
    if not plan or plan.strip() in ["", "N/A", "정보 없음", "없음"]:
        st.info("(구체적인 계획 없이 진행)")
    else:
        st.info(plan)


def render_message(msg, planning_info=None):
    """메시지 렌더링"""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            # Tool call이 있으면 자동으로 정보를 표시하지 않음
            # 사용자가 "계획 확인" 버튼을 선택했을 때만 표시
            pass
        else:
            st.chat_message("assistant", avatar=URL_PROF_IMAGE).write(msg.content)
    elif isinstance(msg, ToolMessage):
        st.chat_message("assistant", avatar=URL_PROF_IMAGE).write(msg.content)


def main():
    st.set_page_config(
        page_title="Stage 1: 계획",
        page_icon="📝",
        layout="wide"
    )

    # 로그인 여부 확인 (로그인하지 않았다면 접근 차단)
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("⚠️ 이 페이지는 로그인이 필요한 페이지입니다.")
        st.info("👈 상단 메뉴에서 '메인 페이지'로 이동해 로그인 후 다시 시도해주세요.")
        st.stop()
    
    st.title("📝 Stage 1: 데이터 분석 계획 수립")
    
    st.markdown("""
    **Learning AI Agent - 계획 단계**
    
    튜터와 대화하며 데이터 분석 계획을 세워봅시다! 다음 정보를 수집합니다:
    - 📊 데이터 설명
    - 🎯 분석 목표
    - 📋 분석 계획 (선택: 계획이 없으면 건너뛸 수 있음)
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()

    # Learning Agent용 DynamoDB 대화 세션 연결 (있으면)
    ensure_learning_conversation()

    # 메인 영역: 대화 인터페이스
    state = agent.get_state()
    
    # 사이드바 렌더링
    render_learning_sidebar(agent, current_page="stage1")
    
    # 초기 실행 (메시지가 없는 경우)
    if not state or not state.get("messages"):
        with st.spinner("튜터와 연결 중..."):
            result = agent.initialize()
            st.rerun()
    
    # 대화 이력 표시
    if state and state.get("messages"):
        st.header("💬 대화")
        
        messages = state["messages"]
        planning_info = state.get("planning_info")
        phase_end_indices = state.get("phase_end_indices")
        
        # Planning 단계 메시지만 필터링
        planning_messages = get_phase_messages(messages, "planning", phase_end_indices)
        
        for msg in planning_messages:
            render_message(msg, planning_info)
        
        st.divider()
        
        # 현재 상태 확인
        planning_completed = state.get("planning_completed", False)
        
        ## TODO: 배포 시 삭제 - Stage 2 테스트 모드 (Stage 1 건너뛰기)
        # if not planning_completed:
            # st.divider()
            # st.subheader("🧪 개발/테스트 모드")
            # st.caption("Stage 1을 건너뛰고 Stage 2를 바로 테스트할 수 있습니다.")
            
            # if st.button("🚀 Stage 2 테스트 모드로 시작", type="secondary"):
            #     with st.spinner("테스트 환경 설정 중..."):
            #         agent.setup_test_execution_stage()
            #     st.success("✅ Stage 2 테스트 환경이 설정되었습니다!")
            #     st.info("👈 상단 메뉴에서 'Stage 2: 실행' 페이지로 이동해 테스트를 진행하세요.")
        
        # 사용자 입력에서 '계획 수정' 확인 의도 체크 (시스템 레벨 처리)
        if messages and isinstance(messages[-1], HumanMessage):
            last_user_input = messages[-1].content.strip().lower()
            reset_keywords = ["계획 수정", "초기화", "다시 시작", "처음부터", "리셋"]
            confirm_keywords = ["확인", "네", "예", "yes", "ok", "좋아"]
            
            # '계획 수정' 관련 키워드가 있고, 확인 의도가 있으면 reset 실행
            has_reset_keyword = any(keyword in last_user_input for keyword in reset_keywords)
            has_confirm = any(keyword in last_user_input for keyword in confirm_keywords)
            
            # 이전 LLM 메시지에서 '계획 수정' 확인 요청이 있었는지 체크
            needs_reset_confirmation = False
            for msg in reversed(messages[:-1]):  # 마지막 사용자 메시지 제외
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    if "처음부터 다시 시작" in msg.content or "계획을 처음부터" in msg.content:
                        needs_reset_confirmation = True
                        break
            
            # 확인이 필요한 상황이고 사용자가 확인했다면 reset 실행
            if needs_reset_confirmation and has_confirm:
                with st.spinner("계획을 초기화하고 처음부터 다시 시작합니다..."):
                    agent.reset_planning()
                st.rerun()

        # ... 기존 완료 메시지 코드 ...
        if planning_completed and planning_info:
            st.success("✅ 계획 단계가 완료되었습니다!")
            st.info("👈️ 좌측 사이드바에서 생성된 프롬프트를 확인하고 'Stage 2: 실행'으로 이동하세요")
        
        else:
            # 사용자 입력
            st.subheader("메시지 입력")
            
            user_input = st.chat_input(
                placeholder="AI 에이전트의 질문에 답해주세요...",
                key="planning_input",
            )

            if user_input:
                try:
                    # 턴 번호 증가
                    if "learning_turn" not in st.session_state:
                        st.session_state["learning_turn"] = 0
                    st.session_state["learning_turn"] += 1
                    current_turn = st.session_state["learning_turn"]

                    conversation_id = st.session_state.get("learning_conversation_id")
                    message_manager = st.session_state.get("learning_message_manager")
                    if conversation_id and message_manager:
                        message_manager.save_message(
                            conversation_id=conversation_id,
                            role="user",
                            content_type="learning_stage1_user_text",
                            content=user_input,
                            llm_model=None,
                            turn=current_turn,
                        )
                except Exception as e:
                    # 로깅 실패는 앱 흐름에 영향을 주지 않음
                    print(f"[Stage1] 사용자 입력 로그 저장 실패: {e}")

                with st.spinner("응답 생성 중..."):
                    _ = agent.send_message(user_input)
                st.rerun()


if __name__ == "__main__":
    main()
