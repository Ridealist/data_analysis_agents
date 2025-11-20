"""
Learning Agent - Stage 3: 반성 (Reflection)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.learning_agent import LearningAgentRunner, generate_summary
from src.graph.sidebars import render_learning_sidebar
from src.graph.utils import get_phase_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from src.graph.learning_logging import ensure_learning_conversation, init_learning_stage3_managers

URL_PROF_IMAGE = "./data/teacher.png"


def render_message(msg, reflection_info=None):
    """메시지 렌더링"""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            pass
            # Tool call 정보 표시 - reflection_info 사용
            # with st.chat_message("assistant"):
            #     st.write("✅ 반성 정보가 수집되었습니다!")
            #     if reflection_info:
            #         with st.expander("수집된 반성 내용 보기"):
            #             st.json(reflection_info)
        else:
            st.chat_message("assistant", avatar=URL_PROF_IMAGE).write(msg.content)
    elif isinstance(msg, ToolMessage):
        st.chat_message("assistant", avatar=URL_PROF_IMAGE).write(msg.content)
    elif isinstance(msg, SystemMessage):
        # SystemMessage는 표시하지 않음
        pass


def main():
    st.set_page_config(
        page_title="Stage 3: 반성",
        page_icon="🪞",
        layout="wide"
    )

    # 로그인 여부 확인 (로그인하지 않았다면 접근 차단)
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("⚠️ 이 페이지는 로그인이 필요한 페이지입니다.")
        st.info("👈 상단 메뉴에서 '메인 페이지'로 이동해 로그인 후 다시 시도해주세요.")
        st.stop()

    st.title("🪞 Stage 3: 학습 반성 및 정리")
    
    st.markdown("""
    **Learning AI Agent - 반성 단계**
    
    이번 학습을 돌아보며 성찰해봅시다! 다음 정보를 수집합니다:
    - 🎯 목표 달성 여부
    - 👍 좋은 프롬프트
    - 💡 새롭게 배운 것
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()

    # 공통 Conversation 보장
    ensure_learning_conversation()

    # Stage3(Reflection)용 메시지 매니저 초기화
    _, _, stage3_message_manager = init_learning_stage3_managers()
    if stage3_message_manager:
        st.session_state["learning_stage3_message_manager"] = stage3_message_manager
    
    # 사이드바 렌더링
    render_learning_sidebar(agent, current_page="stage3")
    
    # 메인 영역
    state = agent.get_state()
    
    if not state or not state.get("execution_completed"):
        st.warning("⚠️ Stage 2: 실행을 먼저 완료하세요")
        st.info("👈 좌측 사이드바에서 'Stage 2: 실행'으로 이동하세요")
        
        ##TODO 배포시 삭제: 테스트 모드 추가
        # st.divider()
        # st.subheader("🧪 개발/테스트 모드")
        # st.caption("Stage 1, 2를 건너뛰고 Stage 3을 바로 테스트할 수 있습니다.")
        
        # if st.button("🚀 Stage 3 테스트 모드로 시작", type="secondary"):
        #     with st.spinner("테스트 환경 설정 중..."):
        #         agent.setup_test_reflection_stage()
        #     st.success("✅ 테스트 환경이 설정되었습니다!")
        #     st.rerun()
    else:
        current_phase = state.get("phase", "reflection")
        reflection_completed = state.get("reflection_completed", False)
        planning_info = state.get("planning_info")  # State에서 계획 정보 가져오기
        
        # 계획 정보 표시 (참조 가능하도록)
        if planning_info:
            with st.expander("📋 나의 학습 계획", expanded=False):
                plan = planning_info.get('data_analysis_plan', '')
                plan_display = plan if plan and plan.strip() not in ["", "N/A", "정보 없음", "없음"] else "(구체적인 계획 없이 진행)"
                st.markdown(f"""
**데이터 설명:**  
{planning_info.get('dataset_description', 'N/A')}

**분석 목표:**  
{planning_info.get('data_analysis_objective', 'N/A')}

**분석 계획 (선택):**  
{plan_display}
""")
        
        st.divider()

        # 메시지/반성 경계 관리용 세션 상태
        if "stage3_reflection_start_idx" not in st.session_state:
            st.session_state["stage3_reflection_start_idx"] = None

        # 현재 메시지 스냅샷
        messages = state.get("messages", [])
        reflection_info = state.get("reflection_info")
        phase_end_indices = state.get("phase_end_indices")

        # Reflection 단계 메시지만 필터링
        reflection_messages = get_phase_messages(messages, "reflection", phase_end_indices)

        st.header("💬 대화")

        # 안내 메시지(항상 고정으로 맨 위에 표시)
        with st.chat_message("assistant", avatar=URL_PROF_IMAGE):
            st.markdown("🪞 Stage 3를 시작해요. 아래 중 하나를 선택해주세요: \n\n 1. 반성 진행 \n 2. 실행 단계 대화 요약 \n\n 원하는 선택을 '반성' 또는 '요약'으로 답해주세요.")
        
        # 반성 단계 메시지 표시
        for msg in reflection_messages:   
            if not isinstance(msg, SystemMessage):
                # 특별한 요약 생성 메시지는 표시하지 않음
                if isinstance(msg, HumanMessage) and msg.content == "__GENERATE_SUMMARY__":
                    continue
                render_message(msg, reflection_info)
        
        # # 요약 메시지 이후의 메시지만 표시 (반성 단계 메시지)
        # reflection_start_idx = 0
        # for i, msg in enumerate(messages):
        #     if isinstance(msg, AIMessage) and "실행 단계 대화 요약이 완료되었습니다" in msg.content:
        #         reflection_start_idx = i + 1
        #         break
        
        # # 요약 메시지도 표시
        # for i, msg in enumerate(messages):
        #     if isinstance(msg, AIMessage) and "실행 단계 대화 요약이 완료되었습니다" in msg.content:
        #         st.chat_message("assistant").write(msg.content)
        #         break
        
        # # 반성 단계 메시지 표시
        # for msg in messages[reflection_start_idx:]:
        #     if not isinstance(msg, SystemMessage):
        #         # 특별한 요약 생성 메시지는 표시하지 않음
        #         if isinstance(msg, HumanMessage) and msg.content == "__GENERATE_SUMMARY__":
        #             continue
        #         render_message(msg, reflection_info)

        if reflection_completed or current_phase == "complete":
            st.success("🎉 모든 학습 과정이 완료되었습니다!")

            st.divider()

            # 최종 마무리 멘트 + 전체 학습 요약 출력
            summary_text = None
            try:
                summary_result = generate_summary(state)
                messages = summary_result.get("messages", [])
                if messages and isinstance(messages[0], AIMessage):
                    summary_text = messages[0].content
            except Exception:
                summary_text = None

            with st.chat_message("assistant", avatar=URL_PROF_IMAGE):
                if summary_text:
                    st.markdown(summary_text)

            if st.button("🆕 새로운 학습 시작하기", type="primary"):
                agent.reset()
                st.success("새로운 학습을 시작합니다!")
                st.rerun()        

        else:
            # 사용자 입력
            st.subheader("메시지 입력")
            
            # 도움말
            st.info("💡 **Tip:** 솔직하게 자신의 학습 과정을 돌아보세요. 실패와 시행착오도 중요한 학습입니다!")
            
            user_input = st.text_area(
                "메시지를 입력하세요:",
                key="reflection_input",
                placeholder="튜터의 질문에 답해주세요...",
                height=80
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                send_button = st.button("📤 전송", type="primary", use_container_width=True)
            
            if send_button and user_input.strip():
                # 1) 턴 번호 증가 (Stage1/2와 공통 카운터)
                if "learning_turn" not in st.session_state:
                    st.session_state["learning_turn"] = 0
                st.session_state["learning_turn"] += 1
                current_turn = st.session_state["learning_turn"]

                # 2) Stage3 user 메시지 DynamoDB 저장
                try:
                    conversation_id = st.session_state.get("learning_conversation_id")
                    stage3_msg_manager = st.session_state.get("learning_stage3_message_manager")

                    if conversation_id and stage3_msg_manager:
                        stage3_msg_manager.save_message(
                            conversation_id=conversation_id,
                            role="user",
                            content_type="learning_stage3_user_text",
                            content=user_input,
                            llm_model=None,
                            turn=current_turn,
                        )
                except Exception as e:
                    print(f"[Stage3] LearningAgent 사용자 입력 로그 저장 실패: {e}")

                # 3) 기존 LearningAgent 호출
                with st.spinner("응답 생성 중..."):
                    result = agent.send_message(user_input)
                st.rerun()
            elif send_button:
                st.warning("메시지를 입력해주세요")

if __name__ == "__main__":
    main()

