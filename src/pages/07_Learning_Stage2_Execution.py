"""
Learning Agent - Stage 2: 실행 (Execution)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.learning_agent_shared import LearningAgentRunner
from src.components.sidebar_learning import render_learning_sidebar
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage


def render_message(msg):
    """메시지 렌더링"""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, ToolMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, SystemMessage):
        # SystemMessage는 표시하지 않음
        pass


def main():
    st.set_page_config(
        page_title="Stage 2: 실행",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Stage 2: 데이터 분석 실행")
    
    st.markdown("""
    **Learning Agent - 실행 단계**
    
    이제 계획한 대로 데이터 분석을 실행해봅시다!  
    튜터와 대화하며 분석을 진행하고, 완료되면 **"완료"**라고 입력하세요.
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()
    
    # 사이드바 렌더링
    render_learning_sidebar(agent)
    
    # 메인 영역
    state = agent.get_state()
    
    if not state or not state.get("planning_completed"):
        st.warning("⚠️ Stage 1: 계획을 먼저 완료하세요")
        st.info("👈 좌측 사이드바에서 'Stage 1: 계획'으로 이동하세요")
    else:
        current_phase = state.get("phase", "planning")
        execution_completed = state.get("execution_completed", False)
        planning_info = state.get("planning_info")  # State에서 계획 정보 가져오기
        
        # 계획 정보 표시 (참조 가능하도록)
        if planning_info:
            with st.expander("📋 나의 학습 계획 (참고용)", expanded=False):
                st.markdown(f"""
**데이터 설명:**  
{planning_info.get('dataset_description', 'N/A')}

**분석 목표:**  
{planning_info.get('data_analysis_objective', 'N/A')}

**분석 계획:**  
{planning_info.get('data_analysis_plan', 'N/A')}
""")
        
        # 실행 단계가 아직 시작 안 된 경우 안내 메시지 표시
        if current_phase == "planning":
            st.info("계획 단계가 완료되었습니다. 아래에서 대화를 시작하세요!")
        
        st.divider()
        
        # 대화 이력 표시
        st.header("💬 대화")
        
        messages = state.get("messages", [])
        
        # 실행 단계 시작 시점 찾기 (finalize_planning의 ToolMessage 이후부터)
        execution_start_idx = 0
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and "계획이 완성되었습니다" in msg.content:
                execution_start_idx = i
                break
        
        # 실행 단계 메시지만 표시
        for msg in messages[execution_start_idx:]:
            if not isinstance(msg, SystemMessage):
                render_message(msg)
        
        st.divider()
        
        if execution_completed:
            st.success("✅ 실행 단계가 완료되었습니다!")
            
            # Phase 확인
            if current_phase == "reflection":
                st.info("👉 좌측 사이드바에서 'Stage 3: 반성'으로 이동하세요")
            else:
                st.warning("⚠️ Phase가 아직 reflection으로 전환되지 않았습니다. 잠시만 기다려주세요...")
                st.rerun()
        else:
            # 사용자 입력
            st.subheader("메시지 입력")
            
            # 도움말
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.info("💡 **Tip:** 분석 중 궁금한 점을 자유롭게 질문하세요!")
            
            with col2:
                st.warning("⚠️ 분석이 완료되면 **'완료'**라고 입력하세요")
            
            # 예제 질문
            with st.expander("💭 예제 질문 보기"):
                example_questions = [
                    "데이터를 먼저 살펴봤는데, 이상한 값들이 보여요",
                    "분석 결과를 어떻게 해석해야 할지 모르겠어요",
                    "시각화를 만들었는데 의미가 있는지 모르겠어요",
                    "다음 단계로 무엇을 해야 할까요?",
                ]
                
                for i, q in enumerate(example_questions, 1):
                    if st.button(f"{i}. {q}", key=f"example_{i}"):
                        st.session_state.execution_input = q
                        st.rerun()
            
            user_input = st.text_area(
                "메시지를 입력하세요:",
                key="execution_input",
                placeholder="분석 중 궁금한 점을 질문하거나, 완료했으면 '완료'라고 입력하세요...",
                height=100
            )
            
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                send_button = st.button("📤 전송", type="primary", use_container_width=True)
            
            with col2:
                complete_button = st.button("✅ 완료", type="secondary", use_container_width=True)
            
            if send_button and user_input.strip():
                with st.spinner("응답 생성 중..."):
                    result = agent.send_message(user_input)
                st.rerun()
            elif complete_button:
                with st.spinner("다음 단계로 이동 중..."):
                    result = agent.send_message("완료")
                st.rerun()
            elif send_button:
                st.warning("메시지를 입력해주세요")
    
    # State Inspector (개발자용)
    with st.expander("🔧 전체 State 보기 (개발자용)"):
        if state:
            display_state = {
                "phase": state.get("phase"),
                "planning_completed": state.get("planning_completed"),
                "execution_completed": state.get("execution_completed"),
                "reflection_completed": state.get("reflection_completed"),
                "planning_info": state.get("planning_info"),
                "reflection_info": state.get("reflection_info"),
                "message_count": len(state.get("messages", []))
            }
            st.json(display_state)


if __name__ == "__main__":
    main()

