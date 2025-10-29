"""
Learning Agent - Stage 1: 계획 (Planning)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.learning_agent_shared import LearningAgentRunner
from src.components.sidebar_learning import render_learning_sidebar
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def render_message(msg, planning_info=None):
    """메시지 렌더링"""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            # Tool call 정보 표시 - planning_info 사용
            with st.chat_message("assistant"):
                st.write("📝 계획이 수립되었습니다!")
                if planning_info:
                    with st.expander("수집된 정보 보기"):
                        st.json(planning_info)
        else:
            st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, ToolMessage):
        st.chat_message("assistant").write(msg.content)


def main():
    st.set_page_config(
        page_title="Stage 1: 계획",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Stage 1: 데이터 분석 계획 수립")
    
    st.markdown("""
    **Learning Agent - 계획 단계**
    
    튜터와 대화하며 데이터 분석 계획을 세워봅시다! 다음 정보를 수집합니다:
    - 📊 데이터 설명
    - 🎯 분석 목표
    - 📋 분석 계획
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()
    
    # 사이드바 렌더링
    render_learning_sidebar(agent)
    
    # 메인 영역: 대화 인터페이스
    state = agent.get_state()
    
    # 초기 실행 (메시지가 없는 경우)
    if not state or not state.get("messages"):
        with st.spinner("튜터와 연결 중..."):
            result = agent.initialize()
            st.rerun()
    
    # 대화 이력 표시
    if state and state.get("messages"):
        st.header("💬 대화")
        
        messages = state["messages"]
        planning_info = state.get("planning_info")  # State에서 직접 가져오기
        
        # 모든 메시지 표시
        for msg in messages:
            render_message(msg, planning_info)
        
        st.divider()
        
        # 현재 상태 확인
        current_phase = state.get("phase", "planning")
        planning_completed = state.get("planning_completed", False)
        
        if planning_completed and planning_info:
            st.success("✅ 계획 단계가 완료되었습니다!")
            st.info("👉 좌측 사이드바에서 'Stage 2: 실행'으로 이동하세요")
            
            # 수집된 계획 정보 표시 - State에서 직접 가져온 정보 사용
            with st.expander("📋 수립된 계획 확인", expanded=True):
                st.markdown(f"""
**데이터 설명:**  
{planning_info.get('dataset_description', 'N/A')}

**분석 목표:**  
{planning_info.get('data_analysis_objective', 'N/A')}

**분석 계획:**  
{planning_info.get('data_analysis_plan', 'N/A')}
""")
        
        else:
            # 사용자 입력
            st.subheader("메시지 입력")
            
            # 예제 응답
            with st.expander("💡 예제 응답 보기"):
                st.write("**데이터 설명 예시:**")
                st.write("- iris 데이터는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비, 그리고 품종 정보를 담고 있어요")
                st.write("- 온라인 쇼핑몰의 고객 구매 데이터입니다. 나이, 성별, 구매 금액 등이 있어요")
                
                st.write("\n**분석 목표 예시:**")
                st.write("- 품종별로 꽃잎의 특징이 어떻게 다른지 알아보고 싶어요")
                st.write("- 어떤 연령대가 가장 많이 구매하는지 파악하고 싶어요")
            
            user_input = st.text_input(
                "메시지를 입력하세요:",
                key="planning_input",
                placeholder="튜터의 질문에 답해주세요..."
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                send_button = st.button("📤 전송", type="primary", use_container_width=True)
            
            if send_button and user_input.strip():
                with st.spinner("응답 생성 중..."):
                    result = agent.send_message(user_input)
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

