"""
Learning Agent - Stage 2: 실행 (Execution)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.learning_agent import LearningAgentRunner
from src.graph.sidebars import render_learning_sidebar


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
    **좌측 사이드바에서** 튜터와 대화하며 분석을 진행하고, 완료되면 **"완료"** 버튼을 클릭하세요.
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()
    
    # 사이드바 렌더링 (Stage 2일 때 사이드바에 대화 인터페이스 표시)
    render_learning_sidebar(agent, current_page="stage2")
    
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
        
        # 실행 단계 안내
        if current_phase == "planning":
            st.info("✅ 계획 단계가 완료되었습니다. 좌측 사이드바에서 대화를 시작하세요!")
        elif current_phase == "execution":
            if execution_completed:
                st.success("✅ 실행 단계가 완료되었습니다!")
                st.info("👈️ 좌측 사이드바에서 'Stage 3: 반성'으로 이동하세요")
            else:
                st.info("💡 좌측 사이드바에서 튜터와 대화하며 분석을 진행하세요!")
        elif current_phase == "reflection":
            st.success("✅ 실행 단계가 완료되었습니다!")
            st.info("👈️ 좌측 사이드바에서 'Stage 3: 반성'으로 이동하세요")
    
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

