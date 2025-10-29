"""
Learning Agent 사이드바 공통 모듈
"""

import streamlit as st
from src.components.learning_agent_shared import LearningAgentRunner, render_thread_manager


def render_learning_sidebar(agent: LearningAgentRunner):
    """Learning Agent 사이드바 렌더링 (모든 페이지 공통)"""
    
    with st.sidebar:
        # 현재 세션 상태
        st.header("📊 현재 세션 상태")
        
        state = agent.get_state()
        
        if state:
            phase = state.get("phase", "planning")
            
            # 진행 단계 체크리스트
            st.write(f"{'✅' if state.get('planning_completed') else '🔄' if phase == 'planning' else '⬜'} Stage 1: 계획")
            st.write(f"{'✅' if state.get('execution_completed') else '🔄' if phase == 'execution' else '⬜'} Stage 2: 실행")
            st.write(f"{'✅' if state.get('reflection_completed') else '🔄' if phase == 'reflection' else '⬜'} Stage 3: 반성")
            
            # 계획 정보 표시
            planning_info = state.get("planning_info")
            if planning_info:
                with st.expander("📋 수립된 계획"):
                    st.write(f"**목표:** {planning_info.get('data_analysis_objective', 'N/A')[:50]}...")
            
            # 디버그 정보
            with st.expander("디버그 정보"):
                st.write(f"**Phase:** `{phase}`")
                st.write(f"**Planning:** {'✅' if state.get('planning_completed', False) else '❌'}")
                st.write(f"**Execution:** {'✅' if state.get('execution_completed', False) else '❌'}")
                st.write(f"**Reflection:** {'✅' if state.get('reflection_completed', False) else '❌'}")
                st.code(st.session_state.get("learning_thread_id", "N/A"), language=None)
            
            st.divider()
            
            # 캐시 삭제
            if st.button("🗑️ 캐시 삭제 (배포시 피쳐 삭제)", use_container_width=True):
                st.cache_resource.clear()
                st.success("캐시 삭제!")
                st.rerun()
        else:
            st.info("학습을 시작하세요")
        
        st.divider()
        
        # 세션 관리 UI
        render_thread_manager(agent)

