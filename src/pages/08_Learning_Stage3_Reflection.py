"""
Learning Agent - Stage 3: 반성 (Reflection)
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.learning_agent import LearningAgentRunner
from src.graph.sidebars import render_learning_sidebar
from src.graph.utils import get_phase_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage


def render_message(msg, reflection_info=None):
    """메시지 렌더링"""
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            # Tool call 정보 표시 - reflection_info 사용
            with st.chat_message("assistant"):
                st.write("✅ 반성 정보가 수집되었습니다!")
                if reflection_info:
                    with st.expander("수집된 반성 내용 보기"):
                        st.json(reflection_info)
        else:
            st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, ToolMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, SystemMessage):
        # SystemMessage는 표시하지 않음
        pass


def main():
    st.set_page_config(
        page_title="Stage 3: 반성",
        page_icon="🪞",
        layout="wide"
    )
    
    st.title("🪞 Stage 3: 학습 반성 및 정리")
    
    st.markdown("""
    **Learning Agent - 반성 단계**
    
    이번 학습을 돌아보며 성찰해봅시다! 다음 정보를 수집합니다:
    - 🎯 목표 달성 여부
    - 👍 좋은 프롬프트
    - 👎 개선이 필요한 프롬프트
    - 💡 새롭게 배운 것
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()
    
    # 사이드바 렌더링
    render_learning_sidebar(agent, current_page="stage3")
    
    # 메인 영역
    state = agent.get_state()
    
    if not state or not state.get("execution_completed"):
        st.warning("⚠️ Stage 2: 실행을 먼저 완료하세요")
        st.info("👈 좌측 사이드바에서 'Stage 2: 실행'으로 이동하세요")
    else:
        current_phase = state.get("phase", "reflection")
        reflection_completed = state.get("reflection_completed", False)
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
        with st.chat_message("assistant"):
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

            # 다운로드 옵션
            st.divider()

            import json
            summary_data = {
                "planning": state.get("planning_info"),
                "reflection": state.get("reflection_info")
            }
            summary_json = json.dumps(summary_data, ensure_ascii=False, indent=2)

            st.download_button(
                label="📥 학습 요약 다운로드 (JSON)",
                data=summary_json,
                file_name="learning_summary.json",
                mime="application/json"
            )

            if st.button("🆕 새로운 학습 시작하기", type="primary"):
                agent.reset()
                st.success("새로운 학습을 시작합니다!")
                st.rerun()        

        else:
            # 사용자 입력
            st.subheader("메시지 입력")
            
            # 도움말
            st.info("💡 **Tip:** 솔직하게 자신의 학습 과정을 돌아보세요. 실패와 시행착오도 중요한 학습입니다!")
            
            # 예제 응답
            with st.expander("💭 예제 응답 보기"):
                st.write("**목표 달성 여부 예시:**")
                st.write("- 목표의 70% 정도 달성한 것 같아요. 품종별 차이는 알았지만 더 깊은 분석은 못했어요")
                st.write("- 완전히 달성했어요! 데이터에서 의미있는 인사이트를 찾았습니다")
                
                st.write("\n**좋은 프롬프트 예시:**")
                st.write("- '이 데이터에서 어떤 이야기를 찾을 수 있을까요?' - 구체적인 방향을 찾는데 도움이 됐어요")
                st.write("- '왜 이 결과가 나왔을까요?' - 결과의 의미를 이해하게 됐어요")
                
                st.write("\n**나쁜 프롬프트 예시:**")
                st.write("- '코드 짜줘' - 너무 막연했고 AI에게 전부 의존했어요")
                st.write("- '이거 어떻게 해?' - 구체적이지 않아서 제대로 된 답을 못 얻었어요")
                
                st.write("\n**새롭게 배운 것 예시:**")
                st.write("- 데이터를 보기 전에 질문을 먼저 정하는 게 중요하다는 걸 알았어요")
                st.write("- AI에게 '왜'를 물어보면 더 깊이 있는 대화가 가능하다는 것을 배웠어요")
            
            user_input = st.text_area(
                "메시지를 입력하세요:",
                key="reflection_input",
                placeholder="튜터의 질문에 솔직하게 답해주세요...",
                height=120
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

