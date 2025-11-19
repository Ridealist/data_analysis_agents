"""
Learning Agent 사이드바 공통 모듈
"""

import streamlit as st
from src.graph.learning_agent import LearningAgentRunner
from src.graph.utils import get_phase_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from src.components.dataclass import MessageRole, MessageType

# ============================================================
# Helper Functions
# ============================================================

def extract_csv_text_conversation() -> str:
    """messages_UploadData에서 TEXT 타입만 추출"""
    conversations = []
    for role, content_list in st.session_state.get("messages_UploadData", []):
        speaker = "사용자" if role == MessageRole.USER else "AI"
        for content in content_list:
            if (isinstance(content, list) and 
                len(content) == 2 and 
                content[0] == MessageType.TEXT):
                conversations.append(f"{speaker}: {content[1]}")
    return "\n\n".join(conversations)

# ============================================================
# UI Helper Functions
# ============================================================

def render_thread_manager(agent: LearningAgentRunner):
    """스레드 관리 UI (사이드바용)"""
    
    st.divider()

    st.header("📚 학습 세션 관리")
    
    # 세션 상태 안전하게 가져오기
    thread_list = st.session_state.get("learning_thread_list", [])
    current_thread_id = st.session_state.get("learning_thread_id", None)
    
    if not thread_list or not current_thread_id:
        st.info("새 학습 세션을 시작하세요")
        if st.button("➕ 새 학습 세션 시작", type="primary", use_container_width=True):
            agent.create_new_thread()
            st.rerun()
        return
    
    if thread_list:
        # 현재 세션 표시
        # current_thread = agent.get_thread_info(current_thread_id)
        # if current_thread:
        #     st.write(f"**현재 세션:** {current_thread['title']}")
        #     st.caption(f"생성: {current_thread['created_at']}")
        #     st.caption(f"단계: {current_thread['phase']}")
        
        # st.divider()
        
        # 세션 선택
        # st.write("**저장된 세션:**")
        
        for i, thread in enumerate(reversed(thread_list)):  # 최신순
            is_current = thread["id"] == current_thread_id
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 세션 정보
                phase_emoji = {
                    "planning": "📝",
                    "execution": "🚀", 
                    "reflection": "🪞",
                    "complete": "✅"
                }
                emoji = phase_emoji.get(thread["phase"], "❓")
                
                label = f"{emoji} {thread['title']}"
                if is_current:
                    label = f"**🔵 {label}** (현재)"
                
                st.write(label)
                st.caption(f"{thread['created_at']} | {thread['phase']}")
            
            with col2:
                if not is_current:
                    if st.button("열기", key=f"switch_{thread['id']}", use_container_width=True):
                        agent.switch_thread(thread["id"])
                        st.rerun()
        
        st.divider()
    
    # 새 세션 시작
    if st.button("➕ 새 학습 세션 시작", type="primary", use_container_width=True):
        agent.create_new_thread()
        st.rerun()


def render_execution_sidebar_chat(agent: LearningAgentRunner, state):
    """
    Execution 단계 사이드바 대화 인터페이스
    
    Args:
        agent: LearningAgentRunner 인스턴스
        state: 현재 state
    """
    # 계획 단계 완료 여부 확인
    planning_completed = state.get("planning_completed", False)

    # 계획 단계가 완료되지 않았으면 메시지와 입력 영역 숨기기
    if not planning_completed:
        return  # 함수 종료하여 메시지와 입력 영역을 표시하지 않음

    st.header("🤔 학습 AI 에이전트의 조언")
    
    messages = state.get("messages", [])
    phase_end_indices = state.get("phase_end_indices")
    execution_completed = state.get("execution_completed", False)
    
    # Execution 단계 메시지만 필터링
    execution_messages = get_phase_messages(messages, "execution", phase_end_indices)

    # 스크롤 가능한 대화 영역
    chat_container = st.container(height=450)  # 실행 단계에서는 프롬프트 없으므로 높이 증가 가능
    
    with chat_container:
        if execution_messages:
            # 학습 AI 에이전트의 조언(AIMessage, ToolMessage)만 표시
            # 사용자 발화(HumanMessage)는 메인 화면에서 이미 보이므로 제외
            for msg in execution_messages:
                if isinstance(msg, AIMessage):
                    st.chat_message("assistant").write(msg.content)
                elif isinstance(msg, ToolMessage):
                    st.chat_message("assistant").write(msg.content)
        else:
            # 메시지가 없을 때 안내
            st.info("💬 메인 화면에서 데이터 분석 질문을 하면, 학습 AI 에이전트의 조언이 여기에 표시됩니다.")

        # 메인 화면 입력에 대한 실시간 조언 표시용 placeholder (대화 영역 안에 위치)
        if not execution_completed:
            if (
                "learning_stream_placeholder" not in st.session_state
                or st.session_state["learning_stream_placeholder"] is None
            ):
                st.session_state["learning_stream_placeholder"] = st.empty()
        
    # 완료 상태 확인 및 안내 / 버튼
    if execution_completed:
        st.success("✅ 실행 단계가 완료되었습니다!")
        st.info("'Stage 3: 반성'으로 이동하세요")
    else:
        # 안내 및 완료 버튼
        st.info("💡 메인 화면에서 질문하면, 위 대화 영역에 튜터의 조언이 자동으로 표시됩니다.")
        
        complete_btn = st.button("✅ 실행 단계 완료", key="execution_sidebar_complete", use_container_width=True, type='primary')
        if complete_btn:
            with st.spinner("다음 단계로 이동 중..."):
                # CSV 대화 추출 및 저장
                csv_history = extract_csv_text_conversation()
                if csv_history:
                    agent.set_csv_execution_history(csv_history)
                
                # Phase 전환
                agent.send_message("분석 완료")
            st.rerun()


def render_learning_sidebar(agent: LearningAgentRunner, current_page: str = None):
    """
    Learning Agent 사이드바 렌더링 (모든 페이지 공통)
    
    Args:
        agent: LearningAgentRunner 인스턴스
        current_page: 현재 페이지 ("stage1", "stage2", "stage3" 또는 None)
                      stage2일 때만 사이드바에 대화 인터페이스 표시
    
    current_page에 따라 다른 UI를 표시:
    - stage2: 사이드바에 대화 인터페이스 표시
    - stage1, stage3, None: 기존처럼 프롬프트 표시
    """
    
    with st.sidebar:
        state = agent.get_state()
        
        # Stage 2 (Execution) 페이지일 때만: 사이드바에 대화 인터페이스
        if current_page == "stage2":
            render_execution_sidebar_chat(agent, state)
        # Stage 3 (Reflection) 페이지일 때: 실행 단계 대화 요약 표시
        elif current_page == "stage3":
            if state:
                execution_summary = state.get("execution_summary")
                if execution_summary:
                    st.header("📊 실행 단계 대화 요약")
                    st.text_area(
                        "Stage 2에서 진행된 대화 내용",
                        value=execution_summary,
                        height=400,
                        disabled=True,
                        key="sidebar_execution_summary",
                        label_visibility="collapsed"
                    )
                    st.caption("💡 이 요약은 실행 단계의 대화를 바탕으로 생성되었습니다.")
        else:
            # 기존 로직: 생성된 시스템 프롬프트 표시
            if state:
                generated_prompt = state.get("generated_prompt")
                if generated_prompt:
                    st.header("🏷️ 생성된 시스템 프롬프트")
                    st.text_area(
                        "데이터 분석을 위한 시스템 프롬프트",
                        value=generated_prompt,
                        height=400,
                        disabled=False,
                        key="sidebar_generated_prompt",
                        label_visibility="collapsed"
                    )
                    st.caption("💡 이 프롬프트는 수집된 정보를 바탕으로 자동 생성되었습니다.")
        
        # 세션 관리 UI (항상 표시)
        render_thread_manager(agent)
