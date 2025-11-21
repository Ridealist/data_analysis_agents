"""
Learning Agent 사이드바 공통 모듈
"""

import streamlit as st
from src.graph.learning_agent import LearningAgentRunner
from src.graph.utils import get_phase_messages, get_thread_specific_key
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.components.dataclass import MessageRole, MessageType
from src.graph.learning_logging import load_learning_threads_for_user

URL_PROF_IMAGE = "./data/teacher.png"

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
    """학습 세션(스레드) 관리 UI (사이드바용)"""

    st.divider()
    st.header("📚 학습 세션 관리")

    # 세션 상태 안전하게 가져오기
    if "learning_thread_list" not in st.session_state:
        try:
            st.session_state.learning_thread_list = load_learning_threads_for_user()
        except Exception as e:
            print(f"[render_thread_manager] thread 목록 로드 실패: {e}")
            st.session_state.learning_thread_list = []

    thread_list = st.session_state.get("learning_thread_list", [])
    current_thread_id = st.session_state.get("learning_thread_id", None)

    # 스레드가 하나도 없거나 현재 스레드가 없으면 새로 생성 안내
    if not thread_list or not current_thread_id:
        st.info("새 학습 세션을 시작하세요")
        if st.button("➕ 새 학습 세션 시작", type="primary", use_container_width=True):
            agent.create_new_thread()
            st.rerun()
        return

    # 학습 세션 목록 표시 (기존 UI 스타일 유지)
    if thread_list:
        for i, thread in enumerate(reversed(thread_list)):  # 최신순
            is_current = thread["id"] == current_thread_id

            col1, col2 = st.columns([3, 1])

            with col1:
                # 세션 정보
                phase_emoji = {
                    "planning": "📝",
                    "execution": "🚀",
                    "reflection": "🪞",
                    "complete": "✅",
                }
                emoji = phase_emoji.get(thread["phase"], "❓")

                label = f"{emoji} {thread['title']}"
                if is_current:
                    label = f"**🔵 {label}** (현재)"

                st.write(label)
                st.caption(f"{thread['created_at']} | {thread['phase']}")

            with col2:
                if not is_current:
                    if st.button(
                        "열기",
                        key=f"switch_{thread['id']}",
                        use_container_width=True,
                    ):
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
            # 실행 단계 전체 대화 표시 (사용자 + 학습 AI 에이전트)
            for msg in execution_messages:
                if isinstance(msg, HumanMessage):
                    st.chat_message("user").write(msg.content)
                elif isinstance(msg, (AIMessage, ToolMessage)):
                    st.chat_message("assistant", avatar=URL_PROF_IMAGE).write(msg.content)
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
        st.info("💡 메인 화면에서 질문하면, 위 대화 영역에 AI 에이전트의 조언이 자동으로 표시됩니다.")
        
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


# ============================================================
# Stage별 사이드바 렌더링 함수
# ============================================================

def _sync_generated_prompt_to_session(state: dict) -> None:
    """
    LangGraph state의 generated_prompt를 session_state로 동기화 (변경 시에만).
    스레드별로 독립적인 키를 사용하여 멀티 스레드 환경을 지원합니다.
    """
    latest = state.get("generated_prompt")
    if not latest:
        return

    # 스레드별 키 생성
    edit_key = get_thread_specific_key("sidebar_generated_prompt")
    last_key = get_thread_specific_key("_last_generated_prompt")

    # 새로 생성되었거나 변경된 경우에만 덮어쓰기
    if st.session_state.get(last_key) != latest:
        st.session_state[edit_key] = latest
        st.session_state[last_key] = latest


def render_stage1_sidebar(agent: LearningAgentRunner, state: dict) -> None:
    """Stage 1 (Planning) 사이드바 - 시스템 프롬프트 표시/수정/저장."""
    st.header("🏷️ 생성된 시스템 프롬프트")

    generated_prompt = state.get("generated_prompt")
    
    # 스레드별 키 생성
    thread_id = st.session_state.get("learning_thread_id", "")
    original_key = get_thread_specific_key("original_generated_prompt", thread_id)
    edit_key = get_thread_specific_key("sidebar_generated_prompt", thread_id)

    # 원본 프롬프트 보관 (스레드별로 최초 한 번만 저장)
    if generated_prompt and original_key not in st.session_state:
        st.session_state[original_key] = generated_prompt

    if generated_prompt:
        _sync_generated_prompt_to_session(state)  # 스레드별 키로 동기화
        is_disabled = False
        caption_text = "💡 이 프롬프트를 수정한 후 아래 '저장' 버튼을 눌러주세요."
    else:
        is_disabled = True
        caption_text = "💡 계획을 완료하고 '완료' 키워드를 입력해주세요."

    edited_prompt = st.text_area(
        "데이터 분석을 위한 시스템 프롬프트",
        placeholder="(계획 단계를 완료하면 생성된 프롬프트가 나타납니다.)",
        height=400,
        disabled=is_disabled,
        key=edit_key,  # 스레드별 키 사용
        label_visibility="collapsed",
    )

    st.caption(caption_text)

    if generated_prompt:
        col1, col2 = st.columns([1, 1])

        with col1:
            save_button = st.button(
                "💾 프롬프트 저장",
                type="primary",
                use_container_width=True,
                key=f"save_prompt_button_{thread_id}",  # 스레드별 버튼 키
                help="수정한 프롬프트를 저장하고 Stage 2에서 사용합니다",
            )

        with col2:
            reset_button = st.button(
                "🔄 원본으로 복원",
                use_container_width=True,
                key=f"reset_prompt_button_{thread_id}",  # 스레드별 버튼 키
                help="AI가 생성한 원본 프롬프트로 되돌립니다",
            )

        # 저장 버튼 클릭 시
        if save_button:
            if edited_prompt.strip():
                try:
                    agent.graph.update_state(
                        agent.config,
                        {"generated_prompt": edited_prompt},
                        as_node="finalize_planning",
                    )

                    st.success("✅ 프롬프트가 저장되었습니다!")
                    st.info("💡 Stage 2에서 수정된 프롬프트가 사용됩니다.")
                except Exception as e:
                    st.error(f"❌ 프롬프트 저장 실패: {e}")
            else:
                st.warning("⚠️ 프롬프트가 비어있습니다. 내용을 입력해주세요.")

        # 원본 복원 버튼 클릭 시
        if reset_button:
            original = st.session_state.get(original_key)
            if original:
                try:
                    agent.graph.update_state(
                        agent.config,
                        {"generated_prompt": original},
                        as_node="finalize_planning",
                    )
                    st.success("✅ 원본 프롬프트로 복원되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 프롬프트 복원 실패: {e}")
            else:
                st.warning("⚠️ 원본 프롬프트를 찾을 수 없습니다.")


def render_stage2_sidebar(agent: LearningAgentRunner, state: dict) -> None:
    """Stage 2 (Execution) 사이드바 - 기존 실행 단계 대화 인터페이스 재사용."""
    render_execution_sidebar_chat(agent, state)


def render_stage3_sidebar(agent: LearningAgentRunner, state: dict) -> None:
    """Stage 3 (Reflection) 사이드바 - 실행 단계 대화 요약 표시."""
    st.header("📊 실행 단계 대화 요약")

    execution_summary = state.get("execution_summary", "")
    placeholder_text = "('요약' 키워드를 입력하면 Stage 2에서 데이터 분석하면서 진행한 대화 요약본이 생성됩니다.)"

    # 요약이 있으면 내용 표시, 없으면 placeholder 표시
    display_value = execution_summary if execution_summary else placeholder_text

    st.text_area(
        "Stage 2에서 진행된 대화 내용",
        value=display_value,
        height=400,
        disabled=True,  # 항상 비활성화 (수정 불가)
        key="sidebar_execution_summary",
        label_visibility="collapsed",
    )

    # 요약이 있을 때만 안내 캡션 표시
    if execution_summary:
        st.caption("💡 이 요약은 실행 단계의 대화를 바탕으로 생성되었습니다.")
    else:
        st.caption("💡 메인 화면에서 '요약' 키워드를 입력하여 요약을 생성하세요.")


# ============================================================
# 메인 사이드바 디스패처
# ============================================================

def render_learning_sidebar(agent: LearningAgentRunner, current_page: str = None):
    """
    Learning Agent 사이드바 렌더링 메인 함수
    
    Args:
        agent: LearningAgentRunner 인스턴스
        current_page: 현재 페이지 ("stage1", "stage2", "stage3" 또는 None)
    """

    with st.sidebar:
        # 로그인된 사용자 정보 표시 (있을 경우)
        if st.session_state.get("logged_in") and st.session_state.get("username"):
            st.caption(f"👤 {st.session_state['username']}님 로그인됨")

        state = agent.get_state()

        if not state:
            st.warning("⚠️ State를 불러올 수 없습니다.")
            render_thread_manager(agent)
            return

        if current_page == "stage2":
            render_stage2_sidebar(agent, state)
        elif current_page == "stage3":
            render_stage3_sidebar(agent, state)
        else:  # stage1 또는 None
            render_stage1_sidebar(agent, state)

        # 세션 관리 UI (항상 표시)
        render_thread_manager(agent)
