"""
Learning Agent - Stage 2: 실행 (Execution)
CSV Agent 통합 버전
"""

import streamlit as st
import sys
from pathlib import Path

from typing import List, Union
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

from langchain_teddynote import logging
from langchain_experimental.tools import PythonAstREPLTool
from openai import OpenAIError, InternalServerError

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.learning_agent import LearningAgentRunner
from src.graph.sidebars import render_learning_sidebar
from src.graph.utils import get_phase_messages, get_thread_specific_key
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.graph.learning_logging import ensure_learning_conversation

from src.components.dynamo import (
    DynamoDBManager,
    S3Manager,
    ConversationManager,
    MessageManager,
    AgencyConversationManager,
)
from src.components.dataclass import MessageRole, MessageType
from src.components.llm import set_openai_api_key, handle_openai_error
from src.prompts.agent_prompts import AGENT_PREFIX_PROMPT, AGENT_POSTFIX_PROMPT
from src.utils.dataframe_handler import safe_dataframe_display

warnings.filterwarnings("ignore")

# ============================================================
# CSV Agent / DB / 환경 설정
# ============================================================

load_dotenv()

# LangSmith 세팅
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.get("LANGCHAIN_ENDPOINT", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
logging.langsmith("Agent 챗봇 1차 현장적용[수원외고] 0828")

# AWS / Dynamo / S3 설정
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = st.secrets.get("AWS_S3_BUCKET_NAME")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-northeast-2")

CONVERSATIONS_ANALYSIS_TABLE = "learning_conversations"
MESSAGES_ANALYSIS_TABLE = "learning_stage2_dataanalysis"  # Stage2 CSV Agent 전용 메시지 테이블
AGENCY_ANALYSIS_TABLE = "learning_stage2_agency_messages"  # Stage2 Agency Agent 전용 메시지 테이블

URL_PROF_IMAGE = "./data/teacher.png"

# ============================================================
# 매니저 초기화
# ============================================================

@st.cache_resource
def init_managers():
    """AWS 매니저들을 초기화하고 캐시"""
    try:
        dynamo_manager = DynamoDBManager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            conversations_table=CONVERSATIONS_ANALYSIS_TABLE,
            messages_table=MESSAGES_ANALYSIS_TABLE,
        )

        s3_manager = S3Manager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            bucket_name=AWS_S3_BUCKET_NAME,
        )

        conversation_manager = ConversationManager(
            dynamo_manager, CONVERSATIONS_ANALYSIS_TABLE
        )
        message_manager = MessageManager(dynamo_manager)

        agency_conversation_manager = AgencyConversationManager(
            dynamo_manager, AGENCY_ANALYSIS_TABLE
        )

        return (
            dynamo_manager,
            s3_manager,
            conversation_manager,
            message_manager,
            agency_conversation_manager,
        )
    except Exception as e:
        st.error(f"매니저 초기화 실패: {e}")
        return None, None, None, None, None

dynamo_manager, s3_manager, conversation_manager, message_manager, agency_conversation_manager = init_managers()

# Stage2 실행 단계에서 사용할 Agency 매니저를 LangGraph에서도 접근할 수 있도록 세션에 저장
if agency_conversation_manager:
    st.session_state["learning_stage2_agency_manager"] = agency_conversation_manager

# ============================================================
# Streamlit Session State (CSV Agent용)
# ============================================================

if "messages_UploadData" not in st.session_state:
    st.session_state["messages_UploadData"] = []

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "session_uuid" not in st.session_state:
    st.session_state["session_uuid"] = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None

if "model_option_general" not in st.session_state:
    st.session_state["model_option_general"] = "gpt-4.1"

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "last_result_df" not in st.session_state:
    st.session_state["last_result_df"] = None

if "last_figure" not in st.session_state:
    st.session_state["last_figure"] = None

# ============================================================
# 메시지 관련 함수 (CSV Agent용)
# ============================================================

def add_message_main(role: MessageRole, content: List[Union[MessageType, str]]):
    """새로운 메시지를 저장하는 함수 (DB 연동)"""
    message_type, message_content = content

    db_content_type = None
    db_content = None

    # DB 저장 (대화가 생성된 이후에만)
    if st.session_state.get("conversation_id") and message_manager:
        if message_type == MessageType.DATAFRAME and isinstance(
            message_content, (pd.DataFrame, pd.Series)
        ):
            if s3_manager:
                s3_path = s3_manager.upload_to_s3(
                    message_content,
                    "dataframe",
                    st.session_state.get("conversation_id"),
                )
                if s3_path:
                    db_content_type = "dataframe_s3"
                    db_content = s3_path

        elif message_type == MessageType.FIGURE and hasattr(
            message_content, "savefig"
        ):
            if s3_manager:
                s3_path = s3_manager.upload_to_s3(
                    message_content,
                    "figure",
                    st.session_state.get("conversation_id"),
                )
                if s3_path:
                    db_content_type = "figure_s3"
                    db_content = s3_path

        elif message_type == MessageType.TEXT:
            db_content_type = "text"
            db_content = message_content

        elif message_type == MessageType.CODE:
            db_content_type = "code"
            db_content = message_content

        elif message_type == MessageType.ERROR:
            db_content_type = "error"
            db_content = message_content

        if db_content_type and db_content and message_manager:
            llm_model = None
            if role == MessageRole.ASSISTANT:
                llm_model = st.session_state.get(
                    "model_option_general", "gpt-4.1-mini"
                )

            # Stage2 학습 턴 번호를 함께 저장 (learning_stage2_dataanalysis 테이블용)
            current_turn = st.session_state.get("learning_turn")

            message_manager.save_message(
                st.session_state.get("conversation_id"),
                role,
                db_content_type,
                db_content,
                llm_model,
                turn=current_turn,
            )

    # 세션 상태에는 원본 객체 그대로 저장
    messages = st.session_state["messages_UploadData"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])


def print_messages_main():
    """저장된 메시지를 화면에 출력하는 함수"""
    for role, content_list in st.session_state["messages_UploadData"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content

                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)

                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)

                    elif message_type == MessageType.CODE:
                        with st.expander("🔍 실행된 코드 보기", expanded=False):
                            st.code(message_content, language="python")

                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content, use_container_width=True)

                    elif message_type == MessageType.ERROR:
                        pass
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def save_agency_response(output: str, intent: str, assessment: str, conversation_id: str = None):
    """Learning Agent(튜터)의 응답을 agency_conversations 테이블에 저장"""
    conv_id = conversation_id or st.session_state.get("conversation_id")
    if conv_id and agency_conversation_manager:
        try:
            current_turn = st.session_state.get("learning_turn")
            agency_conversation_manager.save_agency_response(
                conv_id,
                output,
                intent,
                assessment,
                turn=current_turn,
            )
        except Exception as e:
            print(f"Agency 응답 저장 중 오류: {e}")


def save_user_input_to_agency(user_input: str):
    """사용자 입력을 Stage2 Agency 테이블(learning_stage2_agency_messages)에 저장"""
    # Learning Agent용 대화 ID를 사용하여 Stage1/2/3 로그와 일관되게 관리
    conv_id = st.session_state.get("learning_conversation_id")
    if conv_id and agency_conversation_manager:
        try:
            current_turn = st.session_state.get("learning_turn")
            agency_conversation_manager.save_user_input(
                conv_id,
                user_input,
                turn=current_turn,
            )
        except Exception:
            # 사용자 경험에 영향을 주지 않도록 조용히 무시
            pass


# ============================================================
# DataAnalysisAgent 생성
# ============================================================

def create_agent(
    dataframe,
    selected_model: str = None,
    api_key: str = None,
    prefix_prompt: str = None,
    postfix_prompt: str = None,
    user_column_guideline: str = None,
    shared_python_tool = None,
):
    """데이터프레임 에이전트 생성"""
    from src.components.dataanalysis import DataAnalysisAgent

    if selected_model is None:
        selected_model = st.session_state.get("model_option_general", "gpt-4.1-mini")

    if api_key is None:
        api_key = st.session_state.get("openai_api_key") or os.environ.get(
            "OPENAI_API_KEY"
        )

    if not api_key:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        st.sidebar.error("⚠️ 시스템 설정에서 API 키를 설정해주세요.")
        return None

    shared_tool = st.session_state.get("python_tool", None)

    return DataAnalysisAgent(
        dataframe,
        model_name=selected_model,
        api_key=api_key,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        column_guideline=user_column_guideline,
        shared_python_tool=shared_tool if shared_python_tool is None else shared_python_tool,
    )


# ============================================================
# 멀티스레드: 메인 데이터 분석 응답 + LearningAgent 튜터 응답
# ============================================================

def thread_safe_ask(
    query: str,
    agent,
    python_tool,
    main_queue: Queue,
    error_queue: Queue,
    add_message_func,
    session_state_proxy: dict,
):
    """
    ask 함수를 Thread-safe하게 Queue로 스트리밍
    """
    try:
        if "result_df" in python_tool.locals:
            del python_tool.locals["result_df"]

        response_items = []
        executed_code = ""

        main_queue.put(("USER_MESSAGE", query))

        try:
            for step in agent.stream(query, "abc123"):
                response_items.append(step)

                if "actions" in step:
                    for action in step["actions"]:
                        if action.tool == "python_repl_tool":
                            executed_code = action.tool_input.get("code", "")
                            main_queue.put(("CODE", executed_code))

                            vis_keywords = [
                                "plt.",
                                "sns.",
                                "matplotlib",
                                "seaborn",
                                "plt.show()",
                            ]
                            if any(keyword in executed_code for keyword in vis_keywords):
                                try:
                                    safe_code = executed_code.replace("plt.show()", "")
                                    python_tool.invoke({"query": safe_code})

                                    fig = session_state_proxy.get("last_figure")
                                    if fig and isinstance(fig, plt.Figure):
                                        main_queue.put(("FIGURE", fig))

                                except Exception as e:
                                    error_msg = f"시각화 처리 중 오류 발생: {str(e)}"
                                    main_queue.put(("ERROR", error_msg))

                if session_state_proxy.get("last_result_df") is not None:
                    result_df = session_state_proxy["last_result_df"]
                    if isinstance(result_df, (pd.DataFrame, pd.Series)):
                        main_queue.put(("DATAFRAME", result_df))
                        session_state_proxy["last_result_df"] = None

                if "output" in step:
                    main_queue.put(("TEXT_CHUNK", step["output"]))

            if "result_df" in python_tool.locals:
                result_df = python_tool.locals["result_df"]
                if isinstance(result_df, (pd.DataFrame, pd.Series)):
                    main_queue.put(("DATAFRAME", result_df))
                    del python_tool.locals["result_df"]

            if plt.get_fignums():
                fig = plt.gcf()
                if fig and fig.get_axes():
                    main_queue.put(("FIGURE", fig))

        except OpenAIError as e:
            success, new_model, error_msg = handle_openai_error(
                e, session_state_proxy.get("model_option_general")
            )

            if isinstance(e, InternalServerError):
                error_queue.put(("SERVER_ERROR", error_msg))
            else:
                error_queue.put(("API_ERROR", error_msg))

            if success and new_model:
                main_queue.put(("MODEL_SWITCH", (new_model, error_msg)))

        except Exception as e:
            error_queue.put(("GENERAL_ERROR", str(e)))

        final_text = ""
        for item in response_items:
            if "output" in item:
                final_text += item["output"]

        if final_text:
            main_queue.put(("FINAL_TEXT", final_text))

        main_queue.put(None)

    except Exception as e:
        error_queue.put(("CRITICAL_ERROR", str(e)))
        main_queue.put(None)


def thread_safe_learning_tutor(
    query: str,
    learning_agent: LearningAgentRunner,
    sidebar_queue: Queue,
    error_queue: Queue,
    conversation_id: str = None,
):
    """
    메인 입력 query를 LearningAgent(LangGraph)에 전달하고
    사이드바 튜터 조언을 sidebar_queue로 전달
    (intent/assessment 저장은 learning_agent.execution_node에서 수행)
    """
    try:
        # 먼저 사용자 발화를 사이드바 큐에 전달
        sidebar_queue.put(("USER_TEXT", query))

        # LearningAgent에 메시지 전송 (state 업데이트)
        learning_agent.send_message(query)

        # 최신 state에서 execution 단계 AI/Tool 메시지 추출
        state = learning_agent.get_state()
        advice_text = ""

        if state:
            messages = state.get("messages", [])
            phase_end_indices = state.get("phase_end_indices")
            execution_messages = get_phase_messages(
                messages, "execution", phase_end_indices
            )

            for msg in reversed(execution_messages):
                if isinstance(msg, (AIMessage, ToolMessage)):
                    if getattr(msg, "content", None):
                        advice_text = msg.content
                        break

        if advice_text:
            # 사이드바에 한 번에 표시 (필요시 chunking 가능)
            sidebar_queue.put(("TEXT", advice_text))

        sidebar_queue.put(None)

    except Exception as e:
        error_queue.put(("LEARNING_TUTOR_ERROR", str(e)))
        sidebar_queue.put(None)


def parallel_execution_manager(
    user_input: str,
    agent,
    python_tool,
    learning_agent: LearningAgentRunner,
    add_message_func,
    session_state_proxy: dict,
):
    """
    메인 DataAnalysisAgent와 LearningAgent 튜터를 병렬로 실행하고 Queue들을 반환
    """
    main_queue = Queue()
    sidebar_queue = Queue()
    error_queue = Queue()

    def wrapper_ask():
        thread_safe_ask(
            user_input,
            agent,
            python_tool,
            main_queue,
            error_queue,
            add_message_func,
            session_state_proxy,
        )

        if "result_df" in python_tool.locals:
            result_df = python_tool.locals["result_df"]
            if isinstance(result_df, (pd.DataFrame, pd.Series)):
                main_queue.put(("DATAFRAME_FINAL", result_df))
                st.session_state["last_result_df"] = result_df

        if plt.get_fignums():
            fig = plt.gcf()
            if fig and fig.get_axes():
                main_queue.put(("FIGURE_FINAL", fig))
                st.session_state["last_figure"] = fig

    # DataAnalysisAgent는 별도 스레드에서 실행
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(wrapper_ask)

    # LearningAgent 튜터는 Streamlit 컨텍스트가 필요한 st.session_state를 사용하므로
    # 메인 스레드에서 동기적으로 실행
    conv_id = st.session_state.get("conversation_id")
    thread_safe_learning_tutor(
        user_input,
        learning_agent,
        sidebar_queue,
        error_queue,
        conv_id,
    )

    return main_queue, sidebar_queue, error_queue


def update_ui_from_queues(
    main_queue: Queue,
    sidebar_queue: Queue,
    error_queue: Queue,
    main_container,
    add_message_main,
):
    """
    Queue에서 데이터를 가져와 메인 UI 및 사이드바 UI 업데이트
    """
    main_done = False
    sidebar_done = False

    main_text_placeholder = None
    main_text_buffer = ""
    sidebar_text_buffer = ""

    import traceback

    while not (main_done and sidebar_done):
        if not error_queue.empty():
            try:
                error_type, error_msg = error_queue.get_nowait()

                if error_type == "SERVER_ERROR":
                    st.warning(error_msg)
                    st.info("⏳ 서버 문제가 해결될 때까지 잠시 기다려주세요.")
                    print("[SERVER_ERROR]", error_msg)
                    traceback.print_stack()
                elif error_type == "API_ERROR":
                    st.error(f"API 오류: {error_msg}")
                    print("[API_ERROR]", error_msg)
                    traceback.print_stack()
                elif error_type == "MODEL_SWITCH":
                    new_model, msg = error_msg
                    st.warning(msg)
                    st.info(f"🔄 {new_model}로 다시 시도해주세요.")
                    print("[MODEL_SWITCH]", error_msg)
                    traceback.print_stack()
                else:
                    st.error(f"오류 발생: {error_msg}")
                    print("[ERROR]", error_msg)
                    traceback.print_stack()

            except Empty:
                pass

        # 메인 큐 처리
        if not main_done:
            try:
                item = main_queue.get_nowait()

                if item is None:
                    main_done = True
                    if main_text_buffer:
                        add_message_main(
                            MessageRole.ASSISTANT,
                            [MessageType.TEXT, main_text_buffer],
                        )
                else:
                    msg_type, content = item

                    with main_container:
                        if msg_type == "USER_MESSAGE":
                            pass

                        elif msg_type == "CODE":
                            with st.status("🔍 실행된 코드 보기", expanded=False):
                                st.code(content, language="python")
                            add_message_main(
                                MessageRole.ASSISTANT, [MessageType.CODE, content]
                            )

                        elif msg_type == "DATAFRAME":
                            st.dataframe(content, use_container_width=True)
                            add_message_main(
                                MessageRole.ASSISTANT,
                                [MessageType.DATAFRAME, content],
                            )

                        elif msg_type == "FIGURE":
                            st.pyplot(content)
                            add_message_main(
                                MessageRole.ASSISTANT,
                                [MessageType.FIGURE, content],
                            )
                            plt.close(content)

                        elif msg_type == "TEXT_CHUNK":
                            main_text_buffer += content
                            if main_text_placeholder is None:
                                main_text_placeholder = st.empty()
                            main_text_placeholder.markdown(main_text_buffer)

                        elif msg_type == "FINAL_TEXT":
                            if main_text_placeholder is None:
                                main_text_placeholder = st.empty()
                            main_text_placeholder.markdown(content)
                            main_text_buffer = content

                        elif msg_type == "DATAFRAME_FINAL":
                            st.dataframe(content, use_container_width=True)
                            add_message_main(
                                MessageRole.ASSISTANT,
                                [MessageType.DATAFRAME, content],
                            )
                            st.session_state["last_result_df"] = content

                        elif msg_type == "FIGURE_FINAL":
                            st.pyplot(content)
                            add_message_main(
                                MessageRole.ASSISTANT,
                                [MessageType.FIGURE, content],
                            )
                            plt.close(content)
                            st.session_state["last_figure"] = content

            except Empty:
                pass

        # 사이드바 큐 처리 (LearningAgent 튜터 조언)
        if not sidebar_done:
            try:
                item = sidebar_queue.get_nowait()

                if item is None:
                    sidebar_done = True
                else:
                    msg_type, content = item
                    placeholder = st.session_state.get("learning_stream_placeholder")
                    if placeholder is not None:
                        with placeholder.container():
                            if msg_type == "USER_TEXT":
                                # 사용자의 입력도 사이드바 대화에 표시
                                st.chat_message("user").markdown(content)
                            elif msg_type == "TEXT":
                                # 튜터(학습 AI 에이전트)의 조언 스트리밍
                                sidebar_text_buffer += content
                                st.chat_message(
                                    "assistant",
                                    avatar=URL_PROF_IMAGE,
                                ).markdown(sidebar_text_buffer)

            except Empty:
                pass

        time.sleep(0.01)


# ============================================================
# CSV Agent용 프롬프트 헬퍼
# ============================================================

def get_combined_prefix_prompt(learning_agent: LearningAgentRunner) -> str:
    """
    Stage 1에서 생성된 프롬프트와 기본 prefix 프롬프트를 결합합니다.

    우선순위:
    1) 사용자가 Stage1 사이드바에서 수정한 프롬프트 (스레드별)
    2) LearningAgent state에 저장된 generated_prompt
    3) 기본 AGENT_PREFIX_PROMPT
    """
    # 스레드별 키 생성 (공통 헬퍼 사용)
    edit_key = get_thread_specific_key("sidebar_generated_prompt")

    # 1) Stage1 사이드바에서 수정된 프롬프트 (스레드별)
    edited_prompt = (st.session_state.get(edit_key) or "").strip()

    # 2) LangGraph 상태에 저장된 자동 생성 프롬프트
    state = learning_agent.get_state() if learning_agent is not None else None
    generated_prompt = (state.get("generated_prompt") or "").strip() if state else ""

    # 우선순위: 수정된 값 > 자동 생성 값
    base_prompt = edited_prompt or generated_prompt

    if base_prompt and AGENT_PREFIX_PROMPT:
        return f"{base_prompt}\n\n{AGENT_PREFIX_PROMPT}"
    elif base_prompt:
        return base_prompt
    else:
        return AGENT_PREFIX_PROMPT


# ============================================================
# CSV Agent 메인 UI (Stage2용)
# ============================================================

def render_csv_agent_execution_ui(learning_agent: LearningAgentRunner):
    """
    Stage2 메인 영역에 표시될 CSV 데이터 분석 Agent UI
    """
    # 데이터 업로드
    with st.expander("📁 데이터 업로드", expanded=True):
        uploaded_file = st.file_uploader(
            "CSV 파일을 업로드해주세요",
            type=["csv"],
            help="분석할 데이터를 CSV 형식으로 업로드해주세요.",
        )

        if uploaded_file is not None:
            try:
                dataframe = pd.read_csv(uploaded_file)

                if dataframe.empty or len(dataframe.columns) == 0:
                    st.error("❌ 업로드된 CSV 파일이 비어있거나 컬럼이 없습니다.")
                    st.session_state["uploaded_data"] = None
                else:
                    st.session_state["uploaded_data"] = dataframe
                    st.session_state["df_UploadData"] = dataframe
                    st.success(
                        f"✅ 파일 업로드 완료: {dataframe.shape[0]}행, {dataframe.shape[1]}열"
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("행 수", f"{dataframe.shape[0]:,}")
                    with col2:
                        st.metric("열 수", f"{dataframe.shape[1]:,}")
                    with col3:
                        missing_count = dataframe.isnull().sum().sum()
                        st.metric("총 결측값", f"{missing_count:,}")

                    with st.expander("데이터 미리보기", expanded=False):
                        safe_df = safe_dataframe_display(dataframe, max_rows=10)
                        st.dataframe(safe_df, use_container_width=True)

            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
                st.session_state["uploaded_data"] = None
                st.session_state["df_UploadData"] = None
                st.session_state["agent_with_UploadData"] = None

    # 시스템 설정
    with st.expander("⚙️ 시스템 설정", expanded=False):
        model_option_general = st.selectbox(
            label="🧠 모델 선택지",
            options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
            index=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"].index(
                st.session_state.get("model_option_general", "gpt-4.1")
            ),
            help="데이터 분석을 위한 OpenAI 모델을 설정합니다.",
        )

        if model_option_general != st.session_state["model_option_general"]:
            st.session_state["model_option_general"] = model_option_general
            st.success(f"모델 설정 완료: {model_option_general}")
            time.sleep(1)
            st.rerun()

        openai_api_key = st.text_input(
            "🔑 API 키",
            type="password",
            placeholder="OpenAI API 키를 입력하세요",
            help="데이터 분석 요청을 위한 OpenAI API 키를 설정합니다.",
            key="api_key_input_stage2",
        )
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API 키가 설정되었습니다!")
            st.session_state["api_key_updated"] = True
            time.sleep(1)

    # API 키 업데이트 시 에이전트 재생성
    if st.session_state.get("api_key_updated"):
        if st.session_state.get("df_UploadData") is not None:
            set_openai_api_key()

            # Stage1 프롬프트 + 기본 prefix 결합
            combined_prefix = get_combined_prefix_prompt(learning_agent)

            st.session_state["agent_with_UploadData"] = create_agent(
                dataframe=st.session_state["df_UploadData"],
                selected_model=st.session_state["model_option_general"],
                api_key=st.session_state["openai_api_key"],
                prefix_prompt=combined_prefix,
                postfix_prompt=AGENT_POSTFIX_PROMPT,
                shared_python_tool=st.session_state.get("python_tool", None),
            )
        st.session_state["api_key_updated"] = False

    # 데이터 분석 시작 버튼
    uploaded_file_present = st.session_state.get("uploaded_data") is not None
    df_present = st.session_state.get("df_UploadData") is not None
    agent_exists = "agent_with_UploadData" in st.session_state

    if uploaded_file_present and df_present and not agent_exists:
        apply_btn = st.button(
            "🚀 데이터 분석 시작", type="primary", use_container_width=True
        )

        if apply_btn:
            loaded_data = st.session_state["uploaded_data"]

            st.session_state["messages_UploadData"] = []
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None

            if conversation_manager:
                user_id = st.session_state.get("username", "anonymous")

                current_conversation = conversation_manager.get_or_create_conversation(
                    st.session_state.get("session_uuid"),
                    user_id,
                )
                if current_conversation:
                    st.session_state["conversation_id"] = current_conversation[
                        "session_uuid"
                    ]

            st.success(
                f"📊 데이터 로딩 완료: **{loaded_data.shape[0]}개 항목**, **{loaded_data.shape[1]}개 피처**"
            )
            st.success(
                f"컬럼 정보: **{', '.join(loaded_data.columns.tolist())}**"
            )

            st.subheader("📋 업로드된 데이터 보기")
            add_message_main(
                MessageRole.ASSISTANT, [MessageType.DATAFRAME, loaded_data.head(10)]
            )

            st.session_state["df_UploadData"] = loaded_data
            st.session_state["python_tool"] = PythonAstREPLTool()
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None

            st.session_state["python_tool"].locals["df_UploadData"] = loaded_data
            st.session_state["python_tool"].locals["df"] = loaded_data
            st.session_state["python_tool"].locals["pd"] = pd
            st.session_state["python_tool"].locals["np"] = np
            st.session_state["python_tool"].locals["plt"] = plt
            st.session_state["python_tool"].locals["sns"] = sns

            st.session_state["python_tool"].locals["data_info"] = {
                "shape": loaded_data.shape,
                "columns": loaded_data.columns.tolist(),
                "dtypes": loaded_data.dtypes.to_dict(),
            }

            set_openai_api_key()

            # Stage1 프롬프트 + 기본 prefix 결합
            combined_prefix = get_combined_prefix_prompt(learning_agent)

            st.session_state["agent_with_UploadData"] = create_agent(
                dataframe=loaded_data,
                selected_model=st.session_state["model_option_general"],
                prefix_prompt=combined_prefix,
                postfix_prompt=AGENT_POSTFIX_PROMPT,
                shared_python_tool=st.session_state.get("python_tool", None),
            )

    else:
        if st.session_state.get("uploaded_data") is None:
            st.info("📤 CSV 파일을 업로드해주세요.")

    # 채팅 인터페이스
    if (
        "agent_with_UploadData" in st.session_state
        or st.session_state.get("messages_UploadData")
    ):
        # 에이전트 복원 (이전 대화 기록만 있고 에이전트가 없을 경우)
        if "agent_with_UploadData" not in st.session_state and st.session_state.get(
            "messages_UploadData"
        ):
            for role, content_list in st.session_state["messages_UploadData"]:
                for content in content_list:
                    if isinstance(content, list) and len(content) == 2:
                        message_type, message_content = content
                        if (
                            message_type == MessageType.DATAFRAME
                            and isinstance(message_content, pd.DataFrame)
                        ):
                            st.session_state["df_UploadData"] = message_content
                            st.session_state["uploaded_data"] = message_content

                            if "python_tool" not in st.session_state:
                                st.session_state["python_tool"] = PythonAstREPLTool()

                            st.session_state["python_tool"].locals[
                                "df_UploadData"
                            ] = message_content
                            st.session_state["python_tool"].locals[
                                "df"
                            ] = message_content
                            st.session_state["python_tool"].locals["pd"] = pd
                            st.session_state["python_tool"].locals["np"] = np
                            st.session_state["python_tool"].locals["plt"] = plt
                            st.session_state["python_tool"].locals["sns"] = sns

                            # Stage1 프롬프트 + 기본 prefix 결합
                            combined_prefix = get_combined_prefix_prompt(learning_agent)

                            st.session_state["agent_with_UploadData"] = create_agent(
                                message_content,
                                st.session_state["model_option_general"],
                                prefix_prompt=combined_prefix,
                                postfix_prompt=AGENT_POSTFIX_PROMPT,
                                shared_python_tool=st.session_state.get(
                                    "python_tool", None
                                ),
                            )
                            break

        print_messages_main()

        user_input = st.chat_input("데이터에 질문해보세요!")

        if user_input:
            # 1) Learning Agent Stage2용 턴 증가 + user 텍스트 로그
            try:
                if "learning_turn" not in st.session_state:
                    st.session_state["learning_turn"] = 0
                st.session_state["learning_turn"] += 1
                current_turn = st.session_state["learning_turn"]

            except Exception as e:
                print(f"[Stage2] LearningAgent 사용자 입력 로그 저장 실패: {e}")

            # 2) CSV Agent UI + 로그 (learning_turn이 증가된 상태에서 turn을 함께 저장)
            st.chat_message("user").write(user_input)
            add_message_main(MessageRole.USER, [MessageType.TEXT, user_input])

            # 3) Agency DB에 사용자 입력 저장
            save_user_input_to_agency(user_input)

            # 4) CSV Agent + LearningAgent 병렬 실행
            session_state_proxy = {
                "model_option_general": st.session_state.get("model_option_general"),
                "last_result_df": st.session_state.get("last_result_df"),
                "last_figure": st.session_state.get("last_figure"),
            }

            with st.spinner("AI가 생각하고 있습니다..."):
                main_queue, sidebar_queue, error_queue = parallel_execution_manager(
                    user_input,
                    st.session_state["agent_with_UploadData"],
                    st.session_state["python_tool"],
                    learning_agent,
                    add_message_main,
                    session_state_proxy,
                )

                with st.chat_message("assistant"):
                    main_container = st.container()

                update_ui_from_queues(
                    main_queue,
                    sidebar_queue,
                    error_queue,
                    main_container,
                    add_message_main,
                )

                with main_container:
                    if st.session_state.get("last_result_df") is not None:
                        result_df = st.session_state["last_result_df"]
                        if isinstance(result_df, (pd.DataFrame, pd.Series)):
                            st.dataframe(result_df, use_container_width=True)
                            add_message_main(
                                MessageRole.ASSISTANT,
                                [MessageType.DATAFRAME, result_df],
                            )
                            st.session_state["last_result_df"] = None

                    if st.session_state.get("last_figure") is not None:
                        fig = st.session_state["last_figure"]
                        if isinstance(fig, plt.Figure):
                            st.pyplot(fig)
                            add_message_main(
                                MessageRole.ASSISTANT, [MessageType.FIGURE, fig]
                            )
                            plt.close(fig)
                            st.session_state["last_figure"] = None

            st.success("✅ 분석 완료!")


# ============================================================
# Stage 2 메인 (LearningAgent + CSV_Agent 통합)
# ============================================================

def main():
    st.set_page_config(
        page_title="Stage 2: 실행",
        page_icon="🚀",
        layout="wide"
    )

    # 로그인 여부 확인 (로그인하지 않았다면 접근 차단)
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("⚠️ 이 페이지는 로그인이 필요한 페이지입니다.")
        st.info("👈 상단 메뉴에서 '메인 페이지'로 이동해 로그인 후 다시 시도해주세요.")
        st.stop()

    st.title("🚀 Stage 2: 데이터 분석 실행")
    
    st.markdown("""
    **Learning AI Agent - 실행 단계**
    
    이제 계획한 대로 데이터 분석을 실행해봅시다!  
    **좌측 사이드바에서** 튜터와 대화하며 분석을 진행하고, 완료되면 **"완료"** 버튼을 클릭하세요.
    """)
    
    # Runner 초기화
    agent = LearningAgentRunner()

    # Learning Agent 공통 Conversation 보장
    ensure_learning_conversation()

    # 메인 영역
    state = agent.get_state()
    
    if not state or not state.get("planning_completed"):
        st.warning("⚠️ Stage 1: 계획을 먼저 완료하세요")
        st.info("👈 좌측 사이드바에서 'Stage 1: 계획'으로 이동하세요")
        
    else:
        current_phase = state.get("phase", "planning")
        execution_completed = state.get("execution_completed", False)
        planning_info = state.get("planning_info")

        # Stage 1에서 완성된 시스템 프롬프트 표시 (사용자 수정본 우선, 없으면 자동 생성본)
        thread_id = st.session_state.get("learning_thread_id", "")
        edit_key = get_thread_specific_key("sidebar_generated_prompt")
        edited_prompt = (st.session_state.get(edit_key) or "").strip()
        generated_prompt = (state.get("generated_prompt") or "").strip()
        display_prompt = edited_prompt or generated_prompt

        if display_prompt:
            with st.expander("🏷️ Stage 1에서 완성된 시스템 프롬프트", expanded=False):
                st.text_area(
                    "Stage 1 시스템 프롬프트",
                    value=display_prompt,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"stage2_display_system_prompt_{thread_id}",  # 스레드별 키
                )
        else:
            with st.expander("🏷️ Stage 1에서 완성된 시스템 프롬프트", expanded=False):
                st.info("생성된 시스템 프롬프트가 없습니다. Stage 1을 다시 진행해 주세요.")
        
        # 실행 단계 안내
        if current_phase == "planning":
            st.info("✅ 계획 단계가 완료되었습니다. 좌측 사이드바에서 대화를 시작하세요!")
        elif current_phase == "execution":
            if execution_completed:
                st.success("✅ 실행 단계가 완료되었습니다!")
                st.info("👈️ 좌측 사이드바에서 'Stage 3: 반성'으로 이동하세요")
        elif current_phase == "reflection":
            st.success("✅ 실행 단계가 완료되었습니다!")
            st.info("👈️ 좌측 사이드바에서 'Stage 3: 반성'으로 이동하세요")
        
        st.divider()

        # ✅ Stage1 완료된 경우에만 CSV_Agent 메인 UI 활성화
        render_csv_agent_execution_ui(agent)

    # 🔁 메인 영역에서 LearningAgent state 가 업데이트된 뒤에 사이드바 렌더
    render_learning_sidebar(agent, current_page="stage2")

if __name__ == "__main__":
    main()
