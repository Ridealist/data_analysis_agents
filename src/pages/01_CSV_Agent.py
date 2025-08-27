from typing import List, Union
from dotenv import load_dotenv
import streamlit as st
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
from threading import Thread

# from langchain_teddynote import logging # 로깅 설정
# from langchain_teddynote.messages import stream_response  # 스트리밍 출력

# OpenAI 에러 처리 관련 import
from langchain_experimental.tools import PythonAstREPLTool
from openai import OpenAI, OpenAIError, InternalServerError

# 데이터베이스 관련 import
from src.components.dynamo import DynamoDBManager, S3Manager, ConversationManager, MessageManager
from src.components.dataclass import MessageRole, MessageType
from src.components.sidebar import render_conversation_history
from src.components.llm import set_openai_api_key, handle_openai_error
from src.prompts.agent_prompts import AGENT_PREFIX_PROMPT, AGENT_POSTFIX_PROMPT
from src.utils.dataframe_handler import safe_dataframe_display


from src.agency.Tutor import AgencyTutor

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


##### API 키 및 프로젝트 설정 #####
load_dotenv()

# 초기 API 키 설정
if not set_openai_api_key():
    st.stop()

# ## LangSmith 세팅 (주석 처리)
# os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
# os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
# os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# logging.langsmith("Agent 챗봇 학교급식[창의성] 0813")

# --- 데이터베이스 설정 ---
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = st.secrets.get("AWS_S3_BUCKET_NAME")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-northeast-2")

# DynamoDB 테이블 이름
CONVERSATIONS_TABLE = st.secrets.get("DYNAMODB_CONVERSATIONS_TABLE", "chatbot_conversations")
MESSAGES_TABLE = st.secrets.get("DYNAMODB_MESSAGES_TABLE", "chatbot_messages")

URL_PROF_IMAGE = './data/teacher.png'

#--------------------------------#
#      병렬 처리 함수 추가         #
#--------------------------------#

def thread_safe_ask(query: str, agent, python_tool, main_queue: Queue, error_queue: Queue, 
                   add_message_func, session_state_proxy: dict):
    """
    ask 함수를 Thread-safe하게 Queue로 스트리밍
    
    Args:
        query: 사용자 질문
        agent: DataAnalysisAgent
        python_tool: Python 실행 도구
        main_queue: 메인 출력 Queue
        error_queue: 에러 Queue
        add_message_func: 메시지 저장 함수
        session_state_proxy: 세션 상태 프록시
    """
    try:
        # 이전 결과 초기화
        if 'result_df' in python_tool.locals:
            del python_tool.locals['result_df']
        
        response_items = []
        executed_code = ""
        
        # 사용자 메시지 전송
        main_queue.put(("USER_MESSAGE", query))
        
        try:
            # Agent 스트림 실행
            for step in agent.stream(query, "abc123"):
                response_items.append(step)
                
                # 코드 실행 감지
                if "actions" in step:
                    for action in step["actions"]:
                        if action.tool == "python_repl_tool":
                            executed_code = action.tool_input.get("code", "")
                            main_queue.put(("CODE", executed_code))
                            
                            # 시각화 코드 처리
                            vis_keywords = ["plt.", "sns.", "matplotlib", "seaborn", "plt.show()"]
                            if any(keyword in executed_code for keyword in vis_keywords):
                                try:
                                    safe_code = executed_code.replace("plt.show()", "")
                                    python_tool.invoke({"query": safe_code})
                                    
                                    # Figure 생성 확인
                                    fig = session_state_proxy.get('last_figure')
                                    if fig and isinstance(fig, plt.Figure):
                                        main_queue.put(("FIGURE", fig))
                                        
                                except Exception as e:
                                    error_msg = f"시각화 처리 중 오류 발생: {str(e)}"
                                    main_queue.put(("ERROR", error_msg))
                
                # DataFrame 결과 처리
                if session_state_proxy.get('last_result_df') is not None:
                    result_df = session_state_proxy['last_result_df']
                    if isinstance(result_df, (pd.DataFrame, pd.Series)):
                        main_queue.put(("DATAFRAME", result_df))
                        session_state_proxy['last_result_df'] = None
                
                # 텍스트 스트리밍
                if "output" in step:
                    main_queue.put(("TEXT_CHUNK", step["output"]))
            
            # Agent 실행 후 result_df와 figure 체크
            # python_tool.locals에서 직접 확인
            if 'result_df' in python_tool.locals:
                result_df = python_tool.locals['result_df']
                if isinstance(result_df, (pd.DataFrame, pd.Series)):
                    main_queue.put(("DATAFRAME", result_df))
                    del python_tool.locals['result_df']  # 중복 방지를 위해 삭제
            
            # matplotlib의 현재 figure 확인
            import matplotlib.pyplot as plt
            if plt.get_fignums():  # 열려있는 figure가 있는지 확인
                fig = plt.gcf()
                if fig and fig.get_axes():
                    main_queue.put(("FIGURE", fig))
                    
        except OpenAIError as e:
            success, new_model, error_msg = handle_openai_error(e, session_state_proxy.get("model_option_general"))
            
            if isinstance(e, InternalServerError):
                error_queue.put(("SERVER_ERROR", error_msg))
            else:
                error_queue.put(("API_ERROR", error_msg))
            
            if success and new_model:
                main_queue.put(("MODEL_SWITCH", (new_model, error_msg)))
                
        except Exception as e:
            error_queue.put(("GENERAL_ERROR", str(e)))
        
        # 최종 텍스트 수집 및 전송
        final_text = ""
        for item in response_items:
            if "output" in item:
                final_text += item["output"]
        
        if final_text:
            main_queue.put(("FINAL_TEXT", final_text))
        
        # 스트림 종료
        main_queue.put(None)
        
    except Exception as e:
        error_queue.put(("CRITICAL_ERROR", str(e)))
        main_queue.put(None)


def thread_safe_thinking(query: str, thinking_chain, sidebar_messages_history, sidebar_queue: Queue, error_queue: Queue):
    """
    Thinking chain을 Thread-safe하게 실행
    
    Args:
        query: 사용자 질문
        thinking_chain: 생각할 거리 생성 체인
        sidebar_queue: 사이드바 출력 Queue
        error_queue: 에러 Queue
    """
    try:
        response = ""
        output, intent, assessment = thinking_chain.get_response_stream(sidebar_messages_history, query)

        for chunk in output:
            # ChatCompletionChunk에서 실제 텍스트만 추출
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    response += content
                    sidebar_queue.put(("TEXT", content))
            # 기존 fallback 로직 유지
            elif hasattr(chunk, 'content'):
                content = chunk.content
                if content:
                    response += content
                    sidebar_queue.put(("TEXT", content))
            # if hasattr(chunk, 'content'):
            #     content = chunk.content
            # else:
            #     content = str(chunk)
            
            # response += content
            # sidebar_queue.put(("TEXT", content))
        
        sidebar_queue.put(None)  # 종료 신호
        
    except Exception as e:
        error_queue.put(("THINKING_ERROR", str(e)))
        sidebar_queue.put(None)


def parallel_execution_manager(user_input: str, agent, python_tool, thinking_chain, 
                              add_message_func, session_state_proxy):
    """
    ask와 thinking을 병렬로 실행하는 관리자
    
    Returns:
        tuple: (main_queue, sidebar_queue, error_queue)
    """
    main_queue = Queue()
    sidebar_queue = Queue()
    error_queue = Queue()

    # 세션 상태에서 직접 가져오기 (Thread 실행 전)
    import streamlit as st
    
    # Thread 함수를 감싸는 wrapper 함수
    def wrapper_ask():
        # Thread 내부에서 session_state 값을 주기적으로 체크
        thread_safe_ask(
            user_input,
            agent,
            python_tool,
            main_queue,
            error_queue,
            add_message_func,
            session_state_proxy
        )
        
        # Agent 실행 완료 후 세션 상태 체크 (메인 스레드와 통신)
        # python_tool.locals에서 result_df 확인
        if 'result_df' in python_tool.locals:
            result_df = python_tool.locals['result_df']
            if isinstance(result_df, (pd.DataFrame, pd.Series)):
                main_queue.put(("DATAFRAME_FINAL", result_df))
                # 세션 상태에 저장
                st.session_state['last_result_df'] = result_df
        
        # matplotlib figure 확인
        import matplotlib.pyplot as plt
        if plt.get_fignums():
            fig = plt.gcf()
            if fig and fig.get_axes():
                main_queue.put(("FIGURE_FINAL", fig))
                # 세션 상태에 저장
                st.session_state['last_figure'] = fig

    with ThreadPoolExecutor(max_workers=2) as executor:
        # ask 함수 실행
        executor.submit(wrapper_ask)
        
        # thinking chain 실행
        executor.submit(
            thread_safe_thinking,
            user_input,
            thinking_chain,
            st.session_state["messages_sidebar_UploadData"],
            sidebar_queue,
            error_queue
        )
    
    return main_queue, sidebar_queue, error_queue


def update_ui_from_queues(main_queue: Queue, sidebar_queue: Queue, error_queue: Queue,
                          main_container, sidebar_container, add_message_main):
    """
    Queue에서 데이터를 가져와 UI 업데이트
    """
    main_done = False
    sidebar_done = False
    
    # 플레이스홀더 생성
    main_text_placeholder = None
    
    # 텍스트 버퍼
    main_text_buffer = ""
    sidebar_text_buffer = ""
    
    while not (main_done and sidebar_done):
        # 에러 체크
        if not error_queue.empty():
            try:
                error_type, error_msg = error_queue.get_nowait()
                
                if error_type == "SERVER_ERROR":
                    st.warning(error_msg)
                    st.info("⏳ 서버 문제가 해결될 때까지 잠시 기다려주세요.")
                elif error_type == "API_ERROR":
                    st.error(f"API 오류: {error_msg}")
                elif error_type == "MODEL_SWITCH":
                    new_model, msg = error_msg
                    st.warning(msg)
                    st.info(f"🔄 {new_model}로 다시 시도해주세요.")
                else:
                    st.error(f"오류 발생: {error_msg}")
                    
            except Empty:
                pass
        
        # 메인 큐 처리
        if not main_done:
            try:
                item = main_queue.get_nowait()
                
                if item is None:
                    main_done = True
                    # 최종 텍스트 저장
                    if main_text_buffer:
                        add_message_main(MessageRole.ASSISTANT, [MessageType.TEXT, main_text_buffer])
                else:
                    msg_type, content = item
                    
                    with main_container:
                        if msg_type == "USER_MESSAGE":
                            # 사용자 메시지는 이미 표시됨
                            pass
                            
                        elif msg_type == "CODE":
                            with st.status("🔍 실행된 코드 보기", expanded=False):
                                st.code(content, language="python")
                            add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, content])
                        
                        elif msg_type == "DATAFRAME":
                            st.dataframe(content, use_container_width=True)
                            add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, content])
                        
                        elif msg_type == "FIGURE":
                            st.pyplot(content)
                            add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, content])
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
                            add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, content])
                            # 세션 상태에 저장
                            st.session_state['last_result_df'] = content
                        
                        elif msg_type == "FIGURE_FINAL":
                            st.pyplot(content)
                            add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, content])
                            plt.close(content)
                            # 세션 상태에 저장
                            st.session_state['last_figure'] = content
                            
            except Empty:
                pass
        
        # 사이드바 큐 처리
        if not sidebar_done:
            try:
                item = sidebar_queue.get_nowait()
                
                if item is None:
                    sidebar_done = True
                    # 사이드바에 "tutor"의 메시지를 대화 기록에 추가 (기존 기록 유지)
                    # 화면에는 마지막 메시지만 표시되지만, 전체 대화 기록은 유지됨
                    if sidebar_text_buffer:
                        st.session_state["messages_sidebar_UploadData"].append({
                            "role": "assistant", 
                            "content": sidebar_text_buffer
                        })
                else:
                    msg_type, content = item
                    
                    if msg_type == "TEXT":
                        sidebar_text_buffer += content
                        with sidebar_container:
                            # 세션 상태의 고정 placeholder만 업데이트 (새 요소 생성 방지)
                            if st.session_state.get("thinking_placeholder"):
                                with st.session_state["thinking_placeholder"].container():
                                    with st.chat_message("assistant", avatar=URL_PROF_IMAGE):
                                        st.markdown(sidebar_text_buffer)
                            
            except Empty:
                pass
        
        # CPU 부하 감소
        time.sleep(0.01)


#--------------------------------#
#      Thinking Chain 생성       #
#--------------------------------#
@st.cache_resource
def create_thinking_chain(api_key):
    """생각할 거리 제공용 체인 생성"""
    
    client = OpenAI(api_key=api_key)
    model = "gpt-4o-2024-05-13"

    st.session_state['Tutor'] = AgencyTutor(client, model=model)
    
    return st.session_state['Tutor']


#--------------------------------#
#         매니저 초기화         #
#--------------------------------#
@st.cache_resource
def init_managers():
    """AWS 매니저들을 초기화하고 캐시"""
    try:
        # DynamoDB 매니저 초기화
        dynamo_manager = DynamoDBManager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            conversations_table=CONVERSATIONS_TABLE,
            messages_table=MESSAGES_TABLE
        )
        
        # S3 매니저 초기화
        s3_manager = S3Manager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            bucket_name=AWS_S3_BUCKET_NAME
        )
        
        # 대화 및 메시지 매니저 초기화
        conversation_manager = ConversationManager(dynamo_manager)
        message_manager = MessageManager(dynamo_manager)
        
        return dynamo_manager, s3_manager, conversation_manager, message_manager
    except Exception as e:
        st.error(f"매니저 초기화 실패: {e}")
        return None, None, None, None

# 매니저 인스턴스 가져오기
dynamo_manager, s3_manager, conversation_manager, message_manager = init_managers()

#----------------------------------------#
#         Streamlit Session State        #
#----------------------------------------#
if "messages_UploadData" not in st.session_state:
    st.session_state["messages_UploadData"] = []  # 대화 내용을 저장할 리스트 초기화

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# 데이터베이스 관련 세션 상태 추가
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# 사이드바 튜터 대화 기록 저장용 (튜터 대화만)
if "messages_sidebar_UploadData" not in st.session_state:
    st.session_state["messages_sidebar_UploadData"] = []

# 사이드바 thinking placeholder 저장용 (화면에 하나만 표시하기 위함)
if "thinking_placeholder" not in st.session_state:
    st.session_state["thinking_placeholder"] = None

# LLM chain 모델 설정 세션 상태 추가
if "model_option_general" not in st.session_state:
    st.session_state["model_option_general"] = "gpt-4.1-mini"

# API 키 관련 세션 상태 추가
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


#--------------------------------#
#         메시지 관련 함수         #
#--------------------------------#
def add_message_main(role: MessageRole, content: List[Union[MessageType, str]]):
    """새로운 메시지를 저장하는 함수 (DB 연동)"""
    message_type, message_content = content
    
    # DB 저장용 변수
    db_content_type = None
    db_content = None
    
    # S3 업로드 및 DB 저장 처리
    if st.session_state.conversation_id:
        if message_type == MessageType.DATAFRAME and isinstance(message_content, (pd.DataFrame, pd.Series)):
            # DataFrame을 S3에 업로드
            s3_path = s3_manager.upload_to_s3(
                message_content, 
                'dataframe', 
                st.session_state.conversation_id
            )
            if s3_path:
                db_content_type = 'dataframe_s3'
                db_content = s3_path
                # DB에만 S3 경로 저장, 세션에는 원본 객체 유지
        
        elif message_type == MessageType.FIGURE and hasattr(message_content, 'savefig'):
            # Figure를 S3에 업로드
            s3_path = s3_manager.upload_to_s3(
                message_content,
                'figure',
                st.session_state.conversation_id
            )
            if s3_path:
                db_content_type = 'figure_s3'
                db_content = s3_path
                # DB에만 S3 경로 저장, 세션에는 원본 객체 유지
        
        elif message_type == MessageType.TEXT:
            db_content_type = 'text'
            db_content = message_content
        
        elif message_type == MessageType.CODE:
            db_content_type = 'code'
            db_content = message_content
        
        elif message_type == MessageType.ERROR:
            db_content_type = 'error'
            db_content = message_content
        
        # DB에 저장
        if db_content_type and db_content:
            message_manager.save_message(
                st.session_state.conversation_id,
                role,
                db_content_type,
                db_content
            )
    
    # 세션 상태에는 원본 객체 그대로 저장 (S3 경로가 아닌)
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
                        # 로컬 Figure 객체만 출력
                        st.pyplot(message_content)
                    
                    elif message_type == MessageType.CODE:
                        # AI 응답 형식이 아닌 경우 st.expander 안에서 st.code로 표시
                        with st.expander("🔍 실행된 코드 보기", expanded=False):
                            st.code(message_content, language="python")
                    
                    elif message_type == MessageType.DATAFRAME:
                        # 로컬 DataFrame 객체만 출력
                        st.dataframe(message_content, use_container_width=True)
                    
                    elif message_type == MessageType.ERROR:
                        # ERROR 타입은 화면에 출력하지 않고 pass
                        pass
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")

#--------------------------------#
#         콜백 함수             #
#--------------------------------#
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict): 실행된 도구 정보
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_tool":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("code")
            if not query:
                st.error("코드가 정의되지 않았습니다.")
                return

            with st.status("데이터 분석 중...", expanded=True) as status:
                st.markdown(f"```python\n{query}\n```")
                add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, query])
                status.update(label="코드 출력", state="complete", expanded=False)
            
            # 시각화 코드 감지 및 자동 그래프 출력
            vis_keywords = ["plt.", "sns.", "matplotlib", "seaborn", "plt.show()"]
            if any(keyword in query for keyword in vis_keywords):
                try:
                    # plt.show() 제거하여 안전한 코드 생성
                    safe_code = query.replace("plt.show()", "")
                    
                    # 수정된 코드 실행
                    st.session_state["python_tool"].invoke({"query": safe_code})

                    # 생성된 figure 출력
                    fig = st.session_state["last_figure"]
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                        add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                    plt.close(fig)
                
                except Exception as e:
                    # 시각화 관련 오류는 사용자에게 치명적이지 않으므로 로그만 남김
                    error_msg = f"시각화 처리 중 오류 발생: {str(e)}"
                    print(f"[tool_callback:figure] {e}")
                    add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]

        # DataFrame 형태의 문자열인지 확인
        if isinstance(obs, str):
            # 에러가 아닌 경우 DataFrame 출력일 가능성 확인
            if "Error" in obs:
                # 에러 메시지를 ERROR 타입으로 저장 (화면에는 표시하지 않음)
                add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, obs])
                if st.session_state["messages_UploadData"]:
                    st.session_state["messages_UploadData"][-1][1].clear()  # 에러 발생 시 마지막 메시지 삭제


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    pass  # 현재는 아무 동작도 하지 않습니다


#--------------------------------#
#      에이전트 생성 함수       #
#--------------------------------#
def create_agent(
    dataframe,
    selected_model=st.session_state["model_option_general"],
    api_key=None,
    prefix_prompt=None,
    postfix_prompt=None,
    user_column_guideline=None,
    shared_python_tool=None,
):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4.1"
        api_key (str, optional): OpenAI API 키. 기본값은 None (세션 상태에서 가져옴)

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    from src.components.dataanalysis import DataAnalysisAgent

    # API 키가 전달되지 않은 경우 세션 상태에서 가져오기
    if api_key is None:
        # 먼저 세션 상태에서 가져오기
        api_key = st.session_state.get("openai_api_key")
        
        # 세션 상태에 없으면 환경 변수에서 가져오기
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
    
    # API 키가 여전히 없는 경우 에러
    if not api_key:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        st.sidebar.error("⚠️ 시스템 설정에서 API 키를 설정해주세요.")
        return None

    # 공유 python_tool 전달
    shared_tool = st.session_state.get("python_tool", None)

    return DataAnalysisAgent(
        dataframe,
        model_name=selected_model,
        api_key=api_key,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        column_guideline=user_column_guideline,
        shared_python_tool=shared_tool if shared_python_tool is None else shared_python_tool,  # 공유 도구 전달
    )


#--------------------------------#
#       질문 처리 함수          #
#--------------------------------#
# def ask(query):
#     """
#     사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

#     Args:
#         query (str): 사용자의 질문
#     """
#     if "agent_with_UploadData" not in st.session_state:
#         error_msg = "먼저 CSV 파일을 업로드하고 '데이터 분석 시작' 버튼을 클릭해주세요."
#         st.error(error_msg)
#         add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
#         return

#     st.chat_message("user").write(query)
#     add_message_main(MessageRole.USER, [MessageType.TEXT, query])

#     agent = st.session_state["agent_with_UploadData"]
#     python_tool = st.session_state["python_tool"]

#     # 이전 결과 초기화 (figure는 유지)
#     if 'result_df' in python_tool.locals:
#         del python_tool.locals['result_df']
#     st.session_state['last_result_df'] = None
#     st.session_state['last_figure'] = None

#     # Agent 실행
#     with st.spinner("AI가 생각하고 있습니다..."):
#         response_items = []
#         executed_code = ""
        
#         try:
#             # 스트림 응답 수집
#             for step in agent.stream(query, "abc123"):
#                 response_items.append(step)
                
#                 # 코드 실행 감지
#                 if "actions" in step:
#                     for action in step["actions"]:
#                         if action.tool == "python_repl_tool":
#                             executed_code = action.tool_input.get("code", "")
#         except OpenAIError as e:
#             # OpenAI 에러 처리
#             success, new_model, error_msg = handle_openai_error(e, st.session_state["model_option_general"])
            
#             # 500 에러는 서버 문제이므로 특별 처리
#             if isinstance(e, InternalServerError):
#                 st.warning(error_msg)
#                 st.info("⏳ 서버 문제가 해결될 때까지 잠시 기다려주세요. (보통 몇 초 이내에 해결됩니다)")
#                 return
            
#             if success and new_model:
#                 # 새로운 모델로 재시도
#                 st.warning(error_msg)
#                 st.info("🔄 새로운 모델로 질문을 다시 시도해주세요.")
                
#                 # 에이전트를 새로운 모델로 재생성
#                 try:
#                     # API 키 재설정 (사이드바에서 변경된 경우)
#                     set_openai_api_key()

#                     # 에이전트 재생성
#                     st.session_state["agent_with_UploadData"] = create_agent(
#                         dataframe=st.session_state["df_UploadData"],
#                         selected_model=new_model,
#                         prefix_prompt=AGENT_PREFIX_PROMPT,
#                         postfix_prompt=AGENT_POSTFIX_PROMPT,
#                         shared_python_tool=st.session_state.get("python_tool", None),
#                     )
#                     st.success(f"✅ 에이전트가 {new_model} 모델로 재생성되었습니다!")
#                     st.info("🔄 질문을 다시 시도해주세요.")
#                 except Exception as agent_error:
#                     st.error(f"❌ 에이전트 재생성 실패: {str(agent_error)}")
#                 return
#             else:
#                 # 모든 모델 실패 또는 모델 전환 불가
#                 st.error(error_msg)
#                 return
#         except Exception as e:
#             # 기타 에러 처리
#             st.error(f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
#             return

#     # 응답 출력
#     with st.chat_message("assistant"):
#         # 1. 코드 표시
#         if executed_code:
#             with st.status("🔍 실행된 코드 보기", expanded=False):
#                 st.code(executed_code, language="python")
#             add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, executed_code])

#         # 2. DataFrame 출력 (세션 상태에서 가져오기)
#         if st.session_state.get('last_result_df') is not None:
#             result_df = st.session_state['last_result_df']
#             if isinstance(result_df, (pd.DataFrame, pd.Series)):
#                 st.success("📊 데이터 분석 결과:")
#                 st.dataframe(result_df, use_container_width=True)
#                 add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, result_df])
#                 st.session_state['last_result_df'] = None  # 사용 후 초기화

#         # 3. 시각화 출력 (세션 상태에서 가져오기)
#         if st.session_state.get('last_figure') is not None:
#             fig = st.session_state['last_figure']
#             st.pyplot(fig)
#             add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
#             plt.close(fig)  # 메모리 정리
#             st.session_state['last_figure'] = None  # 사용 후 초기화

#         # 4. 텍스트 답변
#         ai_answer = ""
#         for item in response_items:
#             if "output" in item:
#                 ai_answer += item["output"]
        
#         if ai_answer:
#             st.write(ai_answer)
#             add_message_main(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


#--------------------------------#
#             메인 UI             #
#--------------------------------#

# 로그인 처리
# secrets.toml에서 사용자 정보 가져오기
users_info = st.secrets["users"]

if "logged_in" not in st.session_state or st.session_state.logged_in == False:
    st.error(f"👤 로그인이 필요합니다!")
    time.sleep(1.0)
    st.switch_page("src/pages/00_Main.py")


#--------------------------------#
#         Login 된 이후 로직       #
#--------------------------------#
if st.session_state["logged_in"]:

    # Main 레이아웃
    st.title("🕵🏻‍♂️ 데이터 분석 LLM Agent")

    #-------Sidebar 설정 ----------#
    with st.sidebar:
        #TODO chat 인터페이스 제거 검토
        # render_chat_interface()
        st.caption(f"👤 {st.session_state['username']}님 로그인됨")
        # Thinking Chain 응답용 컨테이너
        st.header("🤔 학습 에이전트의 조언")
        thinking_container = st.container(height=400)
        
        # 고정 placeholder 생성 또는 재사용
        with thinking_container:
            if "thinking_placeholder" not in st.session_state or st.session_state["thinking_placeholder"] is None:
                st.session_state["thinking_placeholder"] = st.empty()
            
            # 항상 마지막 thinking 메시지만 표시 (대화 기록은 유지)
            messages = st.session_state.get("messages_sidebar_UploadData", [])
            if messages:
                # 하나의 placeholder를 통해서만 표시 (이전 내용 완전 교체)
                with st.session_state["thinking_placeholder"].container():
                    last_msg = None
                    for msg in messages[::-1]:
                        if msg["role"] == "assistant":
                            last_msg = msg
                            break
                    if last_msg is None:
                        last_msg = messages[-1]
                    with st.chat_message(last_msg["role"], avatar=URL_PROF_IMAGE):
                        st.write(last_msg["content"])

        # 사이드바에 대화 기록 표시
        render_conversation_history(s3_manager, conversation_manager, message_manager)

    #-------메인 로직 설정 ----------#
    # CSV 파일 업로드 섹션
    with st.expander("📁 데이터 업로드", expanded=False):
        uploaded_file = st.file_uploader("CSV 파일을 업로드해주세요", type=["csv"], help="분석할 데이터를 CSV 형식으로 업로드해주세요.")

        if uploaded_file is not None:
            try:
                dataframe = pd.read_csv(uploaded_file)
                
                # 데이터프레임 유효성 검사
                if dataframe.empty or len(dataframe.columns) == 0:
                    st.error("❌ 업로드된 CSV 파일이 비어있거나 컬럼이 없습니다.")
                    st.session_state["uploaded_data"] = None
                else:
                    st.session_state["uploaded_data"] = dataframe
                    st.session_state["df_UploadData"] = dataframe
                    st.success(f"✅ 파일 업로드 완료: {dataframe.shape[0]}행, {dataframe.shape[1]}열")
                    
                    # 데이터 정보 상세 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("행 수", f"{dataframe.shape[0]:,}")
                    with col2:
                        st.metric("열 수", f"{dataframe.shape[1]:,}")
                    with col3:
                        missing_count = dataframe.isnull().sum().sum()
                        st.metric("총 결측값", f"{missing_count:,}")
                    
                    # 데이터 미리보기
                    with st.expander("데이터 미리보기", expanded=False):
                        safe_df = safe_dataframe_display(dataframe, max_rows=10)
                        st.dataframe(safe_df, use_container_width=True)

            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
                st.session_state["uploaded_data"] = None
                st.session_state["df_UploadData"] = None
                st.session_state["agent_with_UploadData"] = None                 


    with st.expander("⚙️ 시스템 설정", expanded=False):
        model_option_general = st.selectbox(
            label="🧠 모델 선택지",
            options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
            index=1,
            help="데이터 분석을 위한 OpenAI 모델을 설정합니다."
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
            key="api_key_input"
        )
        if openai_api_key:
            # 세션 상태에 저장
            st.session_state["openai_api_key"] = openai_api_key
            # 환경 변수에도 즉시 설정
            os.environ["OPENAI_API_KEY"] = openai_api_key
            # 성공 메시지 표시
            st.success("✅ API 키가 설정되었습니다!")
            # 에이전트 재생성을 위한 플래그 설정
            st.session_state["api_key_updated"] = True
            time.sleep(1)


    # API 키가 업데이트된 경우, 에이전트를 최신 키로 재생성
    if st.session_state.get("api_key_updated"):
        if st.session_state.get("df_UploadData") is not None:
            # 최신 키 적용
            set_openai_api_key()
            # 기존 에이전트 제거 후 재생성
            st.session_state["agent_with_UploadData"] = create_agent(
                dataframe=st.session_state["df_UploadData"],
                selected_model=st.session_state["model_option_general"],
                api_key=st.session_state["openai_api_key"],
                prefix_prompt=AGENT_PREFIX_PROMPT,
                postfix_prompt=AGENT_POSTFIX_PROMPT,
                shared_python_tool=st.session_state.get("python_tool", None),
            )
        # 플래그 리셋
        st.session_state["api_key_updated"] = False

    # 데이터 분석 시작 버튼 (메인 화면) - 에이전트가 생성되지 않은 경우에만 표시
    if (uploaded_file is not None and 
        st.session_state.get("uploaded_data") is not None and 
        st.session_state.get("df_UploadData") is not None and
        "agent_with_UploadData" not in st.session_state):
        
        apply_btn = st.button("🚀 데이터 분석 시작", type="primary", use_container_width=True)
        
        if apply_btn:
            loaded_data = st.session_state["uploaded_data"]
            
            # 데이터 분석 시작 전 상태 초기화
            st.session_state["messages_UploadData"] = []  # 대화 내용 초기화
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None
            
            # 데이터베이스 대화 생성/조회
            if conversation_manager:
                # 사용자 ID 가져오기 (로그인된 사용자 또는 기본값)
                user_id = st.session_state.get('username', 'anonymous')
                if user_id == 'anonymous':
                    warning_msg = "⚠️ 로그인하지 않은 사용자입니다. 'anonymous'로 저장됩니다."
                    st.warning(warning_msg)
                    add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, warning_msg])
                
                current_conversation = conversation_manager.get_or_create_conversation(
                    st.session_state.session_uuid,
                    user_id
                )
                if current_conversation:
                    st.session_state.conversation_id = current_conversation['session_uuid']

            
            # 데이터 정보 표시 (한글로)
            st.success(f"📊 데이터 로딩 완료: **{loaded_data.shape[0]}개 항목**, **{loaded_data.shape[1]}개 피처**")
            st.success(f"컬럼 정보: **{', '.join(loaded_data.columns.tolist())}**")
            
            # 초기 데이터프레임 표시
            st.subheader("📋 업로드된 데이터 보기")
            add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, loaded_data.head(10)])

            # 세션 상태에 저장
            st.session_state["df_UploadData"] = loaded_data
            st.session_state["python_tool"] = PythonAstREPLTool() # Python 실행 도구 생성
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None

            # Python 실행 도구 생성 (중요: 하나만 생성)
            st.session_state["python_tool"].locals["df_UploadData"] = loaded_data  # 데이터프레임을 Python 실행 환경에 추가
            st.session_state["python_tool"].locals["df"] = loaded_data  # df로도 접근 가능하게 추가
            st.session_state["python_tool"].locals["pd"] = pd  # pandas 추가
            st.session_state["python_tool"].locals["np"] = np  # numpy 추가
            st.session_state["python_tool"].locals["plt"] = plt  # matplotlib 추가
            st.session_state["python_tool"].locals["sns"] = sns  # seaborn 추가
            
            # 데이터프레임 정보도 추가
            st.session_state["python_tool"].locals["data_info"] = {
                "shape": loaded_data.shape,
                "columns": loaded_data.columns.tolist(),
                "dtypes": loaded_data.dtypes.to_dict()
            }

            # Agent 생성 - 같은 python_tool 사용하도록
            # API 키 재설정 (사이드바에서 변경된 경우)
            set_openai_api_key()
            
            st.session_state["agent_with_UploadData"] = create_agent(
                dataframe=loaded_data,
                selected_model=st.session_state["model_option_general"],
                prefix_prompt=AGENT_PREFIX_PROMPT,
                postfix_prompt=AGENT_POSTFIX_PROMPT,
                shared_python_tool=st.session_state.get("python_tool", None),  # 중요: 같은 도구 전달
            )  # 에이전트 생성

    else:
        # 에이전트가 생성되지 않고 이전 대화 기록도 없는 경우에만 안내 메시지 표시
        if st.session_state["uploaded_data"] is None:
            if uploaded_file is None:
                st.info("📤 CSV 파일을 업로드해주세요.")
            else:
                st.info("⚠️ 파일 업로드에 실패했습니다. 다시 시도해주세요.")

    #--------------------------------#
    #        병렬 처리 적용 부분        #
    #--------------------------------#
    # 메시지 출력 및 사용자 입력
    # 에이전트가 생성된 경우 또는 이전 대화 기록이 있는 경우
    if ("agent_with_UploadData" in st.session_state or 
        (st.session_state.get("messages_UploadData") and 
         len(st.session_state["messages_UploadData"]) > 0)):
        
        # 이전 대화 기록이 있지만 에이전트가 없는 경우, 에이전트 상태 복원 시도
        if "agent_with_UploadData" not in st.session_state and st.session_state.get("messages_UploadData"):
            # 마지막 데이터프레임을 찾아서 에이전트 상태 복원
            for role, content_list in st.session_state["messages_UploadData"]:
                for content in content_list:
                    if isinstance(content, list) and len(content) == 2:
                        message_type, message_content = content
                        if message_type == MessageType.DATAFRAME and isinstance(message_content, pd.DataFrame):
                            # 데이터프레임 복원
                            st.session_state["df_UploadData"] = message_content
                            st.session_state["uploaded_data"] = message_content
                            
                            # Python 도구 복원
                            if "python_tool" not in st.session_state:
                                st.session_state["python_tool"] = PythonAstREPLTool()
                            
                            st.session_state["python_tool"].locals["df_UploadData"] = message_content
                            st.session_state["python_tool"].locals["df"] = message_content
                            st.session_state["python_tool"].locals["pd"] = pd
                            st.session_state["python_tool"].locals["np"] = np
                            st.session_state["python_tool"].locals["plt"] = plt
                            st.session_state["python_tool"].locals["sns"] = sns
                            
                            # 에이전트 복원
                            st.session_state["agent_with_UploadData"] = create_agent(
                                message_content,
                                st.session_state["model_option_general"],
                                prefix_prompt=AGENT_PREFIX_PROMPT,
                                postfix_prompt=AGENT_POSTFIX_PROMPT,
                                shared_python_tool=st.session_state.get("python_tool", None),
                            )
                            break
        
        # 메시지 출력 및 사용자 입력
        print_messages_main()  # 저장된 메시지 출력

        user_input = st.chat_input("데이터에 질문해보세요!")  # 사용자 입력 받기

        if user_input:
            # 사용자 메시지 표시
            st.chat_message("user").write(user_input)
            add_message_main(MessageRole.USER, [MessageType.TEXT, user_input])

            # 사이드바 대화 기록에서 'user'의 메시지 추가
            st.session_state["messages_sidebar_UploadData"].append({
                "role": "user",
                "content": user_input
            })
            
            # Thinking chain 생성
            thinking_chain = create_thinking_chain(st.session_state.get("openai_api_key"))
            
            # 세션 상태 프록시 생성
            session_state_proxy = {
                "model_option_general": st.session_state.get("model_option_general"),
                "last_result_df": st.session_state.get('last_result_df'),
                "last_figure": st.session_state.get('last_figure')
            }

            with st.spinner("AI가 생각하고 있습니다..."):

                # 사이드바 완전 초기화 - 이전 표시 내용 모두 제거
                thinking_container.empty()

                # 새로운 고정 placeholder 생성
                with thinking_container:
                    st.session_state["thinking_placeholder"] = st.empty()

                # 병렬 실행
                main_queue, sidebar_queue, error_queue = parallel_execution_manager(
                    user_input,
                    st.session_state["agent_with_UploadData"],
                    st.session_state["python_tool"],
                    thinking_chain,
                    add_message_main,
                    session_state_proxy
                )
            
                # 응답 컨테이너 생성
                with st.chat_message("assistant"):
                    main_container = st.container()
                
                # 사이드바 컨테이너 초기화 (이전 내용 지우기)
                thinking_container.empty()
                with thinking_container:
                    sidebar_container = st.container()
                
                # UI 업데이트 (메인 스레드에서 실행)
                update_ui_from_queues(
                    main_queue,
                    sidebar_queue,
                    error_queue,
                    main_container,
                    sidebar_container,
                    add_message_main
                )

            # Agent 실행 완료 후 세션 상태에서 결과 확인 (메인 스레드에서)
            with main_container:
                # DataFrame 확인
                if st.session_state.get('last_result_df') is not None:
                    result_df = st.session_state['last_result_df']
                    if isinstance(result_df, (pd.DataFrame, pd.Series)):
                        st.dataframe(result_df, use_container_width=True)
                        add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, result_df])
                        st.session_state['last_result_df'] = None
                
                # Figure 확인
                if st.session_state.get('last_figure') is not None:
                    fig = st.session_state['last_figure']
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                        add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                        plt.close(fig)
                        st.session_state['last_figure'] = None

            st.success("✅ 분석 완료!")

else:
    st.error("먼저 로그인을 해주세요!")


# st.write(st.session_state)