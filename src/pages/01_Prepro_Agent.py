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

from langchain_teddynote import logging # 로깅 설정

# OpenAI 에러 처리 관련 import
from langchain_experimental.tools import PythonAstREPLTool
from openai import OpenAI, OpenAIError, InternalServerError

# 데이터베이스 관련 import
from src.components.dynamo import DynamoDBManager, S3Manager, ConversationManager, MessageManager, AgencyConversationManager
from src.components.dataclass import MessageRole, MessageType
from src.components.sidebar import render_conversation_history_preprocessing
from src.components.llm import set_openai_api_key, handle_openai_error
from src.prompts.agent_prompts import AGENT_POSTFIX_PROMPT
from src.utils.dataframe_handler import safe_dataframe_display

# Agency Tutor 관련 import
from src.agency.Tutor import AgencyTutor

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


##### API 키 및 프로젝트 설정 #####
load_dotenv()

# 초기 API 키 설정
if not set_openai_api_key():
    st.stop()

## LangSmith 세팅 (주석 처리)
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
logging.langsmith("Agent 챗봇 1차 현장적용[수원외고] 0828")

# --- 데이터베이스 설정 ---
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = st.secrets.get("AWS_S3_BUCKET_NAME")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-northeast-2")

# DynamoDB 테이블 이름
CONVERSATIONS_PROCESSING_TABLE = "chatbot_conversations_preprocessing"
MESSAGES_PROCESSING_TABLE = "chatbot_messages_preprocessing"
AGENCY_PROCESSING_TABLE = "agency_messages_preprocessing"

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
        if 'cleaned_df' in python_tool.locals:
            del python_tool.locals['cleaned_df']
        
        response_items = []
        executed_code = ""
        
        # 사용자 메시지 전송
        main_queue.put(("USER_MESSAGE", query))
        
        try:
            # Agent 스트림 실행
            for step in agent.stream(query, "def123"):
                response_items.append(step)
                
                # 코드 실행 감지
                if "actions" in step:
                    for action in step["actions"]:
                        if action.tool == "python_repl_tool":
                            executed_code = action.tool_input.get("code", "")
                            main_queue.put(("CODE", executed_code))
                
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
            
            # 전처리 에이전트의 cleaned_df 결과도 처리
            if 'cleaned_df' in python_tool.locals:
                cleaned_df = python_tool.locals['cleaned_df']
                if isinstance(cleaned_df, (pd.DataFrame, pd.Series)):
                    main_queue.put(("DATAFRAME", cleaned_df))
                    del python_tool.locals['cleaned_df']  # 중복 방지를 위해 삭제
                    
        except OpenAIError as e:
            success, new_model, error_msg = handle_openai_error(e, session_state_proxy.get("model_option_general"))
            
            if isinstance(e, InternalServerError):
                error_queue.put(("SERVER_ERROR", error_msg))
            else:
                error_queue.put(("API_ERROR", error_msg))
            
            if success and new_model:
                main_queue.put(("MODEL_SWITCH", (new_model, error_msg)))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
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


def thread_safe_thinking(query: str, thinking_chain, sidebar_messages_history, sidebar_queue: Queue, error_queue: Queue, conversation_id: str = None):
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
        
        # Agency 응답을 데이터베이스에 저장
        if response:
            save_agency_response(response, intent, assessment, conversation_id)
        
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
        # python_tool.locals에서 result_df와 cleaned_df 확인
        if 'result_df' in python_tool.locals:
            result_df = python_tool.locals['result_df']
            if isinstance(result_df, (pd.DataFrame, pd.Series)):
                main_queue.put(("DATAFRAME_FINAL", result_df))
                # 세션 상태에 저장
                st.session_state['last_result_df_preprocessing'] = result_df
        
        # 전처리 에이전트의 cleaned_df 결과도 처리
        if 'cleaned_df' in python_tool.locals:
            cleaned_df = python_tool.locals['cleaned_df']
            if isinstance(cleaned_df, (pd.DataFrame, pd.Series)):
                main_queue.put(("DATAFRAME_FINAL", cleaned_df))
                # 세션 상태에 저장 (전처리 결과도 저장)
                st.session_state['last_result_df_preprocessing'] = cleaned_df

    with ThreadPoolExecutor(max_workers=2) as executor:
        # ask 함수 실행
        executor.submit(wrapper_ask)
        
        # thinking chain 실행
        executor.submit(
            thread_safe_thinking,
            user_input,
            thinking_chain,
            st.session_state.get("messages_sidebar_UploadData_preprocessing"),
            sidebar_queue,
            error_queue,
            st.session_state.get("conversation_id")
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
                            
                        elif msg_type == "DATAFRAME":
                            st.dataframe(content, use_container_width=True)
                            add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, content])
                        
                        elif msg_type == "CODE":
                            with st.status("🔍 실행 가능한 전처리 코드 보기", expanded=False):
                                st.code(content, language="python")
                            add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, content])
                        
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
                            st.session_state['last_result_df_preprocessing'] = content
                            
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
                        st.session_state["messages_sidebar_UploadData_preprocessing"].append({
                            "role": "assistant", 
                            "content": sidebar_text_buffer
                        })
                else:
                    msg_type, content = item
                    
                    if msg_type == "TEXT":
                        sidebar_text_buffer += content
                        with sidebar_container:
                            # 세션 상태의 고정 placeholder만 업데이트 (새 요소 생성 방지)
                            if st.session_state.get("thinking_placeholder_preprocessing"):
                                with st.session_state["thinking_placeholder_preprocessing"].container():
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
    model = "gpt-4.1"

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
            conversations_table=CONVERSATIONS_PROCESSING_TABLE,
            messages_table=MESSAGES_PROCESSING_TABLE
        )
        
        # S3 매니저 초기화
        s3_manager = S3Manager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            bucket_name=AWS_S3_BUCKET_NAME
        )
        
        # 대화 및 메시지 매니저 초기화
        conversation_manager = ConversationManager(dynamo_manager, CONVERSATIONS_PROCESSING_TABLE)
        message_manager = MessageManager(dynamo_manager)
        
        # Agency 대화 매니저 초기화
        agency_conversation_manager = AgencyConversationManager(dynamo_manager, AGENCY_PROCESSING_TABLE)
        
        return dynamo_manager, s3_manager, conversation_manager, message_manager, agency_conversation_manager
    except Exception as e:
        st.error(f"매니저 초기화 실패: {e}")
        return None, None, None, None, None

# 매니저 인스턴스 가져오기
dynamo_manager, s3_manager, conversation_manager, message_manager, agency_conversation_manager = init_managers()

#----------------------------------------#
#         Streamlit Session State        #
#----------------------------------------#
if "messages_UploadData_preprocessing" not in st.session_state:
    st.session_state["messages_UploadData_preprocessing"] = []  # 대화 내용을 저장할 리스트 초기화

if "uploaded_data_preprocessing" not in st.session_state:
    st.session_state["uploaded_data_preprocessing"] = None

if "uploaded_file_signature_preprocessing" not in st.session_state:
    st.session_state["uploaded_file_signature_preprocessing"] = None

if "file_processed_preprocessing" not in st.session_state:
    st.session_state["file_processed_preprocessing"] = False

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# 데이터베이스 관련 세션 상태 추가
if "session_uuid" not in st.session_state:
    st.session_state["session_uuid"] = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None

# 사이드바 튜터 대화 기록 저장용 (튜터 대화만)
if "messages_sidebar_UploadData_preprocessing" not in st.session_state:
    st.session_state["messages_sidebar_UploadData_preprocessing"] = []

# 사이드바 thinking placeholder 저장용 (화면에 하나만 표시하기 위함)
if "thinking_placeholder_preprocessing" not in st.session_state:
    st.session_state["thinking_placeholder_preprocessing"] = None

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
                st.session_state.get("conversation_id")
            )
            if s3_path:
                db_content_type = 'dataframe_s3'
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
            # AI 답변인 경우 모델 정보도 함께 저장
            llm_model = None
            if role == MessageRole.ASSISTANT:
                llm_model = st.session_state.get("model_option_general", "gpt-4.1-mini")
            
            message_manager.save_message(
                st.session_state.get("conversation_id"),
                role,
                db_content_type,
                db_content,
                llm_model
            )
    
    # 세션 상태에는 원본 객체 그대로 저장 (S3 경로가 아닌)
    messages = st.session_state["messages_UploadData_preprocessing"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])


def save_agency_response(output: str, intent: str, assessment: str, conversation_id: str = None):
    """Agency 응답 (output, intent, assessment)을 데이터베이스에 저장"""
    conv_id = conversation_id or st.session_state.get("conversation_id")
    if conv_id and agency_conversation_manager:
        try:
            success = agency_conversation_manager.save_agency_response(
                conv_id,
                output,
                intent,
                assessment
            )
            if success:
                print(f"Agency 응답 저장 성공: {intent}")
            else:
                print("Agency 응답 저장 실패")
        except Exception as e:
            print(f"Agency 응답 저장 중 오류: {e}")


def save_user_input_to_agency(user_input: str):
    """사용자 입력을 agency_conversations 테이블에 저장"""
    conv_id = st.session_state.get("conversation_id")
    if conv_id and agency_conversation_manager:
        try:
            agency_conversation_manager.save_user_input(
                conv_id,
                user_input
            )
        except Exception as e:
            pass


def print_messages_main():
    """저장된 메시지를 화면에 출력하는 함수"""
    for role, content_list in st.session_state["messages_UploadData_preprocessing"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)
                    
                    elif message_type == MessageType.DATAFRAME:
                        # 로컬 DataFrame 객체만 출력
                        st.dataframe(message_content, use_container_width=True)
                    
                    elif message_type == MessageType.CODE:
                        # AI 응답 형식이 아닌 경우 st.expander 안에서 st.code로 표시
                        with st.expander("🔍 실행 가능한 전처리 코드 보기", expanded=False):
                            st.code(message_content, language="python")
                    
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
            tool_code_input = tool_input.get("code")
            if not tool_code_input:
                st.error("코드가 정의되지 않았습니다.")
                return

            with st.status("데이터 전처리 중...", expanded=False) as status:
                st.markdown(f"```python\n{tool_code_input}\n```")
                add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, tool_code_input])
                status.update(label="코드 출력", state="complete", expanded=False)
            
            # 전처리 결과 확인 (result_df 또는 cleaned_df)
            if 'result_df' in st.session_state.get("python_tool_preprocessing", {}).locals:
                result_df = st.session_state["python_tool_preprocessing"].locals['result_df']
                if isinstance(result_df, (pd.DataFrame, pd.Series)):
                    st.success("📊 전처리 결과:")
                    st.dataframe(result_df, use_container_width=True)
                    add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, result_df])
                    # 세션 상태에 저장
                    st.session_state['last_result_df_preprocessing'] = result_df
                    # 중복 방지를 위해 삭제
                    del st.session_state["python_tool_preprocessing"].locals['result_df']
            
            # 전처리 에이전트의 cleaned_df 결과도 확인
            if 'cleaned_df' in st.session_state.get("python_tool_preprocessing", {}).locals:
                cleaned_df = st.session_state["python_tool_preprocessing"].locals['cleaned_df']
                if isinstance(cleaned_df, (pd.DataFrame, pd.Series)):
                    st.success("🧹 정리된 데이터:")
                    st.dataframe(cleaned_df, use_container_width=True)
                    add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, cleaned_df])
                    # 세션 상태에 저장
                    st.session_state['last_result_df_preprocessing'] = cleaned_df
                    # 중복 방지를 위해 삭제
                    del st.session_state["python_tool_preprocessing"].locals['cleaned_df']


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
                if st.session_state["messages_UploadData_preprocessing"]:
                    st.session_state["messages_UploadData_preprocessing"][-1][1].clear()  # 에러 발생 시 마지막 메시지 삭제


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
        dataframe (pd.DataFrame): 전처리할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4.1"
        api_key (str, optional): OpenAI API 키. 기본값은 None (세션 상태에서 가져옴)

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    from src.components.preprocessing import DataPreprocessingAgent
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

    return DataPreprocessingAgent(
        dataframe,
        model_name=selected_model,
        api_key=api_key,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,    
        column_guideline=user_column_guideline,
        shared_python_tool=shared_tool if shared_python_tool is None else shared_python_tool,  # 공유 도구 전달
    )


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
    st.title("🏗️ 데이터 전처리 LLM Agent")

    #-------Sidebar 설정 ----------#
    with st.sidebar:
        st.caption(f"👤 {st.session_state['username']}님 로그인됨")
        # Thinking Chain 응답용 컨테이너
        st.header("🤔 학습 에이전트의 조언")
        thinking_container = st.container(height=400)
        
        # 고정 placeholder 생성 또는 재사용
        with thinking_container:
            if "thinking_placeholder" not in st.session_state or st.session_state["thinking_placeholder"] is None:
                st.session_state["thinking_placeholder"] = st.empty()
            
            # 항상 마지막 thinking 메시지만 표시 (대화 기록은 유지)
            messages = st.session_state.get("messages_sidebar_UploadData_preprocessing", [])
            if messages:
                # 하나의 placeholder를 통해서만 표시 (이전 내용 완전 교체)
                with st.session_state["thinking_placeholder_preprocessing"].container():
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
        render_conversation_history_preprocessing(s3_manager, conversation_manager, message_manager, MESSAGES_PROCESSING_TABLE)

    #-------메인 로직 설정 ----------#
    # 데이터 파일 업로드 섹션
    with st.expander("📁 데이터 업로드", expanded=True):
        uploaded_file = st.file_uploader(
            "데이터 파일을 업로드해주세요", 
            type=["csv", "xlsx", "xls"], 
            help="분석할 데이터를 CSV, Excel 형식으로 업로드해주세요."
        )

        if uploaded_file is not None:

            # 파일 변경 감지
            try:
                current_sig = f"{getattr(uploaded_file, 'name', '')}:{getattr(uploaded_file, 'size', None)}"
            except Exception:
                current_sig = getattr(uploaded_file, 'name', '')
            prev_sig = st.session_state.get("uploaded_file_signature_preprocessing")

            file_changed = (prev_sig is not None and current_sig != prev_sig)

            try:
                # 파일 확장자 확인
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                # 파일 형식에 따른 로딩
                if file_extension in ['xlsx', 'xls']:
                    # Excel 파일 로딩 - 더 안정적인 처리
                    try:
                        dataframe = pd.read_excel(uploaded_file, engine='openpyxl')
                        file_type = "Excel"
                        st.success(f"✅ Excel 파일을 성공적으로 읽었습니다.")
                    except Exception as excel_error:
                        st.error(f"❌ Excel 파일 읽기 실패: {str(excel_error)}")
                        st.info("💡 Excel 파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
                        st.session_state["uploaded_data_preprocessing"] = None
                        st.session_state["df_UploadData_preprocessing"] = None
                        st.session_state["agent_with_UploadData_preprocessing"] = None
                        
                else:
                    # CSV 파일 로딩 - 다양한 인코딩 시도
                    dataframe = None
                    file_type = "CSV"
                    
                    # 일반적인 인코딩 방식들을 순차적으로 시도
                    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']
                    
                    for encoding in encodings:
                        try:
                            # 파일 포인터를 처음으로 되돌리기
                            uploaded_file.seek(0)
                            dataframe = pd.read_csv(uploaded_file, encoding=encoding)
                            st.success(f"✅ {encoding} 인코딩으로 파일을 성공적으로 읽었습니다.")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            # 인코딩 외의 다른 오류인 경우
                            st.warning(f"⚠️ {encoding} 인코딩으로 읽기 실패: {str(e)}")
                            continue
                    
                    # 모든 인코딩이 실패한 경우
                    if dataframe is None:
                        st.error("❌ 지원되는 인코딩으로 파일을 읽을 수 없습니다.")
                        st.info("💡 파일을 UTF-8로 저장한 후 다시 업로드해주세요.")
                        st.session_state["uploaded_data_preprocessing"] = None
                        st.session_state["df_UploadData_preprocessing"] = None
                        st.session_state["agent_with_UploadData_preprocessing"] = None
                
                # 파일 포인터를 처음으로 되돌리기 (다음 사용을 위해)
                uploaded_file.seek(0)
                
                # 데이터프레임 유효성 검사
                if dataframe is not None and not dataframe.empty and len(dataframe.columns) > 0:
                    st.session_state["uploaded_data_preprocessing"] = dataframe
                    st.session_state["df_UploadData_preprocessing"] = dataframe
                    # 새로운 파일 업로드 시 처리 완료 플래그 리셋
                    st.session_state["file_processed_preprocessing"] = False
                    # 업로드 파일 시그니처 저장 (파일 변경 감지용)
                    try:
                        current_sig = f"{getattr(uploaded_file, 'name', '')}:{getattr(uploaded_file, 'size', None)}"
                    except Exception:
                        current_sig = getattr(uploaded_file, 'name', '')
                    st.session_state["uploaded_file_signature_preprocessing"] = current_sig
                    st.success(f"✅ {file_type} 파일 업로드 완료: {dataframe.shape[0]}행, {dataframe.shape[1]}열")
                    
                    # 데이터 정보 상세 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("행 수", f"{dataframe.shape[0]:,}")
                    with col2:
                        st.metric("열 수", f"{dataframe.shape[1]:,}")
                    with col3:
                        missing_count = dataframe.isnull().sum().sum()
                        st.metric("총 결측값", f"{missing_count:,}")
                    
                    # 파일 정보 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📄 파일 형식: {file_type}")
                    with col2:
                        if file_type == "CSV":
                            # 성공한 인코딩 표시
                            for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']:
                                try:
                                    uploaded_file.seek(0)
                                    test_df = pd.read_csv(uploaded_file, encoding=encoding)
                                    st.info(f"🔤 인코딩: {encoding}")
                                    break
                                except:
                                    continue
                    
                    # 데이터 미리보기
                    with st.expander("데이터 미리보기", expanded=False):
                        safe_df = safe_dataframe_display(dataframe, max_rows=10)
                        st.dataframe(safe_df, use_container_width=True)
                        
                        # 데이터 타입 정보도 표시 (Arrow 호환성 문제 완전 해결)
                        with st.expander("데이터 타입 정보", expanded=False):
                            try:
                                # 모든 데이터를 문자열로 변환하여 Arrow 호환성 문제 완전 해결
                                dtype_info = pd.DataFrame({
                                    '컬럼명': [str(col) for col in dataframe.columns],
                                    '데이터 타입': [str(dtype) for dtype in dataframe.dtypes],
                                    '결측값 수': [str(count) for count in dataframe.isnull().sum().values],
                                    '고유값 수': [str(count) for count in dataframe.nunique().values]
                                })
                                st.dataframe(dtype_info, use_container_width=True)
                            except Exception as e:
                                st.warning(f"데이터 타입 정보 표시 중 오류 발생: {str(e)}")
                                # 간단한 정보만 표시
                                st.write(f"총 {len(dataframe.columns)}개 컬럼, {dataframe.shape[0]}개 행")

            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
                st.info("💡 파일 형식이나 인코딩을 확인해주세요.")
                st.session_state["uploaded_data_preprocessing"] = None
                st.session_state["df_UploadData_preprocessing"] = None
                st.session_state["agent_with_UploadData_preprocessing"] = None                 

    with st.expander("⚙️ 시스템 설정", expanded=False):
        model_option_general = st.selectbox(
            label="🧠 모델 선택지",
            options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
            index=1,
            help="데이터 전처리를 위한 OpenAI 모델을 설정합니다."
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
            help="데이터 전처리 요청을 위한 OpenAI API 키를 설정합니다.",
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
        if st.session_state.get("df_UploadData_preprocessing") is not None:
            # 최신 키 적용
            set_openai_api_key()
            # 기존 에이전트 제거 후 재생성
            st.session_state["agent_with_UploadData_preprocessing"] = create_agent(
                dataframe=st.session_state["df_UploadData_preprocessing"],
                selected_model=st.session_state["model_option_general"],
                api_key=st.session_state["openai_api_key"],
                prefix_prompt=None,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                postfix_prompt=AGENT_POSTFIX_PROMPT,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                shared_python_tool=st.session_state.get("python_tool_preprocessing", None),
            )
        # 플래그 리셋
        st.session_state["api_key_updated"] = False

    # 데이터 분석 시작 버튼 (메인 화면)
    # 새로운 파일이 업로드되어 처리되지 않았거나(file_processed=False) 파일 시그니처가 바뀐 경우 버튼 노출
    if (uploaded_file is not None and 
        st.session_state.get("uploaded_data_preprocessing") is not None and 
        st.session_state.get("df_UploadData_preprocessing") is not None and
        "agent_with_UploadData_preprocessing" not in st.session_state):

        # 파일 변경 감지
        try:
            current_sig = f"{getattr(uploaded_file, 'name', '')}:{getattr(uploaded_file, 'size', None)}"
        except Exception:
            current_sig = getattr(uploaded_file, 'name', '')
        prev_sig = st.session_state.get("uploaded_file_signature_preprocessing")

        # print(f"prev_sig: {prev_sig}, current_sig: {current_sig}")
        # print(f"file_processed: {st.session_state.get('file_processed_preprocessing')}")

        file_changed = (prev_sig is not None and current_sig != prev_sig)
        # print(f"file_changed: {file_changed}")
        if file_changed or not st.session_state.get("file_processed_preprocessing"):
            apply_btn = st.button("🚀 데이터 전처리 시작", type="primary", use_container_width=True)

            if apply_btn:
                loaded_data = st.session_state["uploaded_data_preprocessing"]
                
                # 데이터베이스 대화 생성/조회
                if conversation_manager:
                    # 사용자 ID 가져오기 (로그인된 사용자 또는 기본값)
                    user_id = st.session_state.get('username', 'anonymous')
                    if user_id == 'anonymous':
                        warning_msg = "⚠️ 로그인하지 않은 사용자입니다. 'anonymous'로 저장됩니다."
                        st.warning(warning_msg)
                        add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, warning_msg])
                    
                    current_conversation = conversation_manager.get_or_create_conversation(
                        st.session_state.get("session_uuid"),
                        user_id
                    )
                    if current_conversation:
                        st.session_state["conversation_id"] = current_conversation["session_uuid"]

            
                # 데이터 정보 표시 (한글로)
                st.success(f"📊 데이터 로딩 완료: **{loaded_data.shape[0]}개 항목**, **{loaded_data.shape[1]}개 피처**")
                
                # 컬럼명을 문자열로 변환하여 안전하게 표시
                column_names = [str(col) for col in loaded_data.columns.tolist()]
                st.success(f"컬럼 정보: **{', '.join(column_names)}**")
                
                # 초기 데이터프레임 표시
                st.subheader("📋 업로드된 데이터 보기")
                add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, loaded_data.head(10)])

                # 세션 상태에 저장
                st.session_state["df_UploadData_preprocessing"] = loaded_data
                st.session_state["python_tool_preprocessing"] = PythonAstREPLTool() # Python 실행 도구 생성


                # Python 실행 도구 생성 (중요: 하나만 생성)
                st.session_state["python_tool_preprocessing"].locals["df_UploadData"] = loaded_data  # 데이터프레임을 Python 실행 환경에 추가
                st.session_state["python_tool_preprocessing"].locals["df"] = loaded_data  # df로도 접근 가능하게 추가
                st.session_state["python_tool_preprocessing"].locals["pd"] = pd  # pandas 추가
                st.session_state["python_tool_preprocessing"].locals["np"] = np  # numpy 추가
                st.session_state["python_tool_preprocessing"].locals["plt"] = plt  # matplotlib 추가
                st.session_state["python_tool_preprocessing"].locals["sns"] = sns  # seaborn 추가
                
                # 데이터프레임 정보도 추가 (안전한 형태로)
                st.session_state["python_tool_preprocessing"].locals["data_info"] = {
                    "shape": loaded_data.shape,
                    "columns": [str(col) for col in loaded_data.columns.tolist()],
                    "dtypes": {str(col): str(dtype) for col, dtype in loaded_data.dtypes.to_dict().items()}
                }

                # Agent 생성 - 같은 python_tool 사용하도록
                # API 키 재설정 (사이드바에서 변경된 경우)
                set_openai_api_key()
                
                st.session_state["agent_with_UploadData_preprocessing"] = create_agent(
                    dataframe=loaded_data,
                    selected_model=st.session_state["model_option_general"],
                    prefix_prompt=None,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                    postfix_prompt=AGENT_POSTFIX_PROMPT,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                    shared_python_tool=st.session_state.get("python_tool_preprocessing", None),  # 중요: 같은 도구 전달
                )  # 에이전트 생성
                
                # 파일 처리 완료 플래그 설정
                st.session_state["file_processed_preprocessing"] = True
                # 파일 시그니처 확정 저장
                st.session_state["uploaded_file_signature_preprocessing"] = current_sig

    else:
        # 에이전트가 생성되지 않고 이전 대화 기록도 없는 경우에만 안내 메시지 표시
        if st.session_state["uploaded_data_preprocessing"] is None:
            if uploaded_file is None:
                st.info("📤 CSV 파일을 업로드해주세요.")
            else:
                st.info("⚠️ 파일 업로드에 실패했습니다. 다시 시도해주세요.")

    #--------------------------------#
    #        병렬 처리 적용 부분        #
    #--------------------------------#
    # 메시지 출력 및 사용자 입력
    # 에이전트가 생성된 경우 또는 이전 대화 기록이 있는 경우
    if ("agent_with_UploadData_preprocessing" in st.session_state or 
        (st.session_state.get("messages_UploadData_preprocessing") and 
         len(st.session_state["messages_UploadData_preprocessing"]) > 0)):
        
        # 이전 대화 기록이 있지만 에이전트가 없는 경우, 에이전트 상태 복원 시도
        if "agent_with_UploadData_preprocessing" not in st.session_state and st.session_state.get("messages_UploadData_preprocessing"):
            # 마지막 데이터프레임을 찾아서 에이전트 상태 복원
            for role, content_list in st.session_state["messages_UploadData_preprocessing"]:
                for content in content_list:
                    if isinstance(content, list) and len(content) == 2:
                        message_type, message_content = content
                        if message_type == MessageType.DATAFRAME and isinstance(message_content, pd.DataFrame):
                            # 데이터프레임 복원
                            st.session_state["df_UploadData_preprocessing"] = message_content
                            st.session_state["uploaded_data_preprocessing"] = message_content
                            
                            # Python 도구 복원
                            if "python_tool_preprocessing" not in st.session_state:
                                st.session_state["python_tool_preprocessing"] = PythonAstREPLTool()
                            
                            st.session_state["python_tool_preprocessing"].locals["df_UploadData"] = message_content
                            st.session_state["python_tool_preprocessing"].locals["df"] = message_content
                            st.session_state["python_tool_preprocessing"].locals["pd"] = pd
                            st.session_state["python_tool_preprocessing"].locals["np"] = np
                            st.session_state["python_tool_preprocessing"].locals["plt"] = plt
                            st.session_state["python_tool_preprocessing"].locals["sns"] = sns
                            
                            # 에이전트 복원
                            st.session_state["agent_with_UploadData_preprocessing"] = create_agent(
                                message_content,
                                st.session_state["model_option_general"],
                                prefix_prompt=None,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                                postfix_prompt=AGENT_POSTFIX_PROMPT,  # 전처리 에이전트는 일반 프롬프트 사용하지 않음
                                shared_python_tool=st.session_state.get("python_tool_preprocessing", None),
                            )
                            break
        
        # 메시지 출력 및 사용자 입력
        print_messages_main()  # 저장된 메시지 출력

        user_input = st.chat_input("데이터에 질문해보세요!")  # 사용자 입력 받기

        if user_input:
            # 사용자 메시지 표시
            st.chat_message("user").write(user_input)
            add_message_main(MessageRole.USER, [MessageType.TEXT, user_input])

            # Agency 테이블에 사용자 입력 저장
            save_user_input_to_agency(user_input)

            # 사이드바 대화 기록에서 'user'의 메시지 추가
            st.session_state["messages_sidebar_UploadData_preprocessing"].append({
                "role": "user",
                "content": user_input
            })
            
            # Thinking chain 생성
            thinking_chain = create_thinking_chain(st.session_state.get("openai_api_key"))
            
            # 세션 상태 프록시 생성
            session_state_proxy = {
                "model_option_general": st.session_state.get("model_option_general"),
            }

            with st.spinner("AI가 생각하고 있습니다..."):

                # 사이드바 완전 초기화 - 이전 표시 내용 모두 제거
                thinking_container.empty()

                # 새로운 고정 placeholder 생성
                with thinking_container:
                    st.session_state["thinking_placeholder_preprocessing"] = st.empty()

                # 병렬 실행
                main_queue, sidebar_queue, error_queue = parallel_execution_manager(
                    user_input,
                    st.session_state["agent_with_UploadData_preprocessing"],
                    st.session_state["python_tool_preprocessing"],
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

            st.success("✅ 전처리 완료!")

else:
    st.error("먼저 로그인을 해주세요!")
