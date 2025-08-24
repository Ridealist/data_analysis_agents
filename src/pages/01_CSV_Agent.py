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

from langchain_experimental.tools import PythonAstREPLTool

# from langchain_teddynote import logging # 로깅 설정
# from langchain_teddynote.messages import stream_response  # 스트리밍 출력

# OpenAI 에러 처리 관련 import
from openai import OpenAIError, InternalServerError

# 데이터베이스 관련 import
from src.components.dynamo import DynamoDBManager, S3Manager, ConversationManager, MessageManager
from src.components.dataclass import MessageRole, MessageType
from src.components.sidebar import render_system_setting, render_data_info, render_conversation_history
from src.components.llm import set_openai_api_key, handle_openai_error
from src.prompts.agent_prompts import AGENT_PREFIX_PROMPT, AGENT_POSTFIX_PROMPT

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

if "messages_sidebar_UploadData" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages_sidebar_UploadData"] = []

if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# 데이터베이스 관련 세션 상태 추가
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# LLM chain 모델 설정 세션 상태 추가
if "model_option_general" not in st.session_state:
    st.session_state["model_option_general"] = "gpt-4.1-mini"


# API 키 관련 세션 상태 추가
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


##### 모델 설정 #####
# 사이드바 하단에 현재 데이터 상태 표시
if st.session_state["logged_in"]:
    render_system_setting()

#--------------------------------#
#      LLM 체인 생성 함수          #
#--------------------------------#
##TODO 추후 SRL_Agent 로직 적용
# @st.cache_resource
# def create_chain(prompt_path="src/prompts/general_gpto3.yaml", model_name=st.session_state["model_option_general"]):
#     # 단계 6: 프롬프트 생성(Create Prompt)
#     # 프롬프트를 생성합니다.
#     prompt = load_prompt(prompt_path, encoding="utf-8")

#     # 단계 7: 언어모델(LLM) 생성
#     # 모델(LLM) 을 생성합니다.
#     llm = ChatOpenAI(model_name=model_name, temperature=1)

#     # 단계 8: 체인(Chain) 생성
#     chain = prompt | llm | StrOutputParser()

#     return chain

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
        st.sidebar.error("⚠️ 사이드바에서 API 키를 설정해주세요.")
        return None
    
    # API 키가 있는 경우 성공 메시지
    st.sidebar.success(f"✅ API 키 사용: {len(api_key)}자")

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
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent_with_UploadData" not in st.session_state:
        error_msg = "먼저 CSV 파일을 업로드하고 '데이터 분석 시작' 버튼을 클릭해주세요."
        st.error(error_msg)
        add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
        return

    st.chat_message("user").write(query)
    add_message_main(MessageRole.USER, [MessageType.TEXT, query])

    agent = st.session_state["agent_with_UploadData"]
    python_tool = st.session_state["python_tool"]

    # 이전 결과 초기화 (figure는 유지)
    if 'result_df' in python_tool.locals:
        del python_tool.locals['result_df']
    st.session_state['last_result_df'] = None
    st.session_state['last_figure'] = None

    # Agent 실행
    with st.spinner("AI가 생각하고 있습니다..."):
        response_items = []
        executed_code = ""
        
        try:
            # 스트림 응답 수집
            for step in agent.stream(query, "abc123"):
                response_items.append(step)
                
                # 코드 실행 감지
                if "actions" in step:
                    for action in step["actions"]:
                        if action.tool == "python_repl_tool":
                            executed_code = action.tool_input.get("code", "")
        except OpenAIError as e:
            # OpenAI 에러 처리
            success, new_model, error_msg = handle_openai_error(e, st.session_state["model_option_general"])
            
            # 500 에러는 서버 문제이므로 특별 처리
            if isinstance(e, InternalServerError):
                st.warning(error_msg)
                st.info("⏳ 서버 문제가 해결될 때까지 잠시 기다려주세요. (보통 몇 초 이내에 해결됩니다)")
                return
            
            if success and new_model:
                # 새로운 모델로 재시도
                st.warning(error_msg)
                st.info("🔄 새로운 모델로 질문을 다시 시도해주세요.")
                
                # 에이전트를 새로운 모델로 재생성
                try:
                    # API 키 재설정 (사이드바에서 변경된 경우)
                    set_openai_api_key()

                    # 에이전트 재생성
                    st.session_state["agent_with_UploadData"] = create_agent(
                        dataframe=st.session_state["df_UploadData"],
                        selected_model=new_model,
                        prefix_prompt=AGENT_PREFIX_PROMPT,
                        postfix_prompt=AGENT_POSTFIX_PROMPT,
                        shared_python_tool=st.session_state.get("python_tool", None),
                    )
                    st.success(f"✅ 에이전트가 {new_model} 모델로 재생성되었습니다!")
                    st.info("🔄 질문을 다시 시도해주세요.")
                except Exception as agent_error:
                    st.error(f"❌ 에이전트 재생성 실패: {str(agent_error)}")
                return
            else:
                # 모든 모델 실패 또는 모델 전환 불가
                st.error(error_msg)
                return
        except Exception as e:
            # 기타 에러 처리
            st.error(f"❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
            return

    # 응답 출력
    with st.chat_message("assistant"):
        # 1. 코드 표시
        if executed_code:
            with st.status("실행된 코드", expanded=False):
                st.code(executed_code, language="python")
            add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, executed_code])

        # 2. DataFrame 출력 (세션 상태에서 가져오기)
        if st.session_state.get('last_result_df') is not None:
            result_df = st.session_state['last_result_df']
            if isinstance(result_df, (pd.DataFrame, pd.Series)):
                st.success("📊 데이터 분석 결과:")
                st.dataframe(result_df, use_container_width=True)
                add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, result_df])
                st.session_state['last_result_df'] = None  # 사용 후 초기화

        # 3. 시각화 출력 (세션 상태에서 가져오기)
        if st.session_state.get('last_figure') is not None:
            fig = st.session_state['last_figure']
            st.pyplot(fig)
            add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
            plt.close(fig)  # 메모리 정리
            st.session_state['last_figure'] = None  # 사용 후 초기화

        # 4. 텍스트 답변
        ai_answer = ""
        for item in response_items:
            if "output" in item:
                ai_answer += item["output"]
        
        if ai_answer:
            st.write(ai_answer)
            add_message_main(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


#--------------------------------#
#         Service Part         #
#--------------------------------#

# 로그인 처리
# secrets.toml에서 사용자 정보 가져오기
users_info = st.secrets["users"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "logged_in" not in st.session_state or st.session_state.logged_in == False:
    st.error(f"👤 로그인이 필요합니다!")
    time.sleep(1.0)
    st.switch_page("src/pages/00_Main.py")


#--------------------------------#
#         Login 된 이후 로직         #
#--------------------------------#
if st.session_state["logged_in"]:

    # Main 레이아웃
    st.title("🕵🏻‍♂️ 데이터 분석 LLM Agent")


if st.session_state["logged_in"]:
    #-------Sidebar 설정 ----------#
    with st.sidebar:
        ### 파일 업로드 섹션
        st.header("📁 데이터 업로드")
        uploaded_file = st.file_uploader("CSV 파일을 업로드해주세요", type=["csv"], help="분석할 데이터를 CSV 형식으로 업로드해주세요.")

        if uploaded_file is not None:
            try:
                dataframe = pd.read_csv(uploaded_file)
                st.session_state["uploaded_data"] = dataframe
                st.success(f"✅ 파일 업로드 완료")
                    
            except Exception as e:
                error_msg = f"파일 읽기 오류: {str(e)}"
                st.error(error_msg)
                st.session_state["uploaded_data"] = None

        # 사이드바 하단에 현재 데이터 상태 표시
        if st.session_state["uploaded_data"] is not None:
            render_data_info(st.session_state["uploaded_data"])

        # 사이드바에 대화 기록 표시
        render_conversation_history(s3_manager, conversation_manager, message_manager)

    #-------메인 로직 설정 ----------#
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
    if uploaded_file is not None and st.session_state["uploaded_data"] is not None and "agent_with_UploadData" not in st.session_state:
        # 업로드된 데이터 미리보기
        st.subheader("📋 업로드된 데이터 미리보기")
        with st.expander("데이터 미리보기", expanded=False):
            st.dataframe(st.session_state["uploaded_data"].head(10), use_container_width=True)
        
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
                st.info("📤 사이드바에서 CSV 파일을 업로드해주세요.")
            else:
                st.info("⚠️ 파일 업로드에 실패했습니다. 다시 시도해주세요.")

    # 메시지 출력 및 사용자 입력
    # 에이전트가 생성된 경우 또는 이전 대화 기록이 있는 경우
    if ("agent_with_UploadData" in st.session_state or 
        (st.session_state.get("messages_UploadData") and len(st.session_state["messages_UploadData"]) > 0)):
        
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
                                prefix_prompt="""
    이 AI는 데이터 분석을 도와주는 도우미입니다.
    - 어려운 용어나 복잡한 개념 대신 쉽고 친근한 말로 설명해주세요
    - 분석 결과를 실생활과 연결해서 설명해주세요
    - 궁금한 점이 있으면 언제든지 물어보세요!

    **중요한 시각화 규칙**:
    - 그래프를 그릴 때는 한글 폰트 설정을 하지 마세요 (plt.rcParams['font.family'] = ... 금지)
    - 한글 폰트는 이미 전역으로 설정되어 있으니 바로 matplotlib과 seaborn을 사용하세요
    - 시각화 코드에서 폰트 관련 설정은 절대 포함하지 마세요

    **중요한 코드 작성 규칙**:
    - 함수를 정의할 때 함수 외부의 변수를 참조하지 마세요
    - 함수 내부에서 필요한 모든 변수를 정의하거나 파라미터로 전달하세요
    - apply() 함수를 사용할 때는 가능한 람다 함수를 사용하세요
    - 예시:
    # 잘못된 방법:
    # keywords = ['a', 'b']
    # def check(x):
    #     return any(k in x for k in keywords)  # 외부 변수 참조
    
    # 올바른 방법 1 - 함수 내부 정의:
    # def check(x):
    #     keywords = ['a', 'b']  # 함수 내부에 정의
    #     return any(k in x for k in keywords)
    
    # 올바른 방법 2 - 람다 사용:
    # df['col'].apply(lambda x: '분류1' if 'a' in x else '분류2')""",
                                postfix_prompt="""
    <출력 형식>
    분석 결과를 설명할 때는 다음 형식을 따라주세요:

    - **핵심 내용**: (한 줄로 핵심 내용을 요약해주세요)
    - **상세 설명**: (핵심 내용에 대한 상세 설명을 3문장이 넘지 않게, 고등학생이 이해하기 쉽게 구체적으로 설명해주세요)

    답변은 친근하고 격려하는 톤으로 작성해주세요.
    </출력 형식>""",
                                shared_python_tool=st.session_state.get("python_tool", None),
                            )
                            break
        
        # 메시지 출력 및 사용자 입력
        print_messages_main()  # 저장된 메시지 출력

        user_input = st.chat_input("데이터에 질문해보세요!")  # 사용자 입력 받기
        if user_input:
            ask(user_input)  # 사용자 질문 처리

else:
    st.error("먼저 로그인을 해주세요!")
