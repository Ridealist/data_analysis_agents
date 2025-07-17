from typing import List, Union
from langchain_experimental.tools import PythonAstREPLTool
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

from src.components.keywords import create_keyword
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk

from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote.messages import stream_response  # 스트리밍 출력


##### 폰트 설정 #####
import platform


# OS 판단
current_os = platform.system()

if current_os == "Windows":
    # Windows 환경 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
    fontprop = fm.FontProperties(fname=font_path, size=12)
    plt.rc("font", family=fontprop.get_name())
elif current_os == "Darwin":  # macOS
    # Mac 환경 폰트 설정
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux 등 기타 OS
    # 기본 한글 폰트 설정 시도
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except:
        print("한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")

##### 마이너스 폰트 깨짐 방지 #####
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지

# API 키 및 프로젝트 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

## LangSmith 세팅
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
logging.langsmith("CSV Agent 챗봇 수원외고 0717")


# Streamlit 앱 설정
st.title("🎤 Kpop 음원순위 분석 Agent 💬")

# 세션 상태 초기화
if "messages_KpopRanking" not in st.session_state:
    st.session_state["messages_KpopRanking"] = []  # 대화 내용을 저장할 리스트 초기화

if "messages_sidebar_KpopRanking" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages_sidebar_KpopRanking"] = []

# DataFrame 로딩
FILE_PATH_KpopRanking = "data/kpop_ranking.csv"
LOADED_DATA_KpopRanking = pd.read_csv(filepath_or_buffer=FILE_PATH_KpopRanking, encoding='latin1')


# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


DEFAULT_SIDEBAR_TUTOR_MODEL = "gpt-4o-mini"
DEFAULT_DATA_ANALYSIS_MODEL = "gpt-4.1"

# 체인 생성
def create_chain(prompt_path="prompts/general_gpto3.yaml", model_name=DEFAULT_SIDEBAR_TUTOR_MODEL):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt(prompt_path, encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    output_parser = StrOutputParser()

    # 단계 8: 체인(Chain) 생성
    chain = prompt | llm | output_parser

    return chain

# 메시지 관련 함수
def print_messages_main():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages_KpopRanking"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def add_message_main(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages_KpopRanking"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


# 콜백 함수
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
            if query:
                df_in_result = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message_main(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df_KpopRanking" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="코드 출력", state="complete", expanded=False)

                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message_main(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                return result
            else:
                st.error(
                    "데이터프레임이 정의되지 않았습니다. CSV, XLSX, XLS 파일을 먼저 업로드해주세요."
                )
                return


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages_KpopRanking"][-1][
                1
            ].clear()  # 에러 발생 시 마지막 메시지 삭제


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    pass  # 현재는 아무 동작도 하지 않습니다


# 에이전트 생성 함수
def create_agent(
    dataframe,
    selected_model=DEFAULT_DATA_ANALYSIS_MODEL,
    prefix_prompt=None,
    postfix_prompt=None,
    user_column_guideline=None,
):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4.1-mini"

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    from dataanalysis import DataAnalysisAgent

    return DataAnalysisAgent(
        dataframe,
        model_name=selected_model,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        column_guideline=user_column_guideline,
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent_KpopRanking" in st.session_state:
        st.chat_message("user").write(query)
        add_message_main(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent_KpopRanking"]
        response = agent.stream(query, "abc123")

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            has_dataframe = False
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
                # DataFrame이 출력되었는지 확인
                if st.session_state["messages_KpopRanking"] and st.session_state["messages_KpopRanking"][-1][0] == MessageRole.ASSISTANT:
                    for content in st.session_state["messages_KpopRanking"][-1][1]:
                        if isinstance(content, list) and content[0] == MessageType.DATAFRAME:
                            has_dataframe = True
                            break
            
            # DataFrame이 출력되지 않은 경우에만 텍스트 답변 출력
            if not has_dataframe and ai_answer:
                st.write(ai_answer)
                add_message_main(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


###### Service Part ######
# selected_model = "gpt-4o"
# selected_prompt = "prompts/doc-rag-scaffold.yaml"
column_info = ", ".join(LOADED_DATA_KpopRanking.columns.tolist())
df_chain = create_chain()
if "df_chain_KpopRanking" not in st.session_state:
    st.session_state["df_chain_KpopRanking"] = df_chain

# 사이드바 설정
with st.sidebar:
    clear_btn = st.button("메인 윈도우 대화 초기화 ➡️")  # 대화 내용을 초기화하는 버튼


    st.text("추천질문")
    st.text("(추천 질문을 참고해 오른쪽 메인 화면에서 데이터 분석을 요청해보세요.)")
    col1, col2 = st.columns(2)
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()


    # selected_model = st.selectbox(
    #     "OpenAI 모델을 선택해주세요.",
    #     ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
    #     index=0,
    # )  # OpenAI 모델 선택 옵션

    st.text("AI와 대화하기")
    messages = st.container(height=400)


    # 이전 대화를 출력
    def print_messages_sidebar():
        for chat_message in st.session_state["messages_sidebar_KpopRanking"]:
            messages.chat_message(chat_message.role).write(chat_message.content)


    # 새로운 메시지를 추가
    def add_message_sidebar(role, message):
        st.session_state["messages_sidebar_KpopRanking"].append(ChatMessage(role=role, content=message))


    # # 초기화 버튼이 눌리면...
    # if clear_btn:
    #     st.session_state["messages_KpopRanking"] = []

    # 키워드 질문 처리
    kw_1, kw_2 = create_keyword(st.session_state["messages_sidebar_KpopRanking"])
    kw_button_1 = placeholder1.button(label=kw_1, use_container_width=True, help="클릭하면 또 다른 질문이 생성돼요")
    kw_button_2 = placeholder2.button(label=kw_2, use_container_width=True, help="클릭하면 또 다른 질문이 생성돼요")

    # 사용자의 입력
    user_input = st.chat_input("🤓분석도우미에게 궁금한 내용을 물어보세요!")

    # 대화 chain 설정
    chain = st.session_state["df_chain_KpopRanking"]

    # 이전 대화 기록 출력
    print_messages_sidebar()

    # if kw_button_1:
    #     user_input = kw_1

    #     # 사용자의 입력
    #     messages.chat_message("user").write(user_input)

    #     input = {
    #         "context": column_info,
    #         "question": user_input
    #     }

    #     # 답변 스트리밍 호출
    #     response = chain.stream(input)
    #     with messages.chat_message("assistant"):
    #         # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
    #         container = st.empty()

    #         ai_answer = ""
    #         for token in response:
    #             if isinstance(token, AIMessageChunk):
    #                 ai_answer += token.content
    #             elif isinstance(token, str):
    #                 ai_answer += token
    #             container.markdown(ai_answer)

    #     # 대화기록을 저장한다.
    #     add_message_sidebar("user", user_input)
    #     add_message_sidebar("assistant", ai_answer)


    #     kw_1, kw_2 = create_keyword(st.session_state["messages_KpopRanking_sidebar"])
    #     placeholder1.button(label=kw_1, use_container_width=True)
    #     placeholder2.button(label=kw_2, use_container_width=True)


    # if kw_button_2:
    #     user_input = kw_2

    #     # 사용자의 입력
    #     messages.chat_message("user").write(user_input)

    #     input = {
    #         "context": column_info,
    #         "question": user_input
    #     }

    #     # 답변 스트리밍 호출
    #     response = chain.stream(input)
    #     with messages.chat_message("assistant"):
    #         # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
    #         container = st.empty()

    #         ai_answer = ""
    #         for token in response:
    #             if isinstance(token, AIMessageChunk):
    #                 ai_answer += token.content
    #             elif isinstance(token, str):
    #                 ai_answer += token
    #             container.markdown(ai_answer)

    #     # 대화기록을 저장한다.
    #     add_message_sidebar("user", user_input)
    #     add_message_sidebar("assistant", ai_answer)

    #     kw_1, kw_2 = create_keyword(st.session_state["messages_KpopRanking_sidebar"])
    #     placeholder1.button(label=kw_1, use_container_width=True)
    #     placeholder2.button(label=kw_2, use_container_width=True)


    # 만약에 사용자 입력이 들어오면...
    if user_input:

        # 사용자의 입력
        messages.chat_message("user").write(user_input)

        input = {
            "context": column_info,
            "question": user_input
        }

        # 답변 스트리밍 호출
        response = chain.stream(input)

        with messages.chat_message("assistant"):

            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                if isinstance(token, AIMessageChunk):
                    ai_answer += token.content
                elif isinstance(token, str):
                    ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message_sidebar("user", user_input)
        add_message_sidebar("assistant", ai_answer)

        kw_1, kw_2 = create_keyword(st.session_state["messages_sidebar_KpopRanking"])
        placeholder1.button(label=kw_1, use_container_width=True)
        placeholder2.button(label=kw_2, use_container_width=True)


# 메인 로직
if clear_btn:
    st.session_state["messages_KpopRanking"] = []  # 대화 내용 초기화

apply_btn = st.button("데이터 분석 시작", type="primary")  # 데이터 분석을 시작하는 버튼


if apply_btn:

    loaded_data = LOADED_DATA_KpopRanking
    add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, loaded_data.head()])

    st.session_state["df_KpopRanking"] = loaded_data  # 데이터프레임 저장
    st.session_state["python_tool"] = PythonAstREPLTool()  # Python 실행 도구 생성
    st.session_state["python_tool"].locals[
        "df_KpopRanking"
    ] = loaded_data  # 데이터프레임을 Python 실행 환경에 추가
    st.session_state["agent_KpopRanking"] = create_agent(
        loaded_data,
        DEFAULT_DATA_ANALYSIS_MODEL,
        prefix_prompt=None,
        postfix_prompt=None,
    )  # 에이전트 생성

    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")

# elif apply_btn:
#     st.warning("파일을 업로드 해주세요.")

# if "agent" in st.session_state:
#     updated_column_guideline = txt_column_guideline.markdown(
#         f"**컬럼 가이드라인**\n\n```\n{st.session_state['agent'].column_guideline}\n```"
#     )

print_messages_main()  # 저장된 메시지 출력

user_input = st.chat_input("데이터에 질문해보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리

# print(f'MESSAGES: {st.session_state["messages_KpopRanking"]}')
# print(f'MESSAGES_KpopRanking_SIDEBAR: {st.session_state["messages_sidebar_KpopRanking"]}')
