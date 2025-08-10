from typing import List, Union
from langchain_experimental.tools import PythonAstREPLTool
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from src.components.keywords import create_keyword
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk

from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
# from langchain_teddynote.messages import stream_response  # 스트리밍 출력

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


##### Streamlit 시각화 설정 #####
def _st_show(*args, **kwargs):
    """plt.show()를 Streamlit-friendly 버전으로 교체"""
    # ❶ 스트림릿 시작 직후(예: import 뒤)에 넣어두기
    fig = plt.gcf()
    if fig and fig.axes:
        st.pyplot(fig)
    plt.close(fig)           # 메모리 누수 방지

plt.show = _st_show        # 원 함수 오버라이드


##### API 키 및 프로젝트 설정 #####
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


##### 모델 설정 #####
DEFAULT_SIDEBAR_TUTOR_MODEL = "gpt-4o-mini"
DEFAULT_DATA_ANALYSIS_MODEL = "gpt-4.1"

## LangSmith 세팅 (주석 처리)
# os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
# os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
# os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# logging.langsmith("CSV Agent 챗봇 학교급식 0717")

# Streamlit 앱 설정
st.title("학교 급식 분석 Agent 🍜")

# 세션 상태 초기화
if "messages_SchoolFood" not in st.session_state:
    st.session_state["messages_SchoolFood"] = []  # 대화 내용을 저장할 리스트 초기화

if "messages_sidebar_SchoolFood" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages_sidebar_SchoolFood"] = []


# 데이터 로딩
FILE_PATH_SchoolFood = "data/school_food.csv"
LOADED_DATA_SchoolFood = pd.read_csv(FILE_PATH_SchoolFood, index_col=0)


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

# 체인 생성
@st.cache_resource
def create_chain(prompt_path="prompts/general_gpto3.yaml", model_name=DEFAULT_SIDEBAR_TUTOR_MODEL):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt(prompt_path, encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=1)
    
    output_parser = StrOutputParser()

    # 단계 8: 체인(Chain) 생성
    chain = prompt | llm | output_parser

    return chain

# 메시지 관련 함수
def print_messages_main():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages_SchoolFood"]:
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
    messages = st.session_state["messages_SchoolFood"]
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
                    fig = plt.gcf()
                    if fig and fig.axes:
                        st.pyplot(fig)
                        add_message_main(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                    plt.close(fig)
                
                except Exception as e:
                    # 시각화 관련 오류는 사용자에게 치명적이지 않으므로 로그만 남김
                    print(f"[tool_callback:figure] {e}")


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
                st.error(obs)
                if st.session_state["messages_SchoolFood"]:
                    st.session_state["messages_SchoolFood"][-1][1].clear()  # 에러 발생 시 마지막 메시지 삭제


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
    shared_python_tool=None,
):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4.1"

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    from dataanalysis import DataAnalysisAgent

    # 공유 python_tool 전달
    shared_tool = st.session_state.get("python_tool", None)

    return DataAnalysisAgent(
        dataframe,
        model_name=selected_model,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        column_guideline=user_column_guideline,
        shared_python_tool=shared_tool if shared_python_tool is None else shared_python_tool,  # 공유 도구 전달
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent_SchoolFood" not in st.session_state:
        st.error("먼저 '데이터 분석 시작' 버튼을 클릭해주세요.")
        return

    st.chat_message("user").write(query)
    add_message_main(MessageRole.USER, [MessageType.TEXT, query])

    agent = st.session_state["agent_SchoolFood"]
    python_tool = st.session_state["python_tool"]

    # 이전 결과 초기화 (figure는 유지)
    if 'result_df' in python_tool.locals:
        del python_tool.locals['result_df']
    st.session_state['last_result_df'] = None
    # st.session_state['last_figure'] = None  # figure 초기화 제거

    # Agent 실행
    with st.spinner("AI가 생각하고 있습니다..."):
        response_items = []
        executed_code = ""
        
        # 스트림 응답 수집
        for step in agent.stream(query, "abc123"):
            response_items.append(step)
            
            # 코드 실행 감지
            if "actions" in step:
                for action in step["actions"]:
                    if action.tool == "python_repl_tool":
                        executed_code = action.tool_input.get("code", "")

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
            # plt.close(fig)  # 메모리 정리 - 제거
            # st.session_state['last_figure'] = None  # 사용 후 초기화 - 제거

        # 4. 텍스트 답변
        ai_answer = ""
        for item in response_items:
            if "output" in item:
                ai_answer += item["output"]
        
        if ai_answer:
            st.write(ai_answer)
            add_message_main(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


###### Service Part ######
# 데이터가 제대로 로딩되었는지 확인
if LOADED_DATA_SchoolFood is not None:
    column_info = ", ".join(LOADED_DATA_SchoolFood.columns.tolist())
    df_chain = create_chain()
    if "df_chain_SchoolFood" not in st.session_state:
        st.session_state["df_chain_SchoolFood"] = df_chain
else:
    st.error("데이터 로딩에 실패했습니다. 파일을 확인해주세요.")
    st.stop()

# 사이드바 설정
with st.sidebar:
    clear_btn = st.button("메인 윈도우 대화 초기화 ➡️")  # 대화 내용을 초기화하는 버튼

#     st.text("추천질문")
#     st.text("(추천 질문을 참고해 오른쪽 메인 화면에서 데이터 분석을 요청해보세요.)")
#     col1, col2 = st.columns(2)
#     placeholder1 = col1.empty()
#     placeholder2 = col2.empty()

#     st.text("AI와 대화하기")
#     messages = st.container(height=400)

#     # 이전 대화를 출력
#     def print_messages_sidebar():
#         for chat_message in st.session_state["messages_sidebar_SchoolFood"]:
#             messages.chat_message(chat_message.role).write(chat_message.content)

#     # 새로운 메시지를 추가
#     def add_message_sidebar(role, message):
#         st.session_state["messages_sidebar_SchoolFood"].append(ChatMessage(role=role, content=message))

#     # 키워드 질문 처리
#     kw_1, kw_2 = create_keyword(st.session_state["messages_sidebar_SchoolFood"])
#     kw_button_1 = placeholder1.button(label=kw_1, use_container_width=True, help="클릭하면 또 다른 질문이 생성돼요")
#     kw_button_2 = placeholder2.button(label=kw_2, use_container_width=True, help="클릭하면 또 다른 질문이 생성돼요")

#     # 사용자의 입력
#     user_input = st.chat_input("🤓분석도우미에게 궁금한 내용을 물어보세요!")

#     # 대화 chain 설정
#     chain = st.session_state["df_chain_SchoolFood"]

#     # 이전 대화 기록 출력
#     print_messages_sidebar()

#     # 만약에 사용자 입력이 들어오면...
#     if user_input:
#         # 사용자의 입력
#         messages.chat_message("user").write(user_input)

#         input = {
#             "context": column_info,
#             "question": user_input
#         }

#         # 답변 스트리밍 호출
#         response = chain.stream(input)

#         with messages.chat_message("assistant"):
#             # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
#             container = st.empty()

#             ai_answer = ""
#             for token in response:
#                 if isinstance(token, AIMessageChunk):
#                     ai_answer += token.content
#                 elif isinstance(token, str):
#                     ai_answer += token
#                 container.markdown(ai_answer)

#         # 대화기록을 저장한다.
#         add_message_sidebar("user", user_input)
#         add_message_sidebar("assistant", ai_answer)

#         kw_1, kw_2 = create_keyword(st.session_state["messages_sidebar_SchoolFood"])
#         placeholder1.button(label=kw_1, use_container_width=True)
#         placeholder2.button(label=kw_2, use_container_width=True)

# 메인 로직
if clear_btn:
    st.session_state["messages_SchoolFood"] = []  # 대화 내용 초기화
    # figure 메모리 정리
    if 'last_figure' in st.session_state and st.session_state['last_figure'] is not None:
        try:
            plt.close(st.session_state['last_figure'])
        except:
            pass
        st.session_state['last_figure'] = None

apply_btn = st.button("데이터 분석 시작", type="primary")  # 데이터 분석을 시작하는 버튼

if apply_btn:
    loaded_data = LOADED_DATA_SchoolFood
    
    # 데이터 정보 표시 (한글로)
    st.info(f"📊 데이터 로딩 완료: {loaded_data.shape[0]}개 음식 항목, {loaded_data.shape[1]}개 컬럼")
    st.info("컬럼 정보: 음식종류, 메뉴명, 배식량, 잔반량, 섭취량")
    
    # 초기 데이터프레임 표시
    st.subheader("📋 학교 급식 데이터 미리보기")    
    add_message_main(MessageRole.ASSISTANT, [MessageType.DATAFRAME, loaded_data.head(10)])

    # 세션 상태에 저장
    st.session_state["df_SchoolFood"] = loaded_data
    st.session_state["python_tool"] = PythonAstREPLTool() # Python 실행 도구 생성
    st.session_state["last_result_df"] = None
    # st.session_state["last_figure"] = None  # figure 초기화 제거

    # Python 실행 도구 생성 (중요: 하나만 생성)
    st.session_state["python_tool"].locals["df_SchoolFood"] = loaded_data  # 데이터프레임을 Python 실행 환경에 추가
    st.session_state["python_tool"].locals["df"] = loaded_data  # df로도 접근 가능하게 추가
    st.session_state["python_tool"].locals["pd"] = pd  # pandas 추가
    st.session_state["python_tool"].locals["plt"] = plt  # matplotlib 추가
    st.session_state["python_tool"].locals["sns"] = sns  # seaborn 추가

    # Agent 생성 - 같은 python_tool 사용하도록
    st.session_state["agent_SchoolFood"] = create_agent(
        loaded_data,
        DEFAULT_DATA_ANALYSIS_MODEL,
        prefix_prompt="""
이 AI는 중학생 여러분을 위한 학교 급식 데이터 분석 도우미입니다.
- 어려운 용어나 복잡한 개념 대신 쉽고 친근한 말로 설명해주세요
- 중학생이 이해할 수 있는 수준으로 답변해주세요
- 분석 결과를 실생활과 연결해서 설명해주세요
- 궁금한 점이 있으면 언제든지 물어보세요!

**중요한 시각화 규칙**:
- 그래프를 그릴 때는 한글 폰트 설정을 하지 마세요 (plt.rcParams['font.family'] = ... 금지)
- 한글 폰트는 이미 전역으로 설정되어 있으니 바로 matplotlib과 seaborn을 사용하세요
- 시각화 코드에서 폰트 관련 설정은 절대 포함하지 마세요""",
        postfix_prompt="""
<출력 형식>
분석 결과를 설명할 때는 다음 형식을 따라주세요:

- **핵심 내용**: (한 줄로 핵심 내용을 요약해주세요)
- **상세 설명**: (핵심 내용에 대한 상세 설명을 3문장이 넘지 않게, 중학생이 이해하기 쉽게 구체적으로 설명해주세요)
- **다음 분석 추천**: (이어서 분석해보면 좋을 내용을 최대 3가지만 추천해주세요)
   - 각 추천은 구체적이고 실행 가능한 분석이어야 합니다
   - 예: "○○별로 비교해보기", "△△와 ◇◇의 관계 알아보기" 등

답변은 친근하고 격려하는 톤으로 작성해주세요.
</출력 형식>""",
        shared_python_tool=st.session_state.get("python_tool", None),  # 중요: 같은 도구 전달
    )  # 에이전트 생성

    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")

print_messages_main()  # 저장된 메시지 출력

user_input = st.chat_input("데이터에 질문해보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리
