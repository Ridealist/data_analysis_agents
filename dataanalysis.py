from langchain.tools import tool
from typing import Annotated, Optional
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import seaborn as sns
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import streamlit as st

load_dotenv()


class DataAnalysisAgent:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
        shared_python_tool: Optional[PythonAstREPLTool] = None,  # 추가: 공유 도구 받기
    ):
        self.df = dataframe
        self.model_name = model_name
        self.temperature = temperature
        self.prefix_prompt = prefix_prompt
        self.postfix_prompt = postfix_prompt
        self.shared_python_tool = shared_python_tool  # 공유 도구 저장
        
        if column_guideline is not None and column_guideline.strip() != "":
            COLUMN_GUIDE_PREFIX = "###\n\n# Column Guideline\n\nHere's the column guideline you'll be working with:\n"
            self.column_guideline = COLUMN_GUIDE_PREFIX + column_guideline
        else:
            self.column_guideline = ""
        self.tools = [self.create_python_repl_tool()]
        self.store = {}
        self.setup_agent()

    def create_python_repl_tool(self):
        @tool
        def python_repl_tool(
            code: Annotated[str, "Any python code(pandas, matplotlib, seaborn) to run"],
        ):
            """Use this tool to run python, pandas query, matplotlib, and seaborn code."""
            try:
                #### 폰트 설정
                # 먼저 시스템에서 사용 가능한 한글 폰트 찾기 (안전한 방식)
                available_fonts = []
                for f in fm.fontManager.ttflist:
                    try:
                        if hasattr(f, 'name'):
                            available_fonts.append(f.name)
                        elif hasattr(f, 'family'):
                            available_fonts.append(f.family)
                        elif isinstance(f, str):
                            available_fonts.append(f)
                    except:
                        continue
                
                korean_fonts = [f for f in available_fonts if any(k in str(f).lower() for k in ['gothic', 'gulim', 'dotum', 'batang', 'malgun', 'nanum'])]
                
                if korean_fonts:
                    # 시스템에 한글 폰트가 있으면 사용
                    plt.rcParams['font.family'] = korean_fonts[0]
                    print(f"전역 시스템 한글 폰트 사용: {korean_fonts[0]}")
                else:
                    # 시스템에 한글 폰트가 없으면 커스텀 폰트 파일 시도
                    font_path = "data/fonts/NanumGothic.ttf"
                    if os.path.exists(font_path):
                        try:
                            font_prop = fm.FontProperties(fname=font_path)
                            font_name = font_prop.get_name()
                            plt.rcParams['font.family'] = font_name
                            print(f"전역 커스텀 한글 폰트 설정 완료: {font_name}")
                        except Exception as font_error:
                            print(f"전역 커스텀 폰트 로딩 실패: {font_error}")
                            plt.rcParams['font.family'] = 'DejaVu Sans'
                            print("전역: 기본 폰트를 사용합니다.")
                    else:
                        # 폰트 파일도 없으면 기본 폰트 사용
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        print("전역: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
                
            except Exception as e:
                print(f"전역 폰트 설정 오류: {e}")
                plt.rcParams['font.family'] = 'DejaVu Sans'

            try:                
                # 마이너스 폰트 깨짐 방지
                plt.rcParams["axes.unicode_minus"] = False

                # 공유 도구가 있으면 사용, 없으면 새로 생성
                if self.shared_python_tool:
                    python_tool = self.shared_python_tool
                    # df가 현재 도구의 locals에 없으면 추가
                    if 'df' not in python_tool.locals:
                        python_tool.locals['df'] = self.df
                    if 'df_SchoolFood' not in python_tool.locals:
                        python_tool.locals['df_SchoolFood'] = self.df
                    python_tool.locals['sns'] = sns
                    python_tool.locals['plt'] = plt
                else:
                    python_tool = PythonAstREPLTool(
                        locals={"df": self.df, "df_SchoolFood": self.df, "sns": sns, "plt": plt}
                    )

                result = python_tool.invoke(code)

                # result_df가 생성되었는지 확인하고 세션 상태에 저장
                if 'result_df' in python_tool.locals:
                    result_df = python_tool.locals['result_df']
                    if isinstance(result_df, pd.DataFrame) or isinstance(result_df, pd.Series):
                        # Streamlit 세션 상태에 저장
                        if 'last_result_df' not in st.session_state:
                            st.session_state['last_result_df'] = None
                        st.session_state['last_result_df'] = result_df
                        print(f"[DataAnalysisAgent] result_df saved to session state: shape={result_df.shape if hasattr(result_df, 'shape') else len(result_df)}")

                # 시각화 코드가 있으면 figure를 저장만 하고 직접 출력하지 않음
                if any(keyword in code for keyword in ["plt.", "sns.", "matplotlib", "seaborn"]):
                    fig = plt.gcf()
                    if fig and fig.axes:
                        # figure를 세션 상태에 저장 (ask 함수에서 출력)
                        st.session_state['last_figure'] = fig
                        print("[DataAnalysisAgent] Figure saved to session state")
                        # 중요: 여기서 직접 출력하지 않음!
                        # st.pyplot(fig) <- 제거
                    # plt.close()는 ask 함수에서 처리하지 않음 (메모리 누수 방지)

                return result
            
            except Exception as e:
                return f"Execution failed. Error: {repr(e)}"

        return python_repl_tool

    def build_system_prompt(self):

        system_prompt = load_prompt("prompts/data-analysis-V02.yaml", encoding="utf-8")
        system_prompt = system_prompt.format(
            dataframe_head=self.df.head().to_string(),
            column_guideline=self.column_guideline,
        )

        if self.prefix_prompt is not None:
            system_prompt = f"{self.prefix_prompt}\n\n{system_prompt}"

        if self.postfix_prompt is not None:
            system_prompt = f"{system_prompt}\n\n{self.postfix_prompt}"
        return system_prompt

    def setup_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.build_system_prompt()),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=60,
            handle_parsing_errors=True,
        )

    def get_session_history(self, session_id):
        return self.store.setdefault(session_id, ChatMessageHistory())

    def get_agent_with_chat_history(self):
        return RunnableWithMessageHistory(
            self.agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def stream(self, input_query, session_id="abc123"):
        agent_with_chat_history = self.get_agent_with_chat_history()
        response = agent_with_chat_history.stream(
            {"input": input_query},
            config={"configurable": {"session_id": session_id}},
        )
        return response
