import os
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from typing import Annotated, Optional

from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


########################################################
import uuid
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageType(Enum):
    TEXT = "text"
    CODE = "code"
    DATAFRAME = "dataframe"
    FIGURE = "figure"
    ERROR = "error"


def add_message_main(role: MessageRole, content_list):
    """메시지를 세션에 추가하는 함수"""
    try:
        if "messages_UploadData" not in st.session_state:
            st.session_state["messages_UploadData"] = []
        
        message = {
            "role": role.value,
            "content": content_list,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state["messages_UploadData"].append(message)
        
        # 데이터베이스 로깅 시도 (옵션)
        try:
            if hasattr(st.session_state, 'conversation_manager') and st.session_state.conversation_manager:
                conversation_manager = st.session_state.conversation_manager
                session_uuid = st.session_state.get('session_uuid', str(uuid.uuid4()))
                
                conversation_manager.add_message(
                    session_uuid=session_uuid,
                    role=role.value,
                    content_list=content_list
                )
        except Exception as db_error:
            # DB 로깅 실패해도 앱은 계속 실행
            print(f"DB 로깅 실패: {db_error}")
            
    except Exception as e:
        print(f"메시지 추가 실패: {e}")
########################################################


class DataAnalysisAgent:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4.1",
        api_key: str = os.environ["OPENAI_API_KEY"],
        temperature: float = 0.2,  # 변경: 1.0 → 0.2
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
        shared_python_tool: Optional[PythonAstREPLTool] = None,  # 추가: 공유 도구 받기
    ):
        self.df = dataframe
        self.model_name = model_name
        self.api_key = api_key
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
            code: Annotated[str, "Any python code(pandas, numpy, matplotlib, seaborn) to run"],
        ):
            """Use this tool to run python, pandas, numpy, matplotlib, and seaborn code."""
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
                            # DB에 에러 로그 저장
                            try:
                                error_msg = f"커스텀 폰트 로딩 실패: {str(font_error)}"
                                add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
                            except:
                                pass  # DB 로깅 실패 시 무시
                            plt.rcParams['font.family'] = 'DejaVu Sans'
                            print("전역: 기본 폰트를 사용합니다.")
                    else:
                        # 폰트 파일도 없으면 기본 폰트 사용
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        print("전역: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
                
            except Exception as e:
                print(f"전역 폰트 설정 오류: {e}")
                # DB에 에러 로그 저장
                try:
                    error_msg = f"폰트 설정 오류: {str(e)}"
                    add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
                except:
                    pass  # DB 로깅 실패 시 무시
                plt.rcParams['font.family'] = 'DejaVu Sans'

            try:                
                # 마이너스 폰트 깨짐 방지
                plt.rcParams["axes.unicode_minus"] = False

                # ========================================
                # 개선 2: 공유 도구 상태 관리
                # ========================================
                # 공유 도구가 있으면 사용, 없으면 새로 생성
                if self.shared_python_tool:
                    python_tool = self.shared_python_tool

                    # df가 현재 도구의 locals에 없으면 추가
                    if 'df' not in python_tool.locals:
                        python_tool.locals['df'] = self.df
                    if 'df_UploadData' not in python_tool.locals:
                        python_tool.locals['df_UploadData'] = self.df
                    
                    # 필수 라이브러리는 덮어쓰지 않고 없을 때만 추가
                    if 'sns' not in python_tool.locals:
                        python_tool.locals['sns'] = sns
                    if 'plt' not in python_tool.locals:
                        python_tool.locals['plt'] = plt
                    if 'np' not in python_tool.locals:
                        python_tool.locals['np'] = np
                    if 'pd' not in python_tool.locals:
                        python_tool.locals['pd'] = pd
                else:
                    python_tool = PythonAstREPLTool(
                        locals={"df": self.df, "df_UploadData": self.df, "sns": sns, "plt": plt, "np": np, "pd": pd}
                    )

                # ========================================
                # 개선: UI 표시용 상태만 초기화
                # ========================================
                # 이전 실행의 UI 결과가 다음 실행에 표시되지 않도록
                st.session_state['last_result_df'] = None
                
                # 코드 실행
                result = python_tool.invoke(code)

                # ========================================
                # 개선: result_df 캡처 로직 (우선순위 기반)
                # ========================================
                # 실행 후 결과 변수 찾기 (우선순위 순서)
                result_var_priority = ['result_df', 'result', 'output_df', 'final_df']

                for var_name in result_var_priority:
                    if var_name in python_tool.locals:
                        var_value = python_tool.locals[var_name]
                        if isinstance(var_value, (pd.DataFrame, pd.Series)):
                            # UI에 표시할 결과만 세션 상태에 저장
                            st.session_state['last_result_df'] = var_value
                            print(f"[DataAnalysisAgent] result_df saved to session state: shape={var_value.shape if hasattr(var_value, 'shape') else len(var_value)}")                        
                            break
                        else:
                            print(f"[DataAnalysisAgent] {var_name} exists but is not DataFrame or Series: type={type(var_value)}")
                    else:
                        print(f"[DataAnalysisAgent] No {var_name} found in python_tool.locals")


                # 시각화 코드가 실행된 후 figure가 실제로 생성되었는지 확인
                if any(keyword in code for keyword in ["plt.", "sns.", "matplotlib", "seaborn"]):
                    fig = plt.gcf()
                    # figure가 유효하고 axes가 있는지 확인 (실제로 그래프가 그려졌는지)
                    if fig and hasattr(fig, 'axes') and fig.axes and len(fig.axes) > 0:
                        # figure를 세션 상태에 저장 (ask 함수에서 출력)
                        st.session_state['last_figure'] = fig                  
                        print(f"[DataAnalysisAgent] Figure saved to session state: {fig}")
                    else:
                        print("[DataAnalysisAgent] Figure exists but no data plotted")
                else:
                    print("[DataAnalysisAgent] No visualization keywords found in code")

                return result

            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"[DataAnalysisAgent Error] {error_detail}")
                
                # DB에 에러 로그 저장
                try:
                    error_msg = f"Python 코드 실행 오류: {str(e)}"
                    add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
                except:
                    pass  # DB 로깅 실패 시 무시
                
                return f"Execution failed. Error: {repr(e)}"

        return python_repl_tool

    def build_system_prompt(self):

        system_prompt = load_prompt("src/prompts/data-analysis-V03.yaml", encoding="utf-8")
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

        # API 키 확인 및 디버깅
        if not self.api_key:
            raise ValueError("❌ OpenAI API 키가 설정되지 않았습니다.")

        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, openai_api_key=self.api_key)
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,  # 변경: 20 → 10
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
