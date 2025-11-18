# -----------------------------------------------
# Raw → Tidy 전처리 전용 에이전트 (DataAnalysisAgent와 동일 인터페이스)
# -----------------------------------------------
from langchain.tools import tool
from typing import Annotated, Optional
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import re
import streamlit as st

# 메시지 로깅 유틸과 Enum은 기존 코드 재사용 (있다고 가정)
# from your_module import add_message_main, MessageRole, MessageType

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


class DataPreprocessingAgent:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4o",
        api_key: str = os.environ["OPENAI_API_KEY"],
        temperature: float = 0.3,
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
        shared_python_tool: Optional[PythonAstREPLTool] = None,
    ):
        # --- DataAnalysisAgent와 동일한 필드 유지 ---
        self.df = dataframe
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.prefix_prompt = prefix_prompt
        self.postfix_prompt = postfix_prompt
        self.shared_python_tool = shared_python_tool

        if column_guideline is not None and column_guideline.strip() != "":
            COLUMN_GUIDE_PREFIX = "###\n\n# Column Guideline\n\nHere's the column guideline you'll be working with:\n"
            self.column_guideline = COLUMN_GUIDE_PREFIX + column_guideline
        else:
            self.column_guideline = ""

        self.tools = [self.create_python_repl_tool()]  # 같은 이름/동작 지점
        self.store = {}
        self.setup_agent()

    # ---- 전처리 전용 Python 툴 (시각화/파일/네트워크 차단, DF/figure 세션 저장 안 함) ----
    def create_python_repl_tool(self):
        @tool
        def python_repl_tool(
            code: Annotated[str, "Any python code (pandas, numpy, regex, datetime) for RAW→TIDY preprocessing to run."]
        ):
            """
            Use this tool to run python, pandas, numpy code for RAW→TIDY preprocessing only.
            """
            # 사전 금지 패턴 검사
            banned_patterns = [
                r"\bimport\s+matplotlib\b", r"\bfrom\s+matplotlib\b",
                r"\bimport\s+seaborn\b", r"\bfrom\s+seaborn\b",
                r"\bimport\s+plotly\b", r"\bfrom\s+plotly\b",
                r"\bimport\s+os\b", r"\bfrom\s+os\b",
                r"\bimport\s+sys\b", r"\bfrom\s+sys\b",
                r"\bimport\s+subprocess\b", r"\bfrom\s+subprocess\b",
                r"\bimport\s+requests\b", r"\bfrom\s+requests\b",
                r"\bimport\s+urllib\b", r"\bfrom\s+urllib\b",
                r"\bto_csv\s*\(", r"\bread_csv\s*\(", r"\bto_parquet\s*\(", r"\bread_parquet\s*\(",
                r"\bplt\.", r"\bsns\.",
            ]
            for pat in banned_patterns:
                if re.search(pat, code):
                    return f"Execution blocked: disallowed operation detected ({pat}). This agent is for RAW→TIDY preprocessing without I/O/plotting."

            try:
                # 공유 도구가 있으면 사용, 없으면 새로 생성 (동일 인터페이스)
                if self.shared_python_tool:
                    python_tool = self.shared_python_tool
                    if 'df' not in python_tool.locals:
                        python_tool.locals['df'] = self.df
                    if 'df_UploadData' not in python_tool.locals:
                        python_tool.locals['df_UploadData'] = self.df
                    python_tool.locals['np'] = np
                    python_tool.locals['pd'] = pd
                else:
                    python_tool = PythonAstREPLTool(
                        locals={"df": self.df, "df_UploadData": self.df, "np": np, "pd": pd}
                    )

                # 실행
                result = python_tool.invoke(code)

                # 전처리 에이전트 정책:
                # - 세션에 DataFrame/figure 저장하지 않음
                # - 결과는 print로 요약(열 수/결측치/타입/중복 제거 수 등)하도록 LLM 프롬프트에서 유도
                # - 전처리 결과는 'cleaned_df' 변수에 저장하되 세션 상태에는 저장하지 않음
                
                # 전처리 결과가 cleaned_df에 있는지 확인 (디버깅용)
                if 'cleaned_df' in python_tool.locals:
                    cleaned_data = python_tool.locals['cleaned_df']
                    if isinstance(cleaned_data, (pd.DataFrame, pd.Series)):
                        print(f"[전처리 완료] 최종 데이터: {cleaned_data.shape[0]}행, {cleaned_data.shape[1]}열")
                        # 전처리 결과를 세션에 저장하지 않음 (전처리 전용 에이전트이므로)
                        # 대신 print로 요약 정보만 출력하도록 유도
                        # cleaned_df는 전처리 결과를 보여주기 위한 임시 변수
                
                # 전처리 에이전트는 코드 실행을 제안하지 않음
                # 사용자가 직접 실행하도록 안내하는 것이 목적
                
                return result

            except Exception as e:
                # 필요 시 메시지 로깅 (기존 add_message_main 이용 가능)
                try:
                    error_msg = f"Python 전처리 코드 실행 오류: {str(e)}"
                    add_message_main(MessageRole.ASSISTANT, [MessageType.ERROR, error_msg])
                except:
                    pass
                return f"Execution failed. Error: {repr(e)}"

        return python_repl_tool

    # ---- 전처리 특화 프롬프트 (파일 의존 제거: YAML 대신 인라인) ----
    def build_system_prompt(self):
        """
        RAW→TIDY 전처리 규칙:
        - 목적: 원본 df/df_UploadData를 'tidy data' 형태로 정제
        - 허용 작업: 열명 스네이크케이스화, 불필요 열 제거, 타입 변환, 결측치 처리, 이상치 처리(간단한 규칙),
                    범주 정규화(트림/소문자/매핑), 중복 제거, 긴 포맷↔넓은 포맷 변환, 날짜 파싱, 정렬/인덱스 재설정 등
        - 금지 작업: 시각화, 파일/네트워크 I/O, 시스템 호출
        - 출력: 최종 상태 요약을 print로 명확히 보고 (행/열 수, 결측치 요약, 변경된 열/타입, 삭제/치환 건수 등)
        - 최종 코드 내에서 결과 DataFrame은 변수명 'cleaned_df'에 저장 (전처리 전용, 세션에 저장하지 않음)
        """
        head_text = self.df.head().to_string()
        guide = self.column_guideline or ""
        sys = f"""
<CACHE_PROMPT>You are an advanced Data Analysis Agent, expertly skilled in using Pandas, matplotlib, and seaborn for data manipulation and visualization tasks. 
Your primary function is to generate code-only responses to user queries, with thorough explanations, data driven analysis and comments.

###

# Tools
You will use the following tools:
- `python_repl_tool`: Use this tool to execute pandas, matplotlib, seaborn python code.

###

You are a Python data preprocessing agent that converts RAW data to TIDY data using pandas/numpy only.
Work on an in-memory DataFrame named `df` (same as `df_UploadData`).
Never produce plots, never read/write files, never use network/system calls.

Context preview (df.head):
{head_text}

Column guideline (optional):
{guide}

When you respond (via the python tool), write COMPLETE, STANDALONE, COPY-PASTE READY code following this EXACT template:

```python
# ==========================================
# Data preprocessing script - ready for Jupyter/Colab
# ==========================================

# 1. Required libraries
import pandas as pd
import numpy as np

# 2. Load your dataset (update the file path as needed)
# CSV example:
df = pd.read_csv('replace_with_your_file.csv', encoding='utf-8')
# Excel example:
# df = pd.read_excel('replace_with_your_file.xlsx')

print("=== Original data ===")
print(f"Shape: {{{{df.shape}}}}")
print(df.head())

# 3. Preprocessing steps (add your transformations below)
# [Write detailed preprocessing logic here]

# 4. Review the result
print("\\n=== Preprocessed data ===")
print(f"Shape: {{{{cleaned_df.shape}}}}")
print(f"Missing values: {{{{cleaned_df.isnull().sum().sum()}}}}")
print(cleaned_df.head())
```

CRITICAL RULES:
1) Always include the complete template above with import statements
2) Fill in the preprocessing logic in section 3 with actual code
3) Final DataFrame must be named `cleaned_df` (NOT `result_df`)
4) Include clear English comments explaining each preprocessing step
5) Code must be fully runnable when copy-pasted (only file path updates allowed)
6) Provide before/after audit logs (row/column counts, missing values, dtype changes)
7) Avoid plotting libraries or file/network I/O

CRITICAL RESPONSE RULES:
- NEVER offer to execute the code for the user
- NEVER imply that you can run the code on their behalf
- ALWAYS end your response with: "**Run this code yourself in Colab or Jupyter Notebook.**"
- Your role is to provide code and guidance only; the user runs the code

Now, respond to the user's query accordingly.</CACHE_PROMPT>
"""
        if self.prefix_prompt:
            sys = f"{self.prefix_prompt}\n\n{sys}"
        if self.postfix_prompt:
            sys = f"{sys}\n\n{self.postfix_prompt}"
        return sys

    # ---- DataAnalysisAgent와 동일한 세팅/실행 흐름 ----
    def setup_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.build_system_prompt()),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        if not self.api_key:
            raise ValueError("❌ OpenAI API 키가 설정되지 않았습니다.")

        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature, openai_api_key=self.api_key)
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

    def stream(self, input_query, session_id="def123"):
        agent_with_chat_history = self.get_agent_with_chat_history()
        response = agent_with_chat_history.stream(
            {"input": input_query},
            config={"configurable": {"session_id": session_id}},
        )
        return response
