import time
import uuid
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.components.dynamo import ConversationManager, S3Manager, timestamp_to_datetime, MessageManager
from src.components.dataclass import MessageType
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage
from src.components.llm import set_openai_api_key

store = {}


def load_messages_from_db(conversation_id: str, message_manager: MessageManager, s3_manager: S3Manager, table_name: str = None):
    """DB에서 메시지를 로드하여 session_state에 저장"""
    if not conversation_id or not message_manager:
        return False
    
    try:
        # DB에서 메시지 로드 (테이블명이 지정된 경우 해당 테이블 사용)
        if table_name:
            db_messages = message_manager.load_messages(conversation_id, table_name)
        else:
            db_messages = message_manager.load_messages(conversation_id)
        
        if not db_messages:
            return False
        
        # 메시지를 세션 상태 형식으로 변환
        session_messages = []
        
        with st.spinner("대화 기록을 불러오는 중..."):
            for msg in db_messages:
                role = msg['role']
                content_type = msg['content_type']
                content = msg['content']
                
                # S3에 저장된 콘텐츠인 경우 복원
                if content_type in ['dataframe_s3', 'figure_s3']:
                    restored_content = s3_manager.download_from_s3(content, content_type)
                    if restored_content is not None:
                        if content_type == 'dataframe_s3':
                            # DataFrame 복원
                            session_messages.append([role, [[MessageType.DATAFRAME, restored_content]]])
                        elif content_type == 'figure_s3':
                            # Figure 복원 (BytesIO를 matplotlib Figure로 변환)
                            try:
                                restored_content.seek(0)
                                # PIL을 사용하여 이미지 로드 (더 안전함)
                                from PIL import Image
                                img = Image.open(restored_content)
                                
                                # matplotlib Figure로 변환
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.imshow(img)
                                ax.axis('off')
                                
                                session_messages.append([role, [[MessageType.FIGURE, fig]]])
                            except Exception as e:
                                print(f"Figure 복원 실패: {e}")
                                # Figure 복원 실패 시 텍스트로 대체
                                session_messages.append([role, [[MessageType.TEXT, f"차트 로드 실패: {e}"]]])
                    else:
                        # 복원 실패 시 에러 메시지
                        session_messages.append([role, [[MessageType.ERROR, f"콘텐츠 복원 실패: {content_type}"]]])
                else:
                    # 일반 텍스트/코드 메시지
                    if isinstance(content, str):
                        # role이 assistant(AI)인 경우에만 AI 응답 형식 감지 로직 적용
                        if role == "assistant":
                            # AI 응답 형식인지 확인 (핵심 내용, 상세 설명 포함 여부)
                            # 더 정확한 패턴 매칭을 위해 여러 패턴 확인
                            ai_response_patterns = [
                                '핵심 내용:', '상세 설명:', '**핵심 내용**', '**상세 설명**',
                                '출력 형식', '분석 결과', '결론', '요약'
                            ]
                            
                            is_ai_response = any(pattern in content for pattern in ai_response_patterns)
                            
                            if is_ai_response:
                                # AI 응답 형식이면 TEXT로 저장
                                session_messages.append([role, [[MessageType.TEXT, content]]])
                            else:
                                # AI 응답 형식이 아니면 CODE로 저장 (st.expander 안에서 st.code로 표시)
                                session_messages.append([role, [[MessageType.CODE, content]]])
                        else:
                            # user인 경우는 이전처럼 일반 텍스트로 저장
                            session_messages.append([role, [[MessageType.TEXT, content]]])
                    else:
                        # 문자열이 아닌 경우 TEXT로 저장
                        session_messages.append([role, [[MessageType.TEXT, content]]])
            
            # 세션 상태에 메시지 저장
            if table_name and "dataanalysis" in table_name:
                st.session_state["messages_UploadData"] = session_messages
            elif table_name and "preprocessing" in table_name:
                st.session_state["messages_UploadData_preprocessing"] = session_messages
            else:
                st.warning("대화 기록 로드 실패: 테이블 이름이 지정되지 않았습니다.")  # 테이블 이름이 지정되지 않았을 때 경고 메시지 출력

            return True
        
    except Exception as e:
        st.error(f"메시지 로드 실패: {e}")
        return False


#--------------------------------#
#      Sidebar Configuration     #
#--------------------------------#
def render_conversation_history_analysis(s3_manager: S3Manager, conversation_manager: ConversationManager, message_manager: MessageManager, table_name: str = None):
    with st.sidebar:
        st.divider()
        st.header("💬 대화 기록")

        # 새 대화 시작 버튼
        if st.button("➕ 새 대화 시작", use_container_width=True):
            st.session_state.session_uuid = str(uuid.uuid4())
            st.session_state.conversation_id = None
                            
            # 데이터 분석 에이전트용 변수 초기화
            st.session_state["messages_UploadData"] = []
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None
            st.session_state["uploaded_data"] = None
            st.session_state["df_UploadData"] = None
            st.session_state["messages_sidebar_UploadData"] = []
            
            if "agent_with_UploadData" in st.session_state:
                del st.session_state["agent_with_UploadData"]
            if "python_tool" in st.session_state:
                del st.session_state["python_tool"]
            if "thinking_placeholder" in st.session_state:
                del st.session_state["thinking_placeholder"]
            
            st.success("🆕 새로운 대화를 시작합니다!")
            time.sleep(1)
            st.rerun()

        # 대화 목록 표시 (사용자 ID가 있는 경우)
        if hasattr(st.session_state, 'username'):
            try:
                conversations = conversation_manager.get_user_conversations(st.session_state.username)
                
                for convo in conversations:
                    # 현재 보고 있는 대화 강조
                    button_type = "primary" if convo['session_uuid'] == st.session_state.conversation_id else "secondary"
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        if st.button(f"{convo['topic']} ({timestamp_to_datetime(convo['created_at']).strftime('%y-%m-%d %H:%M')})", 
                                    key=f"convo_{convo['session_uuid']}", use_container_width=True, type=button_type):
                            st.session_state.session_uuid = convo['session_uuid']
                            st.session_state.conversation_id = convo['session_uuid']
                            
                            # 이전 대화 메시지 로드 (테이블명 전달)
                            if load_messages_from_db(convo['session_uuid'], message_manager, s3_manager, table_name):
                                st.success(f"✅ '{convo['topic']}' 대화를 불러왔습니다!")
                                
                                # 에이전트 상태 복원 (데이터프레임이 있는 경우)
                                if st.session_state.get("messages_UploadData"):
                                    for role, content_list in st.session_state["messages_UploadData"]:
                                        for content in content_list:
                                            if isinstance(content, list) and len(content) == 2:
                                                message_type, message_content = content
                                                if message_type == MessageType.DATAFRAME and isinstance(message_content, pd.DataFrame):
                                                    # 마지막 데이터프레임을 현재 데이터로 설정
                                                    st.session_state["df_UploadData"] = message_content
                                                    if "python_tool" in st.session_state:
                                                        st.session_state["python_tool"].locals["df_UploadData"] = message_content
                                                        st.session_state["python_tool"].locals["df"] = message_content
                                                    break
                            else:
                                st.warning(f"⚠️ '{convo['topic']}' 대화 로드에 실패했습니다.")
                            
                            time.sleep(1)
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{convo['session_uuid']}"):
                            conversation_manager.delete_conversation(convo['session_uuid'], s3_manager)
                            if st.session_state.conversation_id == convo['session_uuid']:
                                st.session_state.session_uuid = str(uuid.uuid4())
                                st.session_state.conversation_id = None
                            st.rerun()
            except Exception as e:
                st.error(f"대화 목록 로드 실패: {e}")


def render_conversation_history_preprocessing(s3_manager: S3Manager, conversation_manager: ConversationManager, message_manager: MessageManager, table_name: str = None):
    with st.sidebar:
        st.divider()
        st.header("💬 대화 기록")

        # 새 대화 시작 버튼
        if st.button("➕ 새 대화 시작", use_container_width=True):
            st.session_state.session_uuid = str(uuid.uuid4())
            st.session_state.conversation_id = None
            
            # 전처리 에이전트 전용 변수 초기화
            st.session_state["messages_UploadData_preprocessing"] = []
            st.session_state["last_result_df_preprocessing"] = None
            st.session_state["uploaded_data_preprocessing"] = None
            st.session_state["df_UploadData_preprocessing"] = None
            st.session_state["file_processed_preprocessing"] = False
            st.session_state["uploaded_file_signature_preprocessing"] = None
            st.session_state["messages_sidebar_UploadData_preprocessing"] = []
            
            # 전처리 에이전트 관련 상태 삭제
            if "agent_with_UploadData_preprocessing" in st.session_state:
                del st.session_state["agent_with_UploadData_preprocessing"]
            if "python_tool_preprocessing" in st.session_state:
                del st.session_state["python_tool_preprocessing"]
            if "thinking_placeholder_preprocessing" in st.session_state:
                del st.session_state["thinking_placeholder_preprocessing"]
            
            st.success("🆕 새로운 대화를 시작합니다!")
            time.sleep(1)
            st.rerun()

        # 대화 목록 표시 (사용자 ID가 있는 경우)
        if hasattr(st.session_state, 'username'):
            try:
                conversations = conversation_manager.get_user_conversations(st.session_state.username)
                
                for convo in conversations:
                    # 현재 보고 있는 대화 강조
                    button_type = "primary" if convo['session_uuid'] == st.session_state.conversation_id else "secondary"
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        if st.button(f"{convo['topic']} ({timestamp_to_datetime(convo['created_at']).strftime('%y-%m-%d %H:%M')})", 
                                    key=f"convo_{convo['session_uuid']}", use_container_width=True, type=button_type):
                            st.session_state.session_uuid = convo['session_uuid']
                            st.session_state.conversation_id = convo['session_uuid']
                            
                            # 이전 대화 메시지 로드 (테이블명 전달)
                            if load_messages_from_db(convo['session_uuid'], message_manager, s3_manager, table_name):
                                st.success(f"✅ '{convo['topic']}' 대화를 불러왔습니다!")
                                
                                # 전처리 에이전트 상태 복원 (데이터프레임이 있는 경우)
                                if st.session_state.get("messages_UploadData_preprocessing"):
                                    for role, content_list in st.session_state["messages_UploadData_preprocessing"]:
                                        for content in content_list:
                                            if isinstance(content, list) and len(content) == 2:
                                                message_type, message_content = content
                                                if message_type == MessageType.DATAFRAME and isinstance(message_content, pd.DataFrame):
                                                    # 마지막 데이터프레임을 현재 데이터로 설정
                                                    st.session_state["df_UploadData_preprocessing"] = message_content
                                                    st.session_state["uploaded_data_preprocessing"] = message_content
                                                    
                                                    # Python 도구가 있으면 데이터프레임 설정
                                                    if "python_tool_preprocessing" in st.session_state:
                                                        st.session_state["python_tool_preprocessing"].locals["df_UploadData"] = message_content
                                                        st.session_state["python_tool_preprocessing"].locals["df"] = message_content
                                                    break
                            else:
                                st.warning(f"⚠️ '{convo['topic']}' 대화 로드에 실패했습니다.")
                            
                            time.sleep(1)
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{convo['session_uuid']}"):
                            conversation_manager.delete_conversation(convo['session_uuid'], s3_manager)
                            if st.session_state.conversation_id == convo['session_uuid']:
                                st.session_state.session_uuid = str(uuid.uuid4())
                                st.session_state.conversation_id = None
                            st.rerun()
            except Exception as e:
                st.error(f"대화 목록 로드 실패: {e}")

#--------------------------------#
#      LLM 체인 생성 함수          #
#--------------------------------#

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# 사이드바 시스템 체인 생성
def generate_chain(model_name="gpt-4o-mini"):
    # API 키 설정
    if not set_openai_api_key():
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        return None
    
    # 현재 선택된 operation 값 가져오기
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """당신은 친근하고 대화형 학습을 돕는 **물리 튜터**입니다."""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history


def render_chat_interface():
    # 추천 질문 사용 여부 토글 추가
    selected_model = "gpt-4o-mini"

    st.header("🤓 튜터와 대화하기")
    messages = st.container(height=400)

    # 이전 대화를 출력
    def print_messages():
        for chat_message in st.session_state["tutor_messages"]:
            messages.chat_message(chat_message.role).write(chat_message.content)

    # 새로운 메시지를 추가
    def add_message(role, message):
        st.session_state["tutor_messages"].append(ChatMessage(role=role, content=message))

    # 초기 대화 기록 추가
    if len(st.session_state["tutor_messages"]) == 0:
        messages.chat_message("assistant").write("안녕하세요! 저는 AI 탐구도우미입니다. 무엇을 도와드릴까요?")
    else:
        pass

    # 이전 대화 기록 출력
    print_messages()

    # 사용자의 입력
    user_input = st.chat_input("🤓 AI 튜터에게 궁금한 내용을 물어보세요!")

    # 만약에 사용자 입력이 들어오면...
    if user_input:
        # 답변 요청
        conv_chain = generate_chain(selected_model)
        
        # API 키가 설정되지 않은 경우 처리
        if conv_chain is None:
            messages.chat_message("assistant").write("⚠️ OpenAI API 키가 설정되지 않았습니다. ⚙️ 시스템 설정에서 API 키를 입력해주세요.")
            return
        
        # 사용자의 입력
        messages.chat_message("user").write(user_input)

        with messages.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            try:
                generator = conv_chain.stream(
                    {
                        "input": user_input,
                    },
                    config={"configurable": {"session_id": "ab12"}}
                )
                ai_answer = ""
                for token in generator:
                    ai_answer += token.content
                    container.markdown(ai_answer)
                
                # 대화기록을 저장한다.
                add_message("user", user_input)
                add_message("assistant", ai_answer)
            except Exception as e:
                error_msg = f"❌ AI 응답 생성 중 오류가 발생했습니다: {str(e)}"
                container.markdown(error_msg)
                add_message("assistant", error_msg)
