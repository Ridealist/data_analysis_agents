import streamlit as st
from typing import Optional, Tuple, List, Dict

from src.components.dynamo import (
    DynamoDBManager,
    ConversationManager,
    MessageManager,
    timestamp_to_datetime,
)


AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-northeast-2")

# Learning Agent용 DynamoDB 테이블 이름
# 공통 Conversation 테이블 + Stage1(planning) / Stage3(reflection) 전용 메시지 테이블
LEARNING_CONVERSATIONS_TABLE = "learning_conversations"
LEARNING_MESSAGES_TABLE = "learning_stage1_messages"        # Stage1 기본 메시지 테이블
LEARNING_STAGE3_MESSAGES_TABLE = "learning_stage3_messages" # Stage3 메시지 테이블


@st.cache_resource
def init_learning_managers() -> Tuple[Optional[DynamoDBManager], Optional[ConversationManager], Optional[MessageManager]]:
    """Learning Agent용 DynamoDB 매니저 초기화."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        # AWS 설정이 없으면 Dynamo 연동을 비활성화
        return None, None, None

    try:
        dynamo_manager = DynamoDBManager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            conversations_table=LEARNING_CONVERSATIONS_TABLE,
            messages_table=LEARNING_MESSAGES_TABLE,
        )

        conversation_manager = ConversationManager(
            dynamo_manager, LEARNING_CONVERSATIONS_TABLE
        )
        message_manager = MessageManager(dynamo_manager)

        return dynamo_manager, conversation_manager, message_manager
    except Exception as e:
        st.error(f"Learning Agent용 DynamoDB 매니저 초기화 실패: {e}")
        return None, None, None


@st.cache_resource
def init_learning_stage3_managers() -> Tuple[Optional[DynamoDBManager], Optional[ConversationManager], Optional[MessageManager]]:
    """Learning Agent - Stage3(Reflection)용 DynamoDB 매니저 초기화."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        # AWS 설정이 없으면 Dynamo 연동을 비활성화
        return None, None, None

    try:
        dynamo_manager = DynamoDBManager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            conversations_table=LEARNING_CONVERSATIONS_TABLE,
            messages_table=LEARNING_STAGE3_MESSAGES_TABLE,
        )

        conversation_manager = ConversationManager(
            dynamo_manager, LEARNING_CONVERSATIONS_TABLE
        )
        message_manager = MessageManager(dynamo_manager)

        return dynamo_manager, conversation_manager, message_manager
    except Exception as e:
        st.error(f"Learning Agent Stage3용 DynamoDB 매니저 초기화 실패: {e}")
        return None, None, None


def ensure_learning_conversation():
    """
    현재 learning_thread_id를 기반으로 DynamoDB에 대화(conversation)를 생성/조회하고
    session_state에 관련 정보를 저장.
    """
    from datetime import datetime

    _, conversation_manager, message_manager = init_learning_managers()

    if conversation_manager is None or message_manager is None:
        return

    thread_id = st.session_state.get("learning_thread_id")
    if not thread_id:
        return

    # 이미 conversation이 연결되어 있다면 재사용
    if st.session_state.get("learning_conversation_id"):
        st.session_state["learning_message_manager"] = message_manager
        return

    user_id = st.session_state.get("username", "anonymous")

    try:
        conv = conversation_manager.get_or_create_conversation(thread_id, user_id)
        if conv:
            st.session_state["learning_conversation_id"] = conv["session_uuid"]
            st.session_state["learning_message_manager"] = message_manager
            # 최근 업데이트 시간을 표시하기 위해 참고용 타임스탬프 저장 (옵션)
            st.session_state["learning_conversation_created_at"] = conv.get(
                "created_at"
            )
    except Exception as e:
        # 로깅만 하고 앱 동작은 계속
        print(f"[learning_logging] 대화 생성/연결 중 오류: {e}")



def load_learning_threads_for_user() -> List[Dict]:
    """
    현재 로그인한 사용자의 학습 세션 목록을 DynamoDB에서 조회하여
    sidebar / LearningAgentRunner 에서 사용할 수 있는 단순 리스트로 변환.
    """
    if not hasattr(st.session_state, "username"):
        return []

    _, conversation_manager, _ = init_learning_managers()
    if conversation_manager is None:
        return []

    try:
        conversations = conversation_manager.get_user_conversations(
            st.session_state.username
        )
    except Exception as e:
        print(f"[load_learning_threads_for_user] Error: {e}")
        return []

    thread_list: List[Dict] = []
    for conv in conversations:
        created_at = conv.get("created_at")
        created_str = ""
        if created_at is not None:
            try:
                dt = timestamp_to_datetime(created_at)
                created_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_str = str(created_at)

        thread_list.append(
            {
                "id": conv.get("session_uuid"),
                "title": conv.get("topic", "학습 세션"),
                "created_at": created_str,
                # 기존 스키마에는 phase가 없을 수 있으므로 기본값을 planning으로 설정
                "phase": conv.get("phase", "planning"),
            }
        )

    return thread_list


