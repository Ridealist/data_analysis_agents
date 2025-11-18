"""
LangChain messages ↔ conversation_history 변환 유틸리티
"""

from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


def convert_langchain_to_history(messages: List) -> List[Dict[str, str]]:
    """
    LangChain 메시지 리스트를 conversation_history 형식으로 변환
    
    Args:
        messages: LangChain 메시지 리스트 (HumanMessage, AIMessage 등)
    
    Returns:
        List[Dict[str, str]]: conversation_history 형식
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Note:
        - SystemMessage, ToolMessage는 제외
        - HumanMessage → {"role": "user", "content": "..."}
        - AIMessage → {"role": "assistant", "content": "..."} (tool_calls 없을 때만)
    """
    history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            # tool_calls가 있는 AIMessage는 제외 (도구 호출 중)
            history.append({"role": "assistant", "content": msg.content})
        # SystemMessage, ToolMessage는 제외
    
    return history


def extract_last_user_message(messages: List) -> str:
    """
    LangChain 메시지 리스트에서 마지막 사용자 메시지 추출
    
    Args:
        messages: LangChain 메시지 리스트
    
    Returns:
        str: 마지막 사용자 메시지 내용 (없으면 빈 문자열)
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""

