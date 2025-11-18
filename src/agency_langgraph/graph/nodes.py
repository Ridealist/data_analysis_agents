"""
Phase 2: 노드 설계 (간결화)

LangGraph 워크플로우의 각 노드 함수 정의
각 노드는 State를 받아서 업데이트할 필드만 반환
"""

from typing import Dict, Any
from .state import AgencyState
from ..agents.assessor import assess_student
from ..agents.selector import select_tutor_acts
from ..agents.generator import generate_messages
from ..utils.helpers import postprocess_response


def assess_node(state: AgencyState) -> Dict[str, Any]:
    """
    노드 1: 학생 상태 평가
    
    IntentAssessor를 사용하여 학생의 메시지를 분석하고 학습 상태를 평가
    
    Args:
        state: 현재 상태 (conversation_history, student_message, config 사용)
    
    Returns:
        Dict: 업데이트할 필드
        - assessment: JSON 형식의 평가 결과
        - tokens: assessor 토큰 사용량
    """
    assessment, tokens = assess_student(
        conversation_history=state["conversation_history"],
        student_message=state["student_message"],
        client=state["config"]["client"],
        model=state["config"]["model"],
        assessment_history=state["config"].get("assessment_history", [])
    )
    
    return {
        "assessment": assessment,
        "tokens": {"assessor": tokens}
    }


def select_node(state: AgencyState) -> Dict[str, Any]:
    """
    노드 2: 튜터 행동 선택
    
    학생 평가 결과를 바탕으로 적절한 튜터 행동(TutorAct)을 선택
    
    Args:
        state: 현재 상태 (assessment, config 사용)
    
    Returns:
        Dict: 업데이트할 필드
        - tutor_acts: 선택된 튜터 행동 리스트
    """
    tutor_acts = select_tutor_acts(
        assessment=state["assessment"],
        client=state["config"]["client"],
        open_mode=state["config"]["open_mode"],
        intent_history=state["config"].get("intent_history", [])
    )
    
    return {"tutor_acts": tutor_acts}


def generate_prompt_node(state: AgencyState) -> Dict[str, Any]:
    """
    노드 3: 프롬프트 생성
    
    대화 이력, 학생 메시지, 튜터 행동을 조합하여 LLM API용 메시지 생성
    
    Args:
        state: 현재 상태 (conversation_history, student_message, tutor_acts 사용)
    
    Returns:
        Dict: 업데이트할 필드
        - messages: LLM API에 전달할 메시지 리스트
    """
    phase_context = state["config"].get("phase_context")
    
    messages = generate_messages(
        conversation_history=state["conversation_history"],
        student_message=state["student_message"],
        tutor_acts=state["tutor_acts"],
        phase_context=phase_context
    )
    
    return {"messages": messages}


def respond_node(state: AgencyState) -> Dict[str, Any]:
    """
    노드 4: 일반 응답 생성 (Non-streaming)
    
    생성된 프롬프트로 LLM을 호출하여 최종 응답 생성
    
    Args:
        state: 현재 상태 (messages, config 사용)
    
    Returns:
        Dict: 업데이트할 필드
        - output: LLM 응답 텍스트
        - tokens: 전체 토큰 사용량 정보
    """
    client = state["config"]["client"]
    model = state["config"]["model"]
    max_tokens = state["config"]["max_tokens"]
    
    # LLM API 호출
    completion = client.chat.completions.create(
        model=model,
        messages=state["messages"],
        max_tokens=max_tokens
    )
    
    # 응답 후처리 (LaTeX 수식 변환)
    response = postprocess_response(completion.choices[0].message.content)
    
    # 토큰 사용량 계산
    final_tokens = completion.usage.total_tokens
    assessor_tokens = state["tokens"]["assessor"]
    total_tokens = assessor_tokens + final_tokens
    
    return {
        "output": response,
        "tokens": {
            **state["tokens"],
            "final": final_tokens,
            "total": total_tokens
        }
    }


def stream_node(state: AgencyState) -> Dict[str, Any]:
    """
    노드 5: 스트리밍 응답 생성
    
    생성된 프롬프트로 LLM을 호출하여 스트리밍 방식으로 응답 생성
    
    Args:
        state: 현재 상태 (messages, config 사용)
    
    Returns:
        Dict: 업데이트할 필드
        - output: 스트리밍 객체
    """
    client = state["config"]["client"]
    model = state["config"]["model"]
    max_tokens = state["config"]["max_tokens"]
    
    # LLM API 호출 (스트리밍 모드)
    stream = client.chat.completions.create(
        model=model,
        messages=state["messages"],
        max_tokens=max_tokens,
        stream=True
    )
    
    return {"output": stream}

