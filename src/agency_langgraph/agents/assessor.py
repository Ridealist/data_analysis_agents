"""
학생 상태 평가 Agent

기존 IntentAssessor 로직을 래핑하여 LangGraph 노드에서 사용 가능하도록 변환
"""

from typing import Tuple, List, Dict


def assess_student(
    conversation_history: List[Dict[str, str]],
    student_message: str,
    client,
    model: str,
    assessment_history: list = []
) -> Tuple[str, int]:
    """
    학생의 메시지를 분석하여 학습 상태를 평가
    
    Args:
        conversation_history: 전체 대화 이력
        student_message: 학생의 현재 메시지
        client: OpenAI/Azure 클라이언트
        model: 사용할 모델명
        assessment_history: 평가 이력
    
    Returns:
        Tuple[str, int]: (assessment_json, tokens_used)
        - assessment_json: JSON 형식의 평가 결과 {"justification": "...", "selection": "abc"}
        - tokens_used: 평가에 사용된 총 토큰 수
    """
    from src.agency.prompts.IntentAssessor import IntentAssessor
    
    # 기존 IntentAssessor 인스턴스 생성 및 평가 수행
    assessor = IntentAssessor(client, model, assessment_history)
    assessment, prompt_tokens, completion_tokens = assessor.assess(
        conversation_history, 
        student_message
    )
    
    total_tokens = prompt_tokens + completion_tokens
    
    return assessment, total_tokens

