"""
프롬프트 생성 Agent

기존 PromptGenerator 로직을 래핑하여 LangGraph 노드에서 사용 가능하도록 변환
"""

from typing import List, Dict, Optional


def generate_messages(
    conversation_history: List[Dict[str, str]],
    student_message: str,
    tutor_acts: list,
    phase_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    대화 이력, 학생 메시지, 튜터 행동을 바탕으로 LLM API용 메시지 생성
    
    Args:
        conversation_history: 전체 대화 이력
        student_message: 학생의 현재 메시지
        tutor_acts: 선택된 튜터 행동 리스트 (List[TutorAct])
        phase_context: 단계 컨텍스트 ("execution" 등)
    
    Returns:
        List[Dict[str, str]]: LLM API에 전달할 메시지 리스트
        예: [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
        ]
    """
    from src.agency.prompts.PromptGenerator import RigidPromptGenerator
    
    # 기존 RigidPromptGenerator 인스턴스 생성 및 프롬프트 생성 수행
    generator = RigidPromptGenerator()
    messages = generator.get_prompt(
        conversation_history,
        student_message,
        tutor_acts,
        phase_context=phase_context
    )
    
    return messages

