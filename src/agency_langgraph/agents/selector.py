"""
튜터 행동 선택 Agent

기존 TutorActSelector 로직을 래핑하여 LangGraph 노드에서 사용 가능하도록 변환
"""

from typing import List


def select_tutor_acts(
    assessment: str,
    client,
    open_mode: bool,
    intent_history: list = []
) -> List:
    """
    학생 평가 결과를 바탕으로 튜터의 다음 행동(TutorAct)을 선택
    
    Args:
        assessment: JSON 형식의 평가 결과 문자열
        client: OpenAI/Azure 클라이언트
        open_mode: 개방형 모드 여부
        intent_history: 의도 이력
    
    Returns:
        List[TutorAct]: 선택된 튜터 행동 리스트
        예: [TutorAct.MEL_EXPLORE, TutorAct.MER_OWNERSHIP]
    """
    from src.agency.prompts.TutorActSelector import LLMIntentSelector
    
    # 기존 LLMIntentSelector 인스턴스 생성 및 의도 선택 수행
    selector = LLMIntentSelector(client, intent_history)
    tutor_acts = selector.get_intent(assessment, open=open_mode)
    
    return tutor_acts
