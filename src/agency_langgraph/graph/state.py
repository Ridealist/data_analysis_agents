"""
Phase 1: 간결한 상태 스키마 설계

LangGraph 워크플로우의 중앙 상태 관리
각 노드가 필요한 필드만 업데이트하는 간결한 구조
"""

from typing import TypedDict, List, Dict, Optional, Union, Any


class AgencyState(TypedDict):
    """
    교육 에이전트의 상태 스키마
    
    데이터 흐름:
    입력 → assessment → tutor_acts → messages → output
    """
    
    # ==================== 입력 (Input) ====================
    conversation_history: List[Dict[str, str]]  # 전체 대화 이력
    student_message: str                         # 학생의 현재 메시지
    
    # ==================== 설정 (Config) ====================
    config: Dict[str, Any]
    # {
    #   "stream_mode": bool,        # 스트리밍 여부
    #   "open_mode": bool,          # 개방형 모드 여부 (튜터 행동에 영향)
    #   "max_tokens": int,          # LLM 응답 토큰 제한
    #   "model": str,               # 사용 모델명
    #   "client": Any,              # OpenAI/Azure 클라이언트
    #   "intent_history": List,     # 의도 이력
    #   "assessment_history": List  # 평가 이력
    # }
    
    # ==================== 파이프라인 (노드별 업데이트) ====================
    assessment: Optional[str]                    # JSON string from IntentAssessor
    tutor_acts: Optional[List[Any]]              # List[TutorAct] from TutorActSelector
    messages: Optional[List[Dict[str, str]]]     # LLM API용 메시지 from PromptGenerator
    
    # ==================== 출력 (Output) ====================
    output: Optional[Union[str, Any]]            # response string OR stream object
    
    # ==================== 메타데이터 (Metadata) ====================
    tokens: Optional[Dict[str, int]]
    # {
    #   "assessor": int,  # IntentAssessor 토큰 사용량
    #   "final": int,     # 최종 LLM 호출 토큰 사용량
    #   "total": int      # 전체 토큰 합계
    # }

