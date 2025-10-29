"""
LangGraph 기반 교육 에이전트 메인 인터페이스

기존 AgencyTutor 클래스와 동일한 API를 제공하여 호환성 유지
내부 구현은 LangGraph 워크플로우 사용
"""

from typing import Tuple, List, Any
from .graph.workflow import create_agency_graph


class AgencyTutorLG:
    """
    LangGraph 기반 교육 에이전트
    
    기존 AgencyTutor 인터페이스와 완벽 호환
    내부적으로 LangGraph 워크플로우를 사용하여 상태 관리 및 실행 흐름 제어
    """
    
    def __init__(
        self,
        client,
        model: str = "gpt-4o-2024-05-13",
        intent_history: list = [],
        assessment_history: list = [],
        open: bool = True,
        version: str = "V1"
    ):
        """
        교육 에이전트 초기화
        
        Args:
            client: OpenAI/Azure 클라이언트
            model: 사용할 LLM 모델명
            intent_history: 의도 이력 (튜터 행동 이력)
            assessment_history: 평가 이력 (학생 상태 평가 이력)
            open: 개방형 모드 여부
            version: 버전 (사용되지 않음, 호환성 유지용)
        """
        self.client = client
        self.model = model
        self.open = open
        self.intent_history = list(intent_history)
        self.assessment_history = list(assessment_history)
        
        # LangGraph 워크플로우 생성
        self.graph = create_agency_graph()
    
    def update_client(self, client):
        """클라이언트 업데이트"""
        self.client = client
    
    def update_model(self, model: str):
        """모델 업데이트"""
        self.model = model
    
    def get_response(
        self,
        conversation_history: list,
        messages_student: str,
        max_tokens: int = 1500
    ) -> Tuple[str, int, int, int, List, str]:
        """
        일반 응답 생성 (Non-streaming)
        
        기존 AgencyTutor.get_response()와 동일한 시그니처
        
        Args:
            conversation_history: 대화 이력 (List[Dict[str, str]])
            messages_student: 학생의 메시지
            max_tokens: 최대 토큰 수
        
        Returns:
            Tuple[str, int, int, int, List, str]:
            - response: LLM 응답 텍스트
            - total_tokens: 총 토큰 사용량
            - prompt_tokens: 프롬프트 토큰 (assessor 토큰)
            - completion_tokens: 완료 토큰 (final LLM 토큰)
            - intent: 선택된 튜터 행동 리스트
            - assessment: JSON 형식의 평가 결과
        """
        # State 구성
        initial_state = {
            "conversation_history": conversation_history,
            "student_message": messages_student,
            "config": {
                "stream_mode": False,
                "open_mode": self.open,
                "max_tokens": max_tokens,
                "model": self.model,
                "client": self.client,
                "intent_history": self.intent_history,
                "assessment_history": self.assessment_history
            }
        }
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        # 기존 형식으로 반환값 변환
        response = result["output"]
        total_tokens = result["tokens"]["total"]
        prompt_tokens = result["tokens"]["assessor"]
        completion_tokens = result["tokens"]["final"]
        intent = result["tutor_acts"]
        assessment = result["assessment"]
        
        # 히스토리 업데이트
        self.intent_history.append(intent)
        self.assessment_history.append(assessment)
        
        return response, total_tokens, prompt_tokens, completion_tokens, intent, assessment
    
    def get_response_stream(
        self,
        conversation_history: list,
        messages_student: str,
        max_tokens: int = 1500
    ) -> Tuple[Any, List, str]:
        """
        스트리밍 응답 생성
        
        기존 AgencyTutor.get_response_stream()과 동일한 시그니처
        
        Args:
            conversation_history: 대화 이력 (List[Dict[str, str]])
            messages_student: 학생의 메시지
            max_tokens: 최대 토큰 수
        
        Returns:
            Tuple[Any, List, str]:
            - stream: 스트리밍 객체
            - intent: 선택된 튜터 행동 리스트
            - assessment: JSON 형식의 평가 결과
        """
        # State 구성
        initial_state = {
            "conversation_history": conversation_history,
            "student_message": messages_student,
            "config": {
                "stream_mode": True,
                "open_mode": self.open,
                "max_tokens": max_tokens,
                "model": self.model,
                "client": self.client,
                "intent_history": self.intent_history,
                "assessment_history": self.assessment_history
            }
        }
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        stream = result["output"]
        intent = result["tutor_acts"]
        assessment = result["assessment"]
        
        # 히스토리 업데이트
        self.intent_history.append(intent)
        self.assessment_history.append(assessment)
        
        return stream, intent, assessment

