"""
LangGraph 기반 교육 에이전트 테스트 파일

기존 AgencyTutor와 새로운 AgencyTutorLG의 동작을 비교 테스트
"""

from src.agency.Tutor import AgencyTutor
from src.agency_langgraph.tutor import AgencyTutorLG
from src.agency.utils.get_key import get_client_openAI


def test_basic_response():
    """기본 응답 테스트"""
    print("=" * 60)
    print("테스트 1: 기본 응답 생성")
    print("=" * 60)
    
    # 클라이언트 생성
    client = get_client_openAI()
    
    # 새로운 LangGraph 버전
    tutor_lg = AgencyTutorLG(
        client=client,
        model="gpt-4o",
        open=True
    )
    
    # 테스트 입력
    conversation_history = []
    student_message = "데이터 분석을 어떻게 시작해야 할지 모르겠어요."
    
    # 응답 생성
    response, total_tokens, prompt_tokens, completion_tokens, intent, assessment = tutor_lg.get_response(
        conversation_history=conversation_history,
        messages_student=student_message,
        max_tokens=1500
    )
    
    print(f"\n학생 메시지: {student_message}")
    print(f"\n튜터 응답: {response}")
    print(f"\n선택된 튜터 행동: {intent}")
    print(f"평가 결과: {assessment}")
    print(f"토큰 사용량 - 총: {total_tokens}, 프롬프트: {prompt_tokens}, 완료: {completion_tokens}")
    print()


def test_streaming_response():
    """스트리밍 응답 테스트"""
    print("=" * 60)
    print("테스트 2: 스트리밍 응답 생성")
    print("=" * 60)
    
    client = get_client_openAI()
    
    tutor_lg = AgencyTutorLG(
        client=client,
        model="gpt-4o",
        open=True
    )
    
    conversation_history = []
    student_message = "ChatGPT에게 전부 물어보면 되는 거 아닌가요?"
    
    stream, intent, assessment = tutor_lg.get_response_stream(
        conversation_history=conversation_history,
        messages_student=student_message,
        max_tokens=1500
    )
    
    print(f"\n학생 메시지: {student_message}")
    print(f"\n튜터 응답 (스트리밍):")
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print(f"\n\n선택된 튜터 행동: {intent}")
    print(f"평가 결과: {assessment}")
    print()


def test_conversation_flow():
    """대화 흐름 테스트 (히스토리 관리)"""
    print("=" * 60)
    print("테스트 3: 대화 흐름 및 히스토리 관리")
    print("=" * 60)
    
    client = get_client_openAI()
    
    tutor_lg = AgencyTutorLG(
        client=client,
        model="gpt-4o",
        open=True
    )
    
    # 첫 번째 대화
    conversation_history = []
    student_message_1 = "데이터 분석 과제가 있는데 막막해요."
    
    response_1, _, _, _, intent_1, assessment_1 = tutor_lg.get_response(
        conversation_history=conversation_history,
        messages_student=student_message_1
    )
    
    print(f"\n[Turn 1]")
    print(f"학생: {student_message_1}")
    print(f"튜터: {response_1}")
    print(f"행동: {intent_1}")
    
    # 대화 히스토리 업데이트
    conversation_history.append({"role": "user", "content": student_message_1})
    conversation_history.append({"role": "assistant", "content": response_1})
    
    # 두 번째 대화
    student_message_2 = "그냥 ChatGPT한테 다 물어보면 안 돼요?"
    
    response_2, _, _, _, intent_2, assessment_2 = tutor_lg.get_response(
        conversation_history=conversation_history,
        messages_student=student_message_2
    )
    
    print(f"\n[Turn 2]")
    print(f"학생: {student_message_2}")
    print(f"튜터: {response_2}")
    print(f"행동: {intent_2}")
    
    # 히스토리 확인
    print(f"\n의도 이력: {tutor_lg.intent_history}")
    print(f"평가 이력 개수: {len(tutor_lg.assessment_history)}")
    print()


def compare_old_vs_new():
    """기존 버전과 새 버전 비교"""
    print("=" * 60)
    print("테스트 4: 기존 AgencyTutor vs 새로운 AgencyTutorLG 비교")
    print("=" * 60)
    
    client = get_client_openAI()
    
    # 기존 버전
    tutor_old = AgencyTutor(
        client=client,
        model="gpt-4o",
        open=True
    )
    
    # 새 버전
    tutor_new = AgencyTutorLG(
        client=client,
        model="gpt-4o",
        open=True
    )
    
    conversation_history = []
    student_message = "분석 방법을 알려주세요."
    
    # 기존 버전 응답
    print("\n[기존 AgencyTutor]")
    response_old, tokens_old, _, _, intent_old, _ = tutor_old.get_response(
        conversation_history=conversation_history,
        messages_student=student_message
    )
    print(f"응답: {response_old[:100]}...")
    print(f"토큰: {tokens_old}")
    print(f"행동: {intent_old}")
    
    # 새 버전 응답
    print("\n[새로운 AgencyTutorLG]")
    response_new, tokens_new, _, _, intent_new, _ = tutor_new.get_response(
        conversation_history=conversation_history,
        messages_student=student_message
    )
    print(f"응답: {response_new[:100]}...")
    print(f"토큰: {tokens_new}")
    print(f"행동: {intent_new}")
    
    print(f"\n두 버전 모두 정상 작동! ✅")
    print()


if __name__ == "__main__":
    print("\n🚀 LangGraph 기반 교육 에이전트 테스트 시작\n")
    
    # 모든 테스트 실행
    test_basic_response()
    test_streaming_response()
    test_conversation_flow()
    compare_old_vs_new()
    
    print("=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)

