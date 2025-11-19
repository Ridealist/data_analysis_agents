#!/usr/bin/env python3
"""
반성 단계(Stage 3) 독립 테스트 스크립트
finalize_reflection 함수를 직접 테스트하여 반성 단계가 잘 작동하는지 확인

사용법:
    python test_reflection.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가 (test/ 디렉토리의 부모)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import AIMessage
from src.graph.learning_agent import (
    finalize_reflection, 
    summarize_execution_conversation,
    LearningAgentState
)

# 테스트용 Mock 데이터
TEST_REFLECTION_DATA = {
    "goal_achievement": "분석 목표를 약 80% 정도 달성했습니다. 품종별 특성 차이는 잘 파악했지만, 시각화 부분은 아직 부족해요.",
    "best_prompt": "품종별로 꽃잎 길이의 평균을 구해주세요. 이때 각 품종별로 그룹화해서 보여주면 좋을 것 같아요.",
    "new_learning": "AI에게 구체적으로 어떤 분석을 원하는지 명확하게 말해야 더 좋은 결과를 얻을 수 있다는 것을 배웠어요. 또한 그룹화나 집계 함수를 활용하면 더 의미있는 인사이트를 얻을 수 있다는 것도 알게 되었습니다."
}

# 실행 단계 대화 기록 테스트 데이터
TEST_EXECUTION_CONVERSATIONS = {
    "Iris 데이터 분석 대화": [
        ("사용자", "품종별로 꽃잎 길이의 평균을 구해주세요."),
        ("튜터", "좋은 질문이에요! 품종별 평균을 구하면 각 품종의 특징을 비교할 수 있겠네요. 어떤 품종이 가장 긴 꽃잎을 가지고 있는지 확인해볼까요?"),
        ("사용자", "네, 그리고 품종별로 그룹화해서 보여주면 좋을 것 같아요."),
        ("튜터", "그룹화를 통해 데이터를 구조화하면 패턴을 더 잘 파악할 수 있어요. 결과를 보면서 어떤 인사이트를 발견할 수 있을지 기대되네요!"),
        ("사용자", "품종별 꽃받침 너비도 비교하고 싶어요."),
        ("튜터", "꽃받침 너비까지 비교하면 더 풍부한 분석이 되겠어요. 꽃잎과 꽃받침의 관계도 함께 살펴보면 흥미로울 것 같아요.")
    ],
    "온라인 쇼핑몰 데이터 분석 대화": [
        ("사용자", "20대와 30대의 월별 구매 금액 추이를 비교해주세요."),
        ("튜터", "연령대별 구매 패턴을 시간에 따라 비교하면 트렌드를 파악할 수 있겠네요. 어떤 연령대가 더 안정적인 구매 패턴을 보이는지 확인해볼까요?"),
        ("사용자", "선 그래프로 보여주면 좋을 것 같아요."),
        ("튜터", "시각화를 통해 데이터의 흐름을 한눈에 파악할 수 있어요. 그래프를 보면서 어떤 시기에 구매가 증가하는지 발견할 수 있을 거예요."),
        ("사용자", "성별 구매 패턴도 비교하고 싶어요."),
        ("튜터", "성별과 연령대를 함께 고려하면 더 세밀한 분석이 가능하겠어요. 어떤 그룹이 특정 상품 카테고리에 관심이 많은지도 확인해볼 수 있을 것 같아요.")
    ]
}


TEST_REFLECTION_DATASETS = {
    "Iris 데이터 분석 반성": {
        "goal_achievement": "분석 목표를 약 80% 정도 달성했습니다. 품종별 특성 차이는 잘 파악했지만, 시각화 부분은 아직 부족해요.",
        "best_prompt": "품종별로 꽃잎 길이의 평균을 구해주세요. 이때 각 품종별로 그룹화해서 보여주면 좋을 것 같아요.",
        "new_learning": "AI에게 구체적으로 어떤 분석을 원하는지 명확하게 말해야 더 좋은 결과를 얻을 수 있다는 것을 배웠어요. 또한 그룹화나 집계 함수를 활용하면 더 의미있는 인사이트를 얻을 수 있다는 것도 알게 되었습니다."
    },
    "온라인 쇼핑몰 데이터 분석 반성": {
        "goal_achievement": "연령대별 구매 패턴은 잘 분석했지만, 시간대별 트렌드 분석은 아직 완료하지 못했어요. 약 70% 정도 달성했습니다.",
        "best_prompt": "20대와 30대의 월별 구매 금액 추이를 비교해주세요. 선 그래프로 보여주면 좋을 것 같아요.",
        "new_learning": "비교 분석을 할 때는 명확한 기준(연령대, 성별 등)을 제시해야 AI가 더 정확한 분석을 해준다는 것을 배웠어요. 또한 시각화 방법을 함께 요청하면 결과를 더 잘 이해할 수 있다는 것도 알게 되었습니다."
    },
    "학교 급식 데이터 분석 반성": {
        "goal_achievement": "영양소 분포와 칼로리는 잘 분석했지만, 계절별 메뉴 패턴 분석은 아직 진행 중이에요. 약 60% 정도 달성했습니다.",
        "best_prompt": "월별로 급식 메뉴의 평균 칼로리를 계산하고, 계절별로 그룹화해서 비교해주세요.",
        "new_learning": "시간에 따른 변화를 분석할 때는 날짜나 월 정보를 활용해야 한다는 것을 배웠어요. 또한 집계 함수와 그룹화를 함께 사용하면 더 의미있는 분석 결과를 얻을 수 있다는 것도 알게 되었습니다."
    }
}


def create_mock_state_for_summary(execution_conversations: list) -> LearningAgentState:
    """실행 단계 대화 요약 테스트용 Mock State 생성"""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    messages = []
    
    # 1. Planning 완료 메시지 (ToolMessage)
    messages.append(ToolMessage(
        content="계획이 완성되었습니다! 이제 실행 단계로 넘어갑니다.",
        tool_call_id="planning_tool_call_123"
    ))
    
    # 2. 실행 단계 대화 메시지들
    for role, content in execution_conversations:
        if role == "사용자":
            messages.append(HumanMessage(content=content))
        elif role == "튜터":
            messages.append(AIMessage(content=content))
    
    # 3. 실행 완료 메시지
    messages.append(AIMessage(content="실행 단계를 완료했습니다! 이제 학습을 반성해보겠습니다."))
    
    return {
        "messages": messages,
        "phase": "reflection",
        "planning_info": None,
        "reflection_info": None,
        "generated_prompt": None,
        "execution_summary": None,
        "planning_completed": True,
        "execution_completed": True,
        "reflection_completed": False,
        "waiting_for_input": False,
        "summary_generated": False
    }


def create_mock_state(reflection_data: dict, include_execution_summary: bool = True, 
                     planning_data: dict = None, execution_conversations: list = None) -> LearningAgentState:
    """테스트용 Mock State 생성 - Stage 1, 2 완료 상태 시뮬레이션"""
    from langchain_core.messages import HumanMessage, ToolMessage
    
    # 기본 planning 데이터 (제공되지 않은 경우)
    if planning_data is None:
        planning_data = {
            "dataset_description": "iris 데이터는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비, 그리고 품종 정보를 담고 있어요",
            "data_analysis_objective": "품종별로 꽃잎의 특징이 어떻게 다른지 알아보고 싶어요",
            "data_analysis_plan": "1. 데이터 탐색 및 기본 통계 확인\n2. 품종별 특성 비교\n3. 시각화를 통한 패턴 발견"
        }
    
    # 기본 execution 대화 (제공되지 않은 경우)
    if execution_conversations is None:
        execution_conversations = [
            ("사용자", "품종별로 꽃잎 길이의 평균을 구해주세요."),
            ("튜터", "좋은 질문이에요! 품종별 평균을 구하면 각 품종의 특징을 비교할 수 있겠네요."),
            ("사용자", "네, 그리고 품종별로 그룹화해서 보여주면 좋을 것 같아요."),
            ("튜터", "그룹화를 통해 데이터를 구조화하면 패턴을 더 잘 파악할 수 있어요.")
        ]
    
    messages = []
    
    # 1. Planning 완료 메시지 (ToolMessage)
    messages.append(ToolMessage(
        content="계획이 완성되었습니다! 이제 실행 단계로 넘어갑니다.",
        tool_call_id="planning_tool_call_123"
    ))
    
    # 2. 생성된 프롬프트 메시지 (선택적)
    generated_prompt = "데이터 분석을 위한 시스템 프롬프트가 생성되었습니다."
    messages.append(AIMessage(
        content=f"✅ 프롬프트가 자동으로 생성되었습니다!\n\n{generated_prompt}"
    ))
    
    # 3. 실행 단계 대화 메시지들
    for role, content in execution_conversations:
        if role == "사용자":
            messages.append(HumanMessage(content=content))
        elif role == "튜터":
            messages.append(AIMessage(content=content))
    
    # 4. 실행 완료 메시지
    messages.append(AIMessage(content="실행 단계를 완료했습니다! 이제 학습을 반성해보겠습니다."))
    
    # 5. 실행 단계 대화 요약 (포함하는 경우)
    execution_summary = None
    if include_execution_summary:
        # 대화를 문자열로 변환
        conversation_text = ""
        for role, content in execution_conversations:
            conversation_text += f"{role}: {content}\n\n"
        
        execution_summary = f"실행 단계 요약:\n- 주요 대화 주제: 데이터 분석 방법론\n- 학습자 활동: {len([c for r, c in execution_conversations if r == '사용자'])}개의 질문을 통해 분석 진행\n- 발견한 인사이트: 그룹화와 집계를 통한 패턴 분석\n- 학습 흐름: 기본 통계부터 그룹화 분석까지 단계적 접근"
        
        summary_message = AIMessage(
            content=f"📊 실행 단계 대화 요약이 완료되었습니다!\n\n{execution_summary}"
        )
        messages.append(summary_message)
    
    # 6. Reflection 단계 Tool call 메시지
    reflection_aimessage = AIMessage(
        content="",
        tool_calls=[{
            "id": "test_tool_call_reflection_123",
            "name": "ReflectionCriteria",
            "args": reflection_data
        }]
    )
    messages.append(reflection_aimessage)
    
    return {
        "messages": messages,
        "phase": "reflection",
        "planning_info": planning_data,  # Stage 1 정보 포함
        "reflection_info": None,
        "generated_prompt": generated_prompt,  # Stage 1에서 생성된 프롬프트
        "execution_summary": execution_summary,  # Stage 2 요약
        "planning_completed": True,
        "execution_completed": True,
        "reflection_completed": False,
        "waiting_for_input": False,
        "summary_generated": include_execution_summary
    }


def test_summary(execution_conversations: list = None):
    """실행 단계 대화 요약 테스트"""
    if execution_conversations is None:
        execution_conversations = TEST_EXECUTION_CONVERSATIONS["Iris 데이터 분석 대화"]
    
    print("=" * 70)
    print("📊 실행 단계 대화 요약 테스트")
    print("=" * 70)
    
    # Mock State 생성
    state = create_mock_state_for_summary(execution_conversations)
    
    print("\n📋 실행 단계 대화 기록:")
    for role, content in execution_conversations:
        print(f"  {role}: {content[:50]}..." if len(content) > 50 else f"  {role}: {content}")
    
    print("\n⏳ 대화 요약 생성 중...")
    
    try:
        result = summarize_execution_conversation(state)
        
        print("\n✅ 테스트 성공!")
        print("\n📊 결과:")
        print(f"  - Summary 생성 완료: {result.get('summary_generated', False)}")
        
        execution_summary = result.get("execution_summary")
        if execution_summary:
            print(f"\n📝 생성된 요약 전체:")
            print("-" * 70)
            print(execution_summary)
            print("-" * 70)
            print(f"\n📏 요약 길이: {len(execution_summary)}자")
            
            # 메시지 확인
            messages = result.get("messages", [])
            print(f"\n💬 생성된 메시지 수: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                msg_preview = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                msg_overall = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"  {i}. {type(msg).__name__}: {msg_overall}")
            
            return True, execution_summary
        else:
            print("\n⚠️ 요약 생성 실패 (결과 없음)")
            return False, None
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_reflection(reflection_data: dict = None, 
                   planning_data: dict = None,
                   execution_conversations: list = None):
    """반성 단계 테스트 - Stage 1, 2 완료 상태 자동 시뮬레이션"""
    if reflection_data is None:
        reflection_data = TEST_REFLECTION_DATA
    
    print("=" * 70)
    print("🧪 반성 단계(Stage 3) 테스트")
    print("=" * 70)
    print("\n💡 테스트 모드: Stage 1, 2 완료 상태를 자동으로 시뮬레이션합니다.")
    
    # Mock State 생성 (Stage 1, 2 완료 상태 포함)
    state = create_mock_state(
        reflection_data, 
        include_execution_summary=True,
        planning_data=planning_data,
        execution_conversations=execution_conversations
    )
    
    print("\n📋 시뮬레이션된 상태:")
    print(f"  - Planning 완료: {state.get('planning_completed', False)}")
    print(f"  - Execution 완료: {state.get('execution_completed', False)}")
    print(f"  - Execution 요약 생성: {state.get('summary_generated', False)}")
    print(f"  - Planning 정보: {'✅' if state.get('planning_info') else '❌'}")
    print(f"  - Generated Prompt: {'✅' if state.get('generated_prompt') else '❌'}")
    print(f"  - Execution Summary: {'✅' if state.get('execution_summary') else '❌'}")
    
    print("\n📋 테스트 데이터:")
    print(f"  - 목표 달성 여부: {reflection_data['goal_achievement']}")
    print(f"  - 좋은 프롬프트: {reflection_data['best_prompt']}")
    print(f"  - 새롭게 배운 것: {reflection_data['new_learning']}")
    
    print("\n⏳ 반성 단계 처리 중...")
    
    try:
        result = finalize_reflection(state)
        
        print("\n✅ 테스트 성공!")
        print("\n📊 결과:")
        print(f"  - Reflection 완료: {result.get('reflection_completed', False)}")
        print(f"  - Phase: {result.get('phase', 'N/A')}")
        print(f"  - Waiting for input: {result.get('waiting_for_input', False)}")
        
        reflection_info = result.get("reflection_info")
        if reflection_info:
            print(f"\n💭 수집된 반성 정보:")
            print(f"  - 목표 달성 여부: {reflection_info.get('goal_achievement', 'N/A')}")
            print(f"  - 좋은 프롬프트: {reflection_info.get('best_prompt', 'N/A')}")
            print(f"  - 새롭게 배운 것: {reflection_info.get('new_learning', 'N/A')}")
            
            # 메시지 확인
            messages = result.get("messages", [])
            print(f"\n💬 생성된 메시지 수: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                msg_preview = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                print(f"  {i}. {type(msg).__name__}: {msg_preview}...")
            
            return True, reflection_info
        else:
            print("\n⚠️ 반성 정보 수집 실패 (결과 없음)")
            return False, None
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_with_custom_data():
    """사용자 정의 데이터로 테스트"""
    print("\n" + "=" * 70)
    print("📝 사용자 정의 데이터로 테스트")
    print("=" * 70)
    
    print("\n테스트 데이터를 입력하세요 (엔터로 기본값 사용):")
    
    goal_achievement = input("목표 달성 여부: ").strip() or TEST_REFLECTION_DATA["goal_achievement"]
    best_prompt = input("좋은 프롬프트: ").strip() or TEST_REFLECTION_DATA["best_prompt"]
    new_learning = input("새롭게 배운 것: ").strip() or TEST_REFLECTION_DATA["new_learning"]
    
    custom_data = {
        "goal_achievement": goal_achievement,
        "best_prompt": best_prompt,
        "new_learning": new_learning
    }
    
    success, reflection_info = test_reflection(custom_data)
    
    if success and reflection_info:
        print("\n" + "=" * 70)
        print("수집된 반성 정보 전체:")
        print("=" * 70)
        for key, value in reflection_info.items():
            print(f"\n{key}:")
            print(f"  {value}")
        print("=" * 70)
        
        # 파일로 저장할지 물어보기
        save = input("\n파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("파일명 (기본: reflection_result.txt): ").strip() or "reflection_result.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                for key, value in reflection_info.items():
                    f.write(f"{key}:\n{value}\n\n")
            print(f"✅ {filename}에 저장되었습니다.")


def test_all_summaries():
    """모든 실행 단계 대화 기록으로 요약 테스트"""
    print("\n" + "=" * 70)
    print("🔄 모든 실행 단계 대화 기록으로 요약 테스트")
    print("=" * 70)
    
    results = {}
    
    for name, conversations in TEST_EXECUTION_CONVERSATIONS.items():
        print(f"\n📊 테스트 중: {name}")
        print("-" * 70)
        success, summary = test_summary(conversations)
        results[name] = {
            "success": success,
            "has_summary": summary is not None,
            "summary_length": len(summary) if summary else 0
        }
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 요약 테스트 결과 요약")
    print("=" * 70)
    for name, result in results.items():
        status = "✅ 성공" if result["success"] else "❌ 실패"
        summary_status = f"요약 생성: {'✅' if result['has_summary'] else '❌'}"
        print(f"  {name}: {status} ({summary_status}, 길이: {result['summary_length']}자)")
    print("=" * 70)


def test_all_datasets():
    """모든 테스트 데이터셋으로 테스트"""
    print("\n" + "=" * 70)
    print("🔄 모든 테스트 데이터셋으로 테스트")
    print("=" * 70)
    
    results = {}
    
    for name, data in TEST_REFLECTION_DATASETS.items():
        print(f"\n📊 테스트 중: {name}")
        print("-" * 70)
        success, reflection_info = test_reflection(data)
        results[name] = {
            "success": success,
            "has_reflection_info": reflection_info is not None,
            "fields_count": len(reflection_info) if reflection_info else 0
        }
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    for name, result in results.items():
        status = "✅ 성공" if result["success"] else "❌ 실패"
        info_status = f"정보 수집: {'✅' if result['has_reflection_info'] else '❌'}"
        print(f"  {name}: {status} ({info_status}, 필드 수: {result['fields_count']})")
    print("=" * 70)


def main():
    """메인 함수"""
    print("\n🚀 반성 단계(Stage 3) 테스트 도구\n")
    print("1. 기본 반성 데이터로 테스트")
    print("2. 사용자 정의 반성 데이터로 테스트")
    print("3. 모든 반성 데이터셋으로 테스트")
    print("4. 실행 단계 대화 요약 테스트 (기본)")
    print("5. 모든 실행 단계 대화 요약 테스트")
    print("6. 종료")
    
    choice = input("\n선택 > ").strip()
    
    if choice == "1":
        test_reflection()
    elif choice == "2":
        test_with_custom_data()
    elif choice == "3":
        test_all_datasets()
    elif choice == "4":
        test_summary()
    elif choice == "5":
        test_all_summaries()
    elif choice == "6":
        print("👋 종료합니다.")
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

