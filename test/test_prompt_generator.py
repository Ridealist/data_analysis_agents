#!/usr/bin/env python3
"""
프롬프트 생성기 독립 테스트 스크립트
finalize_planning 함수를 직접 테스트하여 프롬프트 생성이 잘 작동하는지 확인

사용법:
    python test_prompt_generator.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가 (test/ 디렉토리의 부모)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import AIMessage
from src.graph.learning_agent import finalize_planning, LearningAgentState

# 테스트용 Mock 데이터
TEST_PLANNING_DATA = {
    "dataset_description": "iris 데이터는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비, 그리고 품종 정보를 담고 있어요",
    "data_analysis_objective": "품종별로 꽃잎의 특징이 어떻게 다른지 알아보고 싶어요",
    "data_analysis_plan": "1. 데이터 탐색 및 기본 통계 확인\n2. 품종별 특성 비교\n3. 시각화를 통한 패턴 발견"
}

TEST_DATASETS = {
    "Iris 데이터": {
        "dataset_description": "iris 데이터는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비, 그리고 품종 정보를 담고 있어요",
        "data_analysis_objective": "품종별로 꽃잎의 특징이 어떻게 다른지 알아보고 싶어요",
        "data_analysis_plan": "1. 데이터 탐색 및 기본 통계 확인\n2. 품종별 특성 비교\n3. 시각화를 통한 패턴 발견"
    },
    "온라인 쇼핑몰 데이터": {
        "dataset_description": "온라인 쇼핑몰의 고객 구매 데이터입니다. 나이, 성별, 구매 금액, 구매 날짜, 상품 카테고리 등의 정보가 있어요",
        "data_analysis_objective": "어떤 연령대와 성별이 가장 많이 구매하는지, 그리고 구매 패턴에 어떤 특징이 있는지 파악하고 싶어요",
        "data_analysis_plan": "1. 연령대별 구매 통계 분석\n2. 성별 구매 패턴 비교\n3. 구매 금액 분포 및 상품 카테고리별 선호도 확인\n4. 시간대별 구매 트렌드 분석"
    },
    "학교 급식 데이터": {
        "dataset_description": "서울시 초등학교 급식 데이터입니다. 학교명, 급식일자, 메뉴, 칼로리, 영양소 정보 등이 포함되어 있어요",
        "data_analysis_objective": "급식 메뉴의 영양 균형과 칼로리 분포를 분석하고, 계절별 메뉴 패턴을 파악하고 싶어요",
        "data_analysis_plan": "1. 급식 메뉴 기본 통계 확인\n2. 칼로리 및 영양소 분포 분석\n3. 계절별 메뉴 패턴 분석\n4. 학교별 급식 품질 비교"
    }
}


def create_mock_state(planning_data: dict) -> LearningAgentState:
    """테스트용 Mock State 생성"""
    # Tool call을 시뮬레이션한 AIMessage 생성
    mock_aimessage = AIMessage(
        content="",
        tool_calls=[{
            "id": "test_tool_call_123",
            "name": "PlanningCriteria",
            "args": planning_data
        }]
    )
    
    return {
        "messages": [mock_aimessage],
        "phase": "planning",
        "planning_info": None,
        "reflection_info": None,
        "generated_prompt": None,
        "execution_summary": None,
        "planning_completed": False,
        "execution_completed": False,
        "reflection_completed": False,
        "waiting_for_input": False,
        "summary_generated": False
    }


def test_prompt_generator(planning_data: dict = None):
    """프롬프트 생성기 테스트"""
    if planning_data is None:
        planning_data = TEST_PLANNING_DATA
    
    print("=" * 70)
    print("🧪 프롬프트 생성기 테스트")
    print("=" * 70)
    
    # Mock State 생성
    state = create_mock_state(planning_data)
    
    print("\n📋 테스트 데이터:")
    print(f"  - 데이터 설명: {planning_data['dataset_description']}")
    print(f"  - 분석 목표: {planning_data['data_analysis_objective']}")
    print(f"  - 분석 계획: {planning_data['data_analysis_plan']}")
    
    print("\n⏳ 프롬프트 생성 중...")
    
    try:
        result = finalize_planning(state)
        
        print("\n✅ 테스트 성공!")
        print("\n📊 결과:")
        print(f"  - Planning 완료: {result.get('planning_completed', False)}")
        print(f"  - Phase: {result.get('phase', 'N/A')}")
        
        generated_prompt = result.get("generated_prompt")
        if generated_prompt:
            print(f"\n🎯 생성된 프롬프트 전체:")
            print("-" * 70)
            print(generated_prompt)
            print("-" * 70)
            print(f"\n📏 전체 길이: {len(generated_prompt)}자")
            
            # 메시지 확인
            messages = result.get("messages", [])
            print(f"\n💬 생성된 메시지 수: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                msg_preview = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                print(f"  {i}. {type(msg).__name__}: {msg_preview}...")
            
            return True, generated_prompt
        else:
            print("\n⚠️ 프롬프트 생성 실패 (결과 없음)")
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
    
    dataset_desc = input("데이터 설명: ").strip() or TEST_PLANNING_DATA["dataset_description"]
    objective = input("분석 목표: ").strip() or TEST_PLANNING_DATA["data_analysis_objective"]
    plan = input("분석 계획: ").strip() or TEST_PLANNING_DATA["data_analysis_plan"]
    
    custom_data = {
        "dataset_description": dataset_desc,
        "data_analysis_objective": objective,
        "data_analysis_plan": plan
    }
    
    success, generated_prompt = test_prompt_generator(custom_data)
    
    if success and generated_prompt:
        print("\n" + "=" * 70)
        print("생성된 프롬프트 전체:")
        print("=" * 70)
        print(generated_prompt)
        print("=" * 70)
        
        # 파일로 저장할지 물어보기
        save = input("\n파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("파일명 (기본: generated_prompt.txt): ").strip() or "generated_prompt.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(generated_prompt)
            print(f"✅ {filename}에 저장되었습니다.")


def test_all_datasets():
    """모든 테스트 데이터셋으로 테스트"""
    print("\n" + "=" * 70)
    print("🔄 모든 테스트 데이터셋으로 테스트")
    print("=" * 70)
    
    results = {}
    
    for name, data in TEST_DATASETS.items():
        print(f"\n📊 테스트 중: {name}")
        print("-" * 70)
        success, prompt = test_prompt_generator(data)
        results[name] = {
            "success": success,
            "prompt_length": len(prompt) if prompt else 0
        }
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    for name, result in results.items():
        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"  {name}: {status} (프롬프트 길이: {result['prompt_length']}자)")
    print("=" * 70)


def main():
    """메인 함수"""
    print("\n🚀 프롬프트 생성기 테스트 도구\n")
    print("1. 기본 테스트 데이터로 테스트")
    print("2. 사용자 정의 데이터로 테스트")
    print("3. 모든 테스트 데이터셋으로 테스트")
    print("4. 종료")
    
    choice = input("\n선택 > ").strip()
    
    if choice == "1":
        test_prompt_generator()
    elif choice == "2":
        test_with_custom_data()
    elif choice == "3":
        test_all_datasets()
    elif choice == "4":
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

