#!/usr/bin/env python3
"""
교육 에이전트 대화형 테스트 도구

LangGraph 기반 AgencyTutorLG를 상호작용적으로 테스트합니다.
"""

import json
import os
from datetime import datetime
from typing import List, Dict

from src.agency_langgraph.tutor import AgencyTutorLG
from src.agency.utils.get_key import get_client_openAI


class InteractiveTester:
    """대화형 테스트 도구"""
    
    def __init__(self):
        self.client = get_client_openAI()
        self.tutor = None
        self.conversation_history = []
        self.session_start = datetime.now()
        self.total_tokens = 0
        
    def initialize_tutor(self, model="gpt-4o", open_mode=True):
        """튜터 초기화"""
        print(f"🤖 튜터 초기화 중... (모델: {model}, 개방형 모드: {open_mode})")
        self.tutor = AgencyTutorLG(
            client=self.client,
            model=model,
            open=open_mode
        )
        print("✅ 초기화 완료!\n")
    
    def display_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*70)
        print("🎓 교육 에이전트 대화형 테스트 도구")
        print("="*70)
        print("\n📋 메뉴:")
        print("  1. 단일 메시지 테스트")
        print("  2. 연속 대화 테스트 (추천)")
        print("  3. 스트리밍 테스트")
        print("  4. 디버그 모드 (상세 정보)")
        print("  5. 기존 vs 새 버전 비교")
        print("")
        print("  h. 대화 히스토리 보기")
        print("  c. 히스토리 초기화")
        print("  s. 세션 정보")
        print("  q. 종료")
        print("")
    
    def display_result(self, response, intent, assessment, tokens, show_debug=False):
        """결과 표시"""
        print("\n" + "─"*70)
        
        if show_debug:
            print("🔍 디버그 정보:")
            print(f"\n📊 Assessment:")
            try:
                assessment_data = json.loads(assessment)
                print(f"  - 선택된 코드: {assessment_data.get('selection', 'N/A')}")
                print(f"  - 근거: {assessment_data.get('justification', 'N/A')[:100]}...")
            except:
                print(f"  {assessment}")
            
            print(f"\n🎯 Tutor Acts:")
            for act in intent:
                print(f"  - {act}")
            
            print(f"\n💰 토큰 사용량: {tokens}")
            print("─"*70)
        
        print("\n🎓 튜터:")
        print(f"  {response}")
        print("─"*70 + "\n")
    
    def test_single_message(self):
        """단일 메시지 테스트"""
        print("\n" + "="*70)
        print("📝 단일 메시지 테스트")
        print("="*70)
        
        message = input("\n학생 메시지를 입력하세요: ").strip()
        if not message:
            print("❌ 메시지가 비어있습니다.")
            return
        
        print("\n⏳ 처리 중...")
        
        try:
            response, total_tokens, prompt_tokens, completion_tokens, intent, assessment = self.tutor.get_response(
                conversation_history=self.conversation_history,
                messages_student=message,
                max_tokens=1500
            )
            
            self.total_tokens += total_tokens
            
            print(f"\n📊 분석 결과:")
            print(f"  - 총 토큰: {total_tokens} (프롬프트: {prompt_tokens}, 완료: {completion_tokens})")
            
            self.display_result(response, intent, assessment, total_tokens)
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def test_conversation(self):
        """연속 대화 테스트"""
        print("\n" + "="*70)
        print("💬 연속 대화 테스트")
        print("="*70)
        print("\n💡 팁: 'exit' 입력시 대화 종료")
        print("     'debug' 입력시 디버그 정보 토글")
        print("─"*70)
        
        show_debug = False
        turn = 1
        
        while True:
            print(f"\n[Turn {turn}]")
            message = input("학생: ").strip()
            
            if message.lower() == "exit":
                print("\n👋 대화를 종료합니다.")
                break
            
            if message.lower() == "debug":
                show_debug = not show_debug
                print(f"🔍 디버그 모드: {'ON' if show_debug else 'OFF'}")
                continue
            
            if not message:
                print("❌ 메시지를 입력해주세요.")
                continue
            
            print("⏳ 처리 중...")
            
            try:
                response, total_tokens, prompt_tokens, completion_tokens, intent, assessment = self.tutor.get_response(
                    conversation_history=self.conversation_history,
                    messages_student=message,
                    max_tokens=1500
                )
                
                self.total_tokens += total_tokens
                
                # 대화 이력에 추가
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                self.display_result(response, intent, assessment, total_tokens, show_debug)
                
                turn += 1
                
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
    
    def test_streaming(self):
        """스트리밍 테스트"""
        print("\n" + "="*70)
        print("🌊 스트리밍 테스트")
        print("="*70)
        
        message = input("\n학생 메시지를 입력하세요: ").strip()
        if not message:
            print("❌ 메시지가 비어있습니다.")
            return
        
        print("\n⏳ 처리 중...\n")
        
        try:
            stream, intent, assessment = self.tutor.get_response_stream(
                conversation_history=self.conversation_history,
                messages_student=message,
                max_tokens=1500
            )
            
            print("🎓 튜터 (스트리밍):")
            print("  ", end="", flush=True)
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n")
            print("─"*70)
            print(f"\n🎯 Tutor Acts: {intent}")
            print("─"*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def test_debug_mode(self):
        """디버그 모드 - 각 단계별 상세 정보"""
        print("\n" + "="*70)
        print("🔬 디버그 모드")
        print("="*70)
        
        message = input("\n학생 메시지를 입력하세요: ").strip()
        if not message:
            print("❌ 메시지가 비어있습니다.")
            return
        
        print("\n⏳ 단계별 처리 중...\n")
        
        try:
            # State 구성 확인
            print("1️⃣ State 구성:")
            print(f"  - 대화 이력: {len(self.conversation_history)}개 메시지")
            print(f"  - 학생 메시지: '{message[:50]}...' " if len(message) > 50 else f"  - 학생 메시지: '{message}'")
            
            print("\n2️⃣ 처리 시작...")
            response, total_tokens, prompt_tokens, completion_tokens, intent, assessment = self.tutor.get_response(
                conversation_history=self.conversation_history,
                messages_student=message,
                max_tokens=1500
            )
            
            # Assessment 분석
            print("\n3️⃣ Assessment 결과:")
            try:
                assessment_data = json.loads(assessment)
                print(f"  - 선택된 코드: {assessment_data.get('selection', 'N/A')}")
                print(f"  - 근거:\n    {assessment_data.get('justification', 'N/A')}")
            except:
                print(f"  {assessment}")
            
            # Tutor Acts 분석
            print("\n4️⃣ Tutor Acts 선택:")
            for i, act in enumerate(intent, 1):
                print(f"  {i}. {act}")
            
            # 프롬프트 정보 (추정)
            print("\n5️⃣ 프롬프트 생성 완료")
            print(f"  - 예상 시스템 메시지 포함")
            print(f"  - Tutor Acts 기반 의도 프롬프트 추가")
            
            # LLM 응답
            print("\n6️⃣ LLM 응답:")
            print(f"  - 모델: {self.tutor.model}")
            print(f"  - 토큰: 총 {total_tokens} (프롬프트: {prompt_tokens}, 완료: {completion_tokens})")
            
            print("\n7️⃣ 최종 응답:")
            print(f"  {response}")
            
            print("\n" + "─"*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def compare_versions(self):
        """기존 vs 새 버전 비교"""
        print("\n" + "="*70)
        print("⚖️  기존 AgencyTutor vs 새 AgencyTutorLG 비교")
        print("="*70)
        
        message = input("\n학생 메시지를 입력하세요: ").strip()
        if not message:
            print("❌ 메시지가 비어있습니다.")
            return
        
        print("\n⏳ 두 버전 실행 중...\n")
        
        try:
            # 기존 버전
            from src.agency.Tutor import AgencyTutor
            tutor_old = AgencyTutor(
                client=self.client,
                model=self.tutor.model,
                open=self.tutor.open
            )
            
            print("1️⃣ 기존 AgencyTutor 실행...")
            response_old, tokens_old, _, _, intent_old, assessment_old = tutor_old.get_response(
                conversation_history=self.conversation_history,
                messages_student=message,
                max_tokens=1500
            )
            
            print("2️⃣ 새 AgencyTutorLG 실행...")
            response_new, tokens_new, _, _, intent_new, assessment_new = self.tutor.get_response(
                conversation_history=self.conversation_history,
                messages_student=message,
                max_tokens=1500
            )
            
            # 비교 결과
            print("\n" + "="*70)
            print("📊 비교 결과")
            print("="*70)
            
            print("\n🔵 기존 AgencyTutor:")
            print(f"  - 응답: {response_old[:100]}..." if len(response_old) > 100 else f"  - 응답: {response_old}")
            print(f"  - 토큰: {tokens_old}")
            print(f"  - Acts: {intent_old}")
            
            print("\n🟢 새 AgencyTutorLG:")
            print(f"  - 응답: {response_new[:100]}..." if len(response_new) > 100 else f"  - 응답: {response_new}")
            print(f"  - 토큰: {tokens_new}")
            print(f"  - Acts: {intent_new}")
            
            print("\n📈 차이:")
            print(f"  - 토큰 차이: {tokens_new - tokens_old} ({'+' if tokens_new > tokens_old else ''}{tokens_new - tokens_old})")
            print(f"  - 응답 길이: {len(response_new)} vs {len(response_old)}")
            print(f"  - Acts 동일: {'✅' if intent_new == intent_old else '❌'}")
            
            print("\n" + "─"*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def show_history(self):
        """대화 히스토리 표시"""
        print("\n" + "="*70)
        print("📜 대화 히스토리")
        print("="*70)
        
        if not self.conversation_history:
            print("\n📭 대화 이력이 없습니다.")
            return
        
        print(f"\n총 {len(self.conversation_history)}개 메시지:\n")
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = "🧑‍🎓 학생" if msg["role"] == "user" else "🎓 튜터"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"{i}. {role}: {content}")
        
        print()
    
    def clear_history(self):
        """히스토리 초기화"""
        confirm = input("\n정말 히스토리를 초기화하시겠습니까? (y/n): ").strip().lower()
        if confirm == 'y':
            self.conversation_history = []
            print("✅ 히스토리가 초기화되었습니다.")
        else:
            print("❌ 취소되었습니다.")
    
    def show_session_info(self):
        """세션 정보 표시"""
        duration = datetime.now() - self.session_start
        
        print("\n" + "="*70)
        print("📊 세션 정보")
        print("="*70)
        print(f"\n⏰ 세션 시작: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  경과 시간: {duration.seconds // 60}분 {duration.seconds % 60}초")
        print(f"💬 대화 턴: {len(self.conversation_history) // 2}턴")
        print(f"💰 총 토큰 사용: {self.total_tokens}")
        print(f"🤖 모델: {self.tutor.model}")
        print(f"🔓 개방형 모드: {self.tutor.open}")
        print()
    
    def run(self):
        """메인 실행 루프"""
        print("\n" + "🚀 " * 30)
        print("  교육 에이전트 대화형 테스트 도구 시작!")
        print("🚀 " * 30 + "\n")
        
        # 튜터 초기화
        self.initialize_tutor()
        
        while True:
            self.display_menu()
            choice = input("선택 > ").strip().lower()
            
            if choice == "1":
                self.test_single_message()
            elif choice == "2":
                self.test_conversation()
            elif choice == "3":
                self.test_streaming()
            elif choice == "4":
                self.test_debug_mode()
            elif choice == "5":
                self.compare_versions()
            elif choice == "h":
                self.show_history()
            elif choice == "c":
                self.clear_history()
            elif choice == "s":
                self.show_session_info()
            elif choice == "q":
                print("\n👋 테스트 도구를 종료합니다.")
                self.show_session_info()
                break
            else:
                print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")


def main():
    """메인 함수"""
    try:
        tester = InteractiveTester()
        tester.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

