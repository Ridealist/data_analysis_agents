# By Romain Puech
# 한국어 버전 수정 - By Junbo Koh

from src.agency.prompts.TutorActTaxonomy import TutorAct
import json


# class IntentSelector():
#     """학생 상태(Student State Tracing)에 따라 튜터의 다음 교육 의도를 결정(Intent Selection)"""

#     def __init__(self, intent_history=[]) -> None:
#         self.previous_intent = intent_history[-1] if intent_history != [] else [Intent.S_STRATEGY]

#     def get_intent(self, assessment, open=True):
#         """학생 평가를 바탕으로 다음 교육 의도를 선택"""
#         print("selecting intent... ")
#         previous_intent = self.previous_intent
#         assessment_codes = self.extract_assessment_codes(assessment)
#         intents = self.get_intent_aux(previous_intent, assessment_codes, open=open)
#         print(f"selected intent: {intents}")
#         return intents
    
#     def extract_assessment_codes(self, assessment):
#         """평가 결과에서 코드를 추출"""
#         json_data = json.loads(assessment)
#         selection = json_data['selection']
#         codes = list(selection)
#         return codes
        
#     def get_intent_aux(self, previous_intent, assessment_codes, open=True):
#         """핵심 의도 선택 로직 - Productive Failure 전략 기반"""
#         if not open and 'j' in assessment_codes:
#             assessment_codes.append('k')
#         print(f"평가 코드: {assessment_codes}, 이전 의도: {previous_intent}")
#         intents = []
        
#         # 이전 의도와 무관한 선택들:
#         if Intent.G_GREETINGS in previous_intent:
#             intents.append(Intent.G_GREETINGS)
#         if Intent.P_REFLECTION in previous_intent:
#             intents.append(Intent.G_GREETINGS)
#         if 'k' in assessment_codes:  # 완전한 해답 도달
#             intents.append(Intent.P_REFLECTION)
#         if 'i' in assessment_codes:  # 계산 요청
#             intents.append(Intent.S_OFFLOAD)
#         if 'h' in assessment_codes:  # 정리/정의 요청
#             intents.append(Intent.S_STATE)
#         if 'e' in assessment_codes and 'g' not in assessment_codes:  # 불명확한 답변
#             intents.append(Intent.P_ARTICULATION)
#         if any(code in assessment_codes for code in ('a','b','c')):  # 오류 발생
#             intents.append(Intent.S_SELFCORRECTION)
#         elif 'd' in assessment_codes:  # 직관적/불완전한 해답
#             intents.append(Intent.S_STRATEGY)
#         if 'j' in assessment_codes and 'k' not in assessment_codes and Intent.G_GREETINGS not in previous_intent:
#             intents.append(Intent.P_LIMITS)  # 해답의 한계 탐구
        
#         # 이전 의도에 따른 선택들:
#         if Intent.S_SELFCORRECTION in previous_intent:
#             if 'f' in assessment_codes:  # 올바른 답변
#                 intents.append(Intent.S_STRATEGY)
#             elif 'a' in assessment_codes or 'b' in assessment_codes:  # 여전한 오류
#                 intents.append(Intent.S_CORRECTION)
#             if 'c' in assessment_codes:  # 계산 오류
#                 intents.append(Intent.S_OFFLOAD)

#         if Intent.S_STRATEGY in previous_intent:
#             if 'g' in assessment_codes and 'd' not in assessment_codes and 'f' not in assessment_codes:
#                 intents.append(Intent.P_HYPOTHESIS)  # 직관적 접근 유도
#             elif ('l' in assessment_codes or 'k' in assessment_codes) and 'g' in assessment_codes:
#                 intents.append(Intent.S_HINT)  # 동기 부족시 힌트 제공
                
#         if Intent.P_HYPOTHESIS in previous_intent:
#             if 'f' in assessment_codes:  # 직관 제시 성공
#                 intents.append(Intent.P_ARTICULATION)
                
#         if len(intents) == 0 and ('f' in assessment_codes or 'g' in assessment_codes):
#             intents.append(Intent.S_STRATEGY)
        
#         # 충돌 방지를 위한 조정
#         if Intent.S_CORRECTION in intents and Intent.S_SELFCORRECTION in intents:
#             intents.remove(Intent.S_SELFCORRECTION)

#         if Intent.S_SELFCORRECTION in intents and Intent.S_OFFLOAD in intents:
#             intents.remove(Intent.S_SELFCORRECTION)
        
#         self.previous_intent = intents
#         return intents
        
        

# class StrategyIntentSelector(IntentSelector):

#     def get_intent_aux(self, previous_intent, assessment_codes, open=True):

#         if Intent.G_GREETINGS in previous_intent:
#             intent = [Intent.G_GREETINGS]
#         elif 'k' in assessment_codes:  # 완전한 해답
#             intent = [Intent.G_GREETINGS]
#         elif 'j' in assessment_codes and not open:  # 문제 해결 완료
#             intent = [Intent.G_GREETINGS]
#         else:
#             intent = [Intent.S_STRATEGY]

#         self.previous_intent = intent
#         return intent


class LLMIntentSelector():
    """LLM 기반 Intent Selector - GPT를 사용하여 의도를 선택"""

    def __init__(self,client,intent_history=[]) -> None:
        self.client = client
        self.previous_intent = intent_history[-1] if intent_history != [] else [TutorAct.MEL_EXPLORE]

    def get_intent(self,assessment,open=True):
        print("selecting intent... ")
        previous_intent = self.previous_intent
        assessment_codes = self.extract_assessment_codes(assessment)
        intents = self.get_intent_aux(self.client,previous_intent,assessment_codes,open=open)
        print(f"selected intent: {intents}")
        return intents
    
    def extract_assessment_codes(self, assessment):
        """평가 결과에서 코드를 추출"""
        json_data = json.loads(assessment)
        selection = json_data['selection']
        codes = list(selection)
        return codes

    def get_prompt(self,previous_intent,assessment_codes):
        prompt = \
        f"""학생과 교육 에이전트가 생성형 AI를 활용한 데이터 분석 학습을 진행하고 있습니다. 
교육 에이전트는 학생이 AI와 건강한 협력 관계를 구축하도록 지원하며, AI에 과도하게 의존하거나 불신하지 않고 주도적으로 학습하도록 돕습니다.

교육 에이전트는 '학습자 주도성(Student Agency)' 원칙에 따라 행동을 선택합니다. 
이 접근법의 핵심 목표는:
1. 학생이 생성형 AI를 도구로 활용하되, 학습의 주체로서 책임감을 유지하도록 지원
2. 프롬프트 리터러시를 함양하여 AI와 효과적으로 상호작용하도록 안내
3. AI 결과를 비판적으로 검토하고, 자신의 학습 과정을 성찰하도록 격려

다음 *TutorAct 분류체계*를 바탕으로 교육 에이전트의 다음 행동에 가장 적합한 의도를 선택하세요:

*TutorAct 분류체계*:
##
[자기효능감 - Self-Efficacy]
    SE_ACKNOWLEDGE : 학습자의 AI 활용/과제 해결 방식을 긍정적으로 인정
    SE_GROWTH : 성과와 발전을 구체적으로 강조하여 성장 인식
    SE_CONFIDENCE : 어려운 과제도 단계적 접근으로 해결 가능함을 확신

[메타인지-수행-리터러시 - Metacognition-Execution-Literacy]
    MEL_EXPLORE : 다양한 질문(프롬프트) 방식을 탐색하도록 유도
    MEL_REFINE : AI 결과에 따라 프롬프트를 수정/개선하도록 판단 지원
    MEL_VERIFY : AI 결과를 비판적으로 검증(출처, 사실 여부 등)하도록 요청

[메타인지-수행-책임의식 - Metacognition-Execution-Responsibility]
    MER_OWNERSHIP : 학습 주체로서 노력을 AI에 위임하지 않도록 인지
    MER_BOUNDARY : AI를 사용하지 말아야 할 시점/상황을 스스로 결정

[내재적 동기 - Intrinsic Motivation]
    IM_DISCOVERY : 새로운 지식 발견의 즐거움과 가치를 강조
    IM_ACHIEVEMENT : 과제 완수의 성취감을 충분히 느끼도록 지원
##

학생의 마지막 메시지에 대한 평가는 다음과 같습니다: {assessment_codes}.

평가와 튜터의 마지막 의도를 바탕으로, 튜터의 다음 메시지에 가장 적합한 의도 세트를 선택하세요.
단계별로 진행하세요. 먼저 선택을 간단히 정당화한 다음, *분류체계*에서 선택된 의도들을 포함하는 목록을 제공하세요.

다음 JSON 형식으로 답변하세요:
##
{{
    "justification": "TutorAct 선택 근거...",
    "selected_acts": ["주요 Act", "보조 Act (선택적)"],
}}
##
{{"""
        return prompt
    
    def get_intent_aux(self,client, previous_intent,assessment_codes, open=True):
        print("LLM Intent Selector called")
        client = self.client
        prompt = self.get_prompt(previous_intent,assessment_codes)
        intents=[]

        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages = [{"role":"system", "content":prompt}],
            max_tokens=300, # should never be reached
            temperature=0,
            top_p=0.1
        )

        response = (completion.choices[0].message.content).replace("\\(","$").replace("\\)","$").replace("\\[","$$").replace("\\]","$$").replace("\\","")
        print("The LLM intent selector thinks:")
        print(response)
        # postprocess make sure the format is right
        s = response.find('{')
        e = response.rfind('}')
        response = response[s:e+1]
        json_data = json.loads(response)
        strintents_list = json_data['selected_acts']
        for strintent in strintents_list:
            try:
                intent = TutorAct[strintent]
                intents.append(intent)
            except:
                pass


        self.previous_intent = intents
        return intents
