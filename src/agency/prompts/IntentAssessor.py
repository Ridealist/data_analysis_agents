# by Romain Puech
# 한국어 버전으로 수정 - Korean Version By Junbo Koh

from src.agency.utils.utils import print_logs
import json

class IntentAssessor():
    """Student State Assessor - 학생의 발화를 분석하여 학습 상태를 평가"""

    def __init__(self, client, model, assessment_history=[]) -> None:
        self.client = client
        self.model = model
        self.history = list(assessment_history)

    def get_purpose(self, conversation_history, student_prompt):
        purpose = \
        f"""학생이 생성형 AI를 활용하여 데이터 분석 과제를 수행하고 있습니다.

다음은 학생과 생성형 AI 간의 대화 기록입니다:
{conversation_history}

학생의 최신 프롬프트:
"{student_prompt}"      

위 학생의 프롬프트를 분석하여 다음 항목 "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p" 중 가장 해당 가능성이 높은 상위 2개의 학습 상태 지표를 선택하세요:

**중요**: 반드시 정확히 2개만 선택하세요. 더 많거나 적으면 안 됩니다.

**분석 접근의 오류 (Analysis Approach Errors)**
(a) 과제 전체 위임: AI에게 데이터 분석 과제 전체를 대신 수행하도록 요청하는 경우
(b) 검증 없는 수용: AI가 제시한 분석 결과나 해석을 비판적 검토 없이 그대로 받아들이는 경우
(c) 맥락 정보 누락: 데이터의 변수 설명, 분석 목적 등 필수 정보 없이 모호한 프롬프트를 작성하는 경우
(d) 과정 생략 요구: 분석 과정과 방법론 이해 없이 최종 결과물만 요구하는 경우
(e) 불명확한 지시: 프롬프트가 모호하거나 데이터 분석의 방향성이 명확하지 않은 경우

**학습 지원 요청 (Learning Support Requests)**
(f) 분석 방법 질문: 특정 데이터에 어떤 통계 기법을 적용해야 하는지 직접 묻는 경우
(g) 전체 코드 요청: 분석에 필요한 전체 코드를 처음부터 끝까지 작성해달라고 요구하는 경우
(h) 해석 대체 요구: 분석 결과의 의미를 스스로 해석하지 않고 AI에게 전적으로 의존하는 경우

**긍정적 학습 행동 (Positive Learning Behaviors)**
(i) 단계적 학습 접근: 기초부터 차근차근 이해하려 하며 단계별로 질문을 발전시키는 경우
(j) 결과 검증 시도: AI 분석 결과의 타당성을 확인하거나 검증 방법을 묻는 경우
(k) 프롬프트 개선: 피드백을 반영해 더 명확하고 구체적인 프롬프트로 수정하는 경우

**부정적 학습 행동 (Negaitive Learning Behaviors)**
(l) 무관한 요청: 데이터 분석 및 학습 목표와 직접 관련이 없는 질문이나 지시를 하는 경우
(m) 비협조적 태도: 지나치게 무례하거나 상대방(튜터/AI)에 대한 배려가 부족한 발화를 하는 경우

**학습 목표 달성 (Learning Goal Achievement)**
(n) 독립적 문제 해결: AI의 도움을 받되 최종적으로는 스스로 데이터 분석을 완성한 경우

**정서적 상태 (Emotional State)**
(o) 학습 동기 부족: 최소한의 노력만 하려 하거나 과제에 대한 회피적 태도를 보이는 경우
(p) 자신감 결여: 데이터 분석 능력에 대한 과도한 불안이나 AI 의존성을 표현하는 경우

단계별로 분석하세요:
1. 학생 프롬프트의 주요 의도와 특징을 파악
2. 각 지표별로 해당 여부를 판단
3. 선택한 항목들을 간략히 설명

**IMPORTANT**: 반드시 다음 JSON 형식으로 답변하세요:
##
{{
    "justification": "학생 프롬프트 분석 근거와 선택한 지표들의 이유",
    "selection": "선택한 알파벳들을 쉼표로 구분 (예: a,c,e)",

}}
##
Exactly follow JSON format.
Analyze the last student's utterance.
{{"""
        return purpose

    def create_prompt(self, conversation_history, student_prompt):
        """프롬프트 생성 - 대화 기록과 함께 평가 요청"""
        purpose = self.get_purpose(conversation_history, student_prompt)
        prompt = [
            {"role": "system", "content": "You are an expert educational assessor specializing in analyzing students' learning behaviors when using generative AI for data analysis."},
            {"role": "user", "content": purpose}
            ]
        #     [
        #     msg for i in range(len(conversation_history))
        #         for msg in (
        #             {"role": "user", "content": f"튜터: \"{conversation_history[i]}\"\n학생: \"{student_prompt[i]}\"\n"},
        #             {"role": "user", "content": purpose}
        #         )
        #     ]
        # prompt.append({"role": "user", "content":purpose })
        return prompt

    def assess(self, conversation_history, student_prompt):
        prompt = self.create_prompt(conversation_history, student_prompt)
        #print("Assessor called with prompt:")
        #print(print_logs(prompt))
        client = self.client
        completion = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            response_format = {"type": "json_object"},
            max_tokens=300, # should never be reached
            temperature=0.0,
            top_p=0.1
        )

        response = (completion.choices[0].message.content).replace("\\(","$").replace("\\)","$").replace("\\[","$$").replace("\\]","$$").replace("\\","")
        #print("ASSESSMENT IS \n",response)
        # postprocess make sure the format is right
        s = response.find('{')
        e = response.rfind('}')
        response = response[s:e+1]
        json_data = json.loads(response)
        selection = json_data['selection']
        selection = selection.replace(' ','').replace(',','')
        json_data['selection'] = selection
        assessment = json.dumps(json_data)

        self.history.append(assessment)

        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        
        return assessment, prompt_tokens, completion_tokens
    

# class NoJustificationAssessor(Assessor):
#     """정당화 설명(justification)이 없는 간단한 Assessor"""

#     def get_purpose(self, pb, sol):
#         purpose = \
#         f"""학생과 튜터가 물리학 역학 단원 문제를 함께 풀고 있습니다:

# *문제 진술*:
# ##
# {pb}
# ##

# 이 문제의 *제공된 해답*은 다음과 같습니다:
# ##
# {sol}
# ##

# 튜터의 발언은 "튜터:"로 시작하고 학생의 발언은 "학생:"으로 시작합니다.

# 학생의 마지막 발언을 분석하세요.
# "a,b,c,d,e,f,g,h,i,j,k" 중에서 해당하는 모든 피드백을 선택하세요:

# a) 학생이 잘못된 방법을 사용하거나 제안하고 있거나, 문제를 푸는 잘못된 접근을 하고 있음
# b) 학생이 문제에 대한 풀이로 제시한 자유물체도 그림에서 실수를 범함
# c) 학생이 이유에 대한 명시적 대답없이 자유물체도 그림만 제시하거나, 자유물체도 그림 없이 이유에 대해서만 대답함
# d) 학생의 답변이 명확하지 않거나 모호함

# e) 학생이 문제를 어떻게 풀어야 하는지 명시적으로 질문함
# f) 학생이 자유물체도 그림을 어떻게 그려야 하는지 명시적으로 질문함
# g) 학생이 문제에 적용되는 특정 물리 법칙을 알려달라고 튜터에게 명시적으로 요청함

# h) 학생이 튜터의 이전 질문에 올바르게 답변함
# i) 학생과 튜터가 초기 *문제 진술* 전체에 대한 완전 해답에 도달함

# j) 학생이 강한 학습 동기 부족을 보임
# k) 학생이 강한 자신감 부족을 보임

# **IMPORTANT**: MUST Answer in the following JSON format:
# ##
# {{
#     "justification": "..",
#     "selection": ".."

# }}
# ##
# Exactly follow JSON format.
# Analyze the last student's utterance.
# {{"""
#         return purpose
    
# class ShortMemoryAssessor(Assessor):
#     """단기 기억 Assessor - 최근 3개 대화만 고려"""

#     def assess(self, pb, sol, student_messages, tutor_messages):
#         if len(student_messages) > 3:
#             student_messages = student_messages[-3:]
#         if len(tutor_messages) > 3:
#             tutor_messages = tutor_messages[-3:]

#         return super().assess(pb, sol, student_messages, tutor_messages)
    
#     def create_prompt(self,pb,sol,tutor_messages,student_messages):
        
#         purpose = self.get_purpose(pb,sol)
#         prompt = [{"role": "system", "content": purpose}]
#         extension = \
#             [
#             msg for i in range(len(tutor_messages)-1)
#                 for msg in (
#                     {"role": "user", "content": f"튜터: \"{tutor_messages[i]}\"\n학생: \"{student_messages[i]}\"\n"},
#                     {"role": "assistant", "content": self.history[-len(tutor_messages)+1+i]}
#                 )
#             ]
#         prompt = prompt + extension
#         prompt.append({"role": "user", "content":f"튜터: \"{tutor_messages[-1]}\"\n학생: \"{student_messages[-1]}\"\n" })
#         #print("Short mem prompt len is:",len(''.join([msg["content"] for msg in prompt])))
#         return prompt
    
# class EndAssessor(Assessor):
#     """종료 Assessor - 문제 완료 여부만 확인"""

#     def get_purpose(self, pb, sol):
#         purpose = \
#         f"""학생과 튜터가 물리학 역학 단원 문제를 함께 풀고 있습니다:

# *문제 진술*:
# ##
# {pb}
# ##

# 이 문제의 *제공된 해답*은 다음과 같습니다:
# ##
# {sol}
# ##

# 튜터의 발언은 "튜터:"로 시작하고 학생의 발언은 "학생:"으로 시작합니다.

# 학생의 마지막 발언을 분석하세요.
# "j,k" 중에서 해당하는 모든 피드백을 선택하세요:

# j) 학생과 튜터가 초기 *문제 진술*의 모든 질문에 대한 완전한 해답에 도달함
# k) 학생과 튜터가 초기 *문제 진술*의 모든 질문에 대해 *제공된 해답*의 방법과 동등한 완전한 해답에 도달함

# Proceed step by step. First briefly justify your selection, then provide a string containing the selected letters.

# **IMPORTANT**: MUST Answer in the following JSON format:
# ##
# {{
#     "justification": "..",
#     "selection": ".."

# }}
# ##
# Exactly follow JSON format.
# Analyze the last student's utterance.
# {{"""
#         return purpose
