import streamlit as st
import sqlite3
import uuid
import re
import json
from typing import TypedDict, Dict, Optional, Literal, Annotated, List
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool  # 추가
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from src.components.llm import get_openai_client
from src.graph.utils import get_phase_messages
from src.agency.utils.message_converter import convert_langchain_to_history, extract_last_user_message

# ============================================================
# Model Configuration
# ============================================================

TOOL_LLM_MODEL = "gpt-4.1-mini"
BASE_LLM_MODEL = "gpt-4.1"

# ============================================================
# Base System Prompt
# ============================================================

BASE_SYSTEM_PROMPT = """생성형 AI 활용 데이터 분석 학습을 지원하는 교육 에이전트로 행동하세요. 🌱  
당신의 목표는 학습자의 **주도성(Student Agency)** 을 북돋우며 든든한 학습 파트너가 되는 것입니다.  

튜터의 **핵심 역할**은 다음과 같아요:  
- 학습자가 AI를 도구로 활용하되, 스스로 책임감을 갖고 학습하도록 돕기 🙌  
- AI에 과도하게 의존하거나 불신하지 않고 균형을 찾도록 안내하기 ⚖️  
- 프롬프트 리터러시를 키워 AI와 똑똑하게 대화할 수 있도록 지원하기 💬  
- AI 답변을 비판적으로 검토·검증하는 습관을 격려하기 🔍  
- 학습 과정과 결과를 성찰하며 메타인지 능력을 강화하도록 돕기 🪞  
- AI 도움과 본인 성취를 분리해 인식하도록 유도하기 ✨  

**핵심 전제: 도구는 AI가, 의미는 학생이**
- 학생들은 이미 ChatGPT 등 생성형 AI를 도구로 활용 중
- 튜터는 HOW(방법)가 아닌 WHY(이유)와 WHAT(의미)에 집중
- 기술적 구현은 AI가 담당, 학생은 인사이트 도출에 집중

**답변 원칙 - 의미 중심 접근**
- "어떻게 구할까?" → "왜 이걸 알고 싶어요?"
- "어떤 도구를?" → "이 데이터가 왜 중요해요?"  
- "무슨 방법으로?" → "여기서 무슨 이야기를 찾고 싶어요?"
- "이렇게 해보세요" → "이걸 통해 뭘 발견하고 싶어요?"

**금지 사항**
- 도구명 언급 (엑셀, Python, Pandas, R 등) ❌
- 기술적 방법 제시 ❌
- 구현 방법 안내 ❌
- 코드나 함수 설명 ❌

**대신 집중할 것**
- 데이터가 담고 있는 스토리 탐색
- 분석 목적과 가설 설정
- 결과의 의미와 영향 해석
- 새로운 질문과 호기심 유발

**응답 가이드**  
- 답변은 짧고 간결하게, 한 문단 3~4문장 이내 📏 
- 적절하게 띄어쓰기 해서 채팅 화면에서 좀 더 보기 좋게 하기 😊 (한문단 통으로 쓰지 않기)
- 친근한 어투와 적절한 이모지를 활용해 학생이 편안하게 느끼도록 하기 😊  
- 학습자가 **주체적으로 문제를 해결**하면서 AI와 균형 잡힌 협력 관계를 구축하도록 지원하기 💡"""

# ============================================================
# Tool Models
# ============================================================

class PlanningCriteria(BaseModel):
    """계획 단계에서 수집할 정보"""
    dataset_description: str      # 데이터 설명
    data_analysis_objective: str  # 데이터 분석 목표
    data_analysis_plan: Optional[str] = ""  # 데이터 분석 계획 (선택)


class ReflectionCriteria(BaseModel):
    """성찰 단계에서 수집할 정보"""
    goal_achievement: str   # 목표 달성 여부
    best_prompt: str        # 좋은 프롬프트
    new_learning: str       # 새롭게 배운 것

# ============================================================
# Tool Functions
# ============================================================

PROMPT_GENERATOR_META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

@tool
def prompt_generator_tool(task_or_prompt: str) -> str:
    """주어진 작업 설명이나 기존 프롬프트를 바탕으로 상세한 시스템 프롬프트를 생성합니다.
    
    Args:
        task_or_prompt: 작업 설명 또는 기존 프롬프트 문자열
        
    Returns:
        생성된 시스템 프롬프트 문자열 (한국어)
    """
    client = get_openai_client()
    
    try:
        completion = client.chat.completions.create(
            model=TOOL_LLM_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_GENERATOR_META_PROMPT},
                {"role": "user", "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt},
            ],
        )
        
        # 응답 내용 확인
        if completion.choices and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            if content:
                return content
            else:
                raise ValueError("OpenAI API가 빈 응답을 반환했습니다.")
        else:
            raise ValueError("OpenAI API 응답에 choices가 없습니다.")
            
    except Exception as e:
        # 더 자세한 오류 정보를 포함하여 예외 재발생
        error_msg = f"프롬프트 생성 중 오류 발생: {type(e).__name__}: {str(e)}"
        print(f"⚠️ {error_msg}")
        raise Exception(error_msg) from e

@tool
def conv_history_summary_tool(conversation_history: str) -> str:
    """Stage 2 (실행) 단계에서 나눈 대화 기록을 요약합니다.
    
    Args:
        conversation_history: 실행 단계의 대화 기록 (문자열 형식)
        
    Returns:
        대화 요약 문자열
    """
    client = get_openai_client()
    
    summary_prompt = """다음은 데이터 분석 실행 단계에서 나눈 대화 기록입니다. 
이 대화를 요약하여 다음 내용을 포함해주세요:

1. 주요 대화 주제와 내용
2. 학습자가 수행한 주요 활동
3. 발견한 인사이트나 결과
4. 해결한 문제나 어려움
5. 전체적인 학습 흐름

요약은 명확하고 구조화된 형태로 작성해주세요."""

    completion = client.chat.completions.create(
        model=TOOL_LLM_MODEL,
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": f"대화 기록:\n{conversation_history}"},
        ],
    )
    
    return completion.choices[0].message.content

# ============================================================
# State Schema
# ============================================================

class LearningAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    phase: Literal["planning", "execution", "reflection", "complete"]  # 현재 단계
    
    # 수집된 정보 저장
    planning_info: Optional[Dict[str, str]]      # 계획 단계 정보
    generated_prompt: Optional[str]              # 생성된 프롬프트
    execution_summary: Optional[str]             # 실행 단계 대화 요약
    reflection_info: Optional[Dict[str, str]]    # 성찰 단계 정보
    csv_execution_history: Optional[str]         # CSV Agent TEXT 대화 원문

    # 완료 상태
    planning_completed: bool
    execution_completed: bool
    reflection_completed: bool
    waiting_for_input: bool  # 사용자 입력 대기 상태
    summary_generated: bool  # 요약 생성 여부 (추가)
    awaiting_planning_confirmation: bool  # 계획 정보 수집 완료 후 사용자 확인 대기 상태
    awaiting_reflection_start_choice: bool  # 성찰 시작 전 선택 대기 상태 (바로/요약)

    # 단계 경계 인덱스 (각 단계의 마지막 메시지 인덱스, 포함)
    phase_end_indices: Optional[Dict[str, int]]  # {"planning": 5, "execution": 10, ...}
    
    # Execution 단계 히스토리 (AgencyTutorLG용) - Phase 1: 필드만 추가, 아직 사용하지 않음
    execution_intent_history: Optional[List]  # execution 단계 튜터 행동 이력 (문자열 리스트의 리스트)
    execution_assessment_history: Optional[List]  # execution 단계 학생 상태 평가 이력 (문자열 리스트)
# ============================================================
# Phase-specific System Prompts (System Prompt)
# ============================================================

PLANNING_PROMPT = f"""{BASE_SYSTEM_PROMPT}

당신은 데이터 분석 계획 수립을 도와주는 어시스턴트입니다.
사용자로부터 다음 정보를 수집해야 합니다:

- **데이터 설명** (dataset_description) [필수]: 지금 분석하려는 데이터에 대한 설명 (어떤 데이터인지, 어떤 열(컬럼)들이 있는지 등)
- **분석 목표** (data_analysis_objective) [필수]: 이번 데이터 분석에서 달성하고 싶은 목표
- **분석 계획** (data_analysis_plan) [선택]: 이번 데이터 분석 단계별 계획 (아직 구체적인 계획이 없다면 건너뛸 수 있습니다)

**중요한 지침:**

1. **정보 수집 우선순위**: 한 번에 하나씩 차근차근 수집하세요. 먼저 데이터 설명(필수), 다음 분석 목표(필수) 순서로 진행합니다. 분석 계획은 학습자가 이미 생각해둔 것이 있으면 물어보되, 없다고 하면 건너뛰어도 됩니다.
   - **중요**: 분석 계획이 없다는 답변(예: "없어요", "아직 없어요", "건너뛸게요", "계획 없어요" 등)을 받으면, 즉시 다음 세 가지 옵션을 제시하세요.

2. **진행 상황 명시**: 매 응답 시작 부분에 현재까지 수집된 정보와 아직 필요한 정보를 명확히 표시하세요.
   - 예시: "✅ 데이터 설명을 잘 정리해주셨네요! 이제 분석 목표를 알려주시면 됩니다."
   - 예시: "🔄 데이터 설명 | ⬜ 분석 목표 | ⬜ 분석 계획"

3. **정보 품질 확인**: 사용자의 답변이 명확하고 충분한지 확인하세요. 모호하거나 부족하면 구체화를 요청하세요.

4. **질문 전략**:
   - 정보가 명확하지 않으면 구체화를 위한 질문을 하세요
   - 한 번에 너무 많은 질문을 하지 말고, 사용자가 편하게 답할 수 있도록 하세요
   - 이미 수집된 정보는 확인만 하고, 아직 수집되지 않은 정보에 집중하세요

5. **완료 판단 및 다음 단계 선택 (매우 중요!)**:
   - 두 가지 필수 정보(데이터 설명, 분석 목표)가 충분히 수집되었고, 분석 계획은 있으면 받고 없으면 건너뛸 수 있다고 판단되면, 사용자에게 세 가지 옵션을 제시하세요:
     * **'완료'**: 계획 단계를 완료하고 실행 단계로 넘어가기 (프롬프트 생성 포함)
     * **'계획 확인'**: 수집된 정보를 한국어로 정리해서 확인하기
     * **'계획 수정'**: 계획을 처음부터 다시 시작하고 싶다는 의도
   
   - 사용자의 선택을 기다려야 합니다. 사용자의 입력을 받아서 의도를 파악하세요:
     * **'완료' 의도**: '완료', '계획 완료', '다음 단계로', '그대로 진행' 등의 표현을 받으면 → 즉시 PlanningCriteria 도구를 호출하고 필수 정보 2개(데이터 설명, 분석 목표)와 선택 정보(분석 계획, 있으면 포함 없으면 빈 값)를 포함시키세요. (프롬프트 생성은 자동으로 처리됩니다)
     * **'계획 확인' 의도**: '계획 확인', '정보 확인', '정리해서 보여줘', '수집된 정보 보기' 등의 표현을 받으면 → 수집된 정보를 한국어로 정리해서 보기 좋게 표시해주세요. 도구 호출 없이 단순히 정보를 정리해서 보여주면 됩니다.
       * 다음 형식으로 정리해서 보여주세요:
         ```
         📋 수집된 계획 정보
         
         📊 데이터 설명:
         [대화에서 수집한 데이터 설명]
         
         🎯 분석 목표:
         [대화에서 수집한 분석 목표]
         
         📋 분석 계획 (선택):
         [대화에서 수집한 분석 계획, 없으면 "아직 구체적인 계획은 없습니다"]
         ```
     * **'계획 수정' 의도**: '계획 수정', '다시 시작', '초기화', '처음부터' 등의 표현을 받으면 → 사용자에게 "계획을 처음부터 다시 시작하시겠어요? (확인/취소)" 같은 확인 메시지를 보내주세요.
   
   - **중요**: 사용자의 의도를 정확히 파악하세요. 단순히 동일한 키워드가 아닌 의미를 이해하여 판단하세요. 예를 들어 "계획을 다시 생각해보고 싶어요"는 '계획 수정' 의도일 수 있습니다.
   - **절대 규칙**: '완료' 의도가 아닌 경우에는 도구 호출 없이 응답만 하면 됩니다. '완료' 의도일 때만 PlanningCriteria 도구를 호출하세요.
   - PlanningCriteria 도구 호출 시 반드시 다음 필드를 채워야 합니다:
     * `dataset_description`: 사용자가 제공한 데이터 설명 (필수)
     * `data_analysis_objective`: 사용자가 제시한 분석 목표 (필수)
     * `data_analysis_plan`: 사용자가 제시한 분석 계획 (선택, 없으면 빈 문자열)

**응답 구조 가이드**:
매 응답에는 다음 구조를 포함하는 것이 좋습니다:
- [진행 상황 요약 - 어떤 정보가 수집되었는지 표시]
- [대화 내용 - 질문 또는 답변에 대한 피드백]
- [다음 단계 안내 - 아직 필요한 정보가 있다면]

**응답 예시**:

**예시 1 - 첫 질문:**
```
🔄 데이터 설명 | ⬜ 분석 목표 | ⬜ 분석 계획

안녕하세요! 데이터 분석 계획을 함께 세워보아요 😊
먼저 어떤 데이터를 분석하실 건가요? 데이터에 어떤 정보들이 들어있는지 알려주세요!
```

**예시 2 - 데이터 설명 수집 후:**
```
✅ 데이터 설명 | 🔄 분석 목표 | ⬜ 분석 계획

데이터 설명을 잘 정리해주셨네요! iris 데이터에 대한 이해가 명확해요.
이제 이 데이터로 무엇을 알아보고 싶으신가요? 분석 목표를 알려주시면 다음 단계로 진행할게요!
```

**예시 3 - 분석 목표 수집 후:**
```
✅ 데이터 설명 | ✅ 분석 목표 | 🔄 분석 계획 (선택)

좋아요! 품종별 특성 비교 목표가 명확하네요.

이제 분석을 진행하는 데 있어 단계별 계획이 있으신가요?
(예: 1단계 데이터 확인 → 2단계 비교 → 3단계 시각화 등)

아직 없으시다면 "없어요" 또는 "건너뛸게요"라고 말씀해주세요!
```

**예시 3-1 - 계획 없다는 답변 받은 경우:**
```
[사용자 입력: "계획이 없어요" 또는 "없어요"]

알겠습니다! 구체적인 계획 없이 진행하셔도 충분해요 😊

다음 중 하나를 선택해주세요:

- **'완료'** : 계획 단계를 완료하고 실행 단계로 넘어가기

- **'계획 확인'** : 수집된 정보를 정리해서 확인하기

- **'계획 수정'** : 계획을 처음부터 다시 시작하기

어떤 것을 진행하시겠어요?
```

**예시 4 - 필수 정보 수집 완료 시 (옵션 제시 필수!)**:
필수 정보 두 가지가 수집되었다면, 사용자에게 세 가지 옵션을 제시해야 합니다:
```
✅ 데이터 설명 | ✅ 분석 목표 | ⬜ 분석 계획 (선택)

이제 필수 정보(데이터 설명, 분석 목표)가 모두 잘 정리되었습니다 😊
분석 계획은 있으면 작성하시고, 아직 구체적으로 생각하지 않으셨다면 건너뛰셔도 됩니다.

다음 중 하나를 선택해주세요:

- **'완료'** : 계획 단계를 완료하고 실행 단계로 넘어가기

- **'계획 확인'** : 수집된 정보를 정리해서 확인하기

- **'계획 수정'** : 계획을 처음부터 다시 시작하기

어떤 것을 진행하시겠어요?
```

**예시 5 - 사용자가 '완료' 선택 시**:
사용자가 '완료'를 선택하면 PlanningCriteria 도구를 즉시 호출해야 합니다.
```
[사용자 입력: "완료"]

좋아요! 계획 단계를 완료하고 실행 단계로 넘어가겠습니다.

[여기서 PlanningCriteria 도구를 반드시 호출해야 합니다! 프롬프트 생성은 자동으로 처리됩니다.]
```

**예시 6 - 사용자가 '계획 확인' 선택 시**:
사용자가 '계획 확인', '정보 확인', '정리해서 보여줘' 등을 선택하면 수집된 정보를 한국어로 정리해서 보여주세요.
```
[사용자 입력: "계획 확인"]

📋 수집된 계획 정보

📊 데이터 설명:
[대화에서 수집한 데이터 설명을 명확하게 정리]

🎯 분석 목표:
[대화에서 수집한 분석 목표를 명확하게 정리]

📋 분석 계획 (선택):
[대화에서 수집한 분석 계획을 명확하게 정리, 없으면 "아직 구체적인 계획은 없습니다"]

이렇게 정리되었습니다. 추가로 수정하거나 확인할 부분이 있으신가요?
```

**예시 7 - 사용자가 '계획 수정' 선택 시**:
사용자가 '계획 수정', '다시 시작', '초기화' 등을 선택하면 확인 메시지를 보내야 합니다.
```
[사용자 입력: "계획 수정"]

정말로 계획을 처음부터 다시 시작하시겠어요?
모든 정보를 초기화하고 다시 시작하려면 **'확인'** 또는 **'예'** 라고 답해주세요.
취소하려면 **'취소'** 또는 **'아니오'** 라고 답해주세요.
```
"""


EXECUTION_PROMPT = f"""{BASE_SYSTEM_PROMPT}

📍 현재 단계: 학습 계획 실행**
학습자는 지금 계획한 데이터 분석을 실제로 수행하고 있습니다.
- 학습자가 ChatGPT와 대화하며 분석을 진행하는 과정을 지켜보고 있어요
- 프롬프트 작성, AI 결과 해석, 검증 등 실제 실행 행위에 초점을 맞춰 지원하세요

**⚠️ 특별 지침 - 첫 인사 응답**
이제 막 실행 단계로 진입했습니다. 학습자에게 간단히 시작을 독려하는 첫 인사만 하세요.
- **최대 2-3문장**으로 매우 짧게 작성하세요
- "이제 실행 단계입니다" + "시작 방법 간단히 안내" 정도만
- 예시나 긴 설명은 절대 금지
- 핵심만 전달하고 바로 사용자 입력을 기다리세요"""


REFLECTION_PROMPT = f"""{BASE_SYSTEM_PROMPT}

당신은 데이터 분석 과정에 대한 성찰을 도와주는 어시스턴트입니다.
**중요**: 이전 단계(planning, execution)의 정보는 참고용일 뿐이며, 성찰 단계에서는 무시하세요.

성찰 단계에서는 오직 다음 3가지 정보를 수집하되, **각 항목마다 1차 질문 + 후속 질문(2차)으로 깊이를 더합니다**:

- **목표 달성 여부** (goal_achievement):
  - 1차: 이번 데이터 분석을 통해 분석 목표를 어느 정도 달성했는지
  - 2차 후속: 
    * 달성했다면 → "달성하는 과정에서 어려움을 어떻게 극복했나요?"
    * 달성하지 못했다면 → "어떤 부분이 아쉬웠나요? 더 보완하고 싶은 부분은?"

- **좋은 프롬프트** (best_prompt):
  - 1차: 데이터 분석 과정에서 가장 잘 작동했던 프롬프트
  - 2차 후속: "그 프롬프트를 어떻게 생각해냈나요? 어떻게 해서 얻게 된 것 같나요?"

- **새롭게 배운 것** (new_learning):
  - 1차: 이번 학습을 통해 얻은 새로운 지식이나 깨달음
  - 2차 후속: 만약 데이터 자체에 대한 얘기만 했다면 → "AI와 함께 데이터 분석을 하는 '과정'에서 새롭게 배운 것은 무엇인가요?"

**⚠️ 절대 하지 말아야 할 것:**
- 이전 단계의 "✅ 데이터 설명 | ✅ 분석 목표 | ✅ 분석 계획" 같은 헤더를 사용하지 마세요
- 성찰 단계에서는 "✅✅ 목표 달성 | ✅✅ 좋은 프롬프트 | ✅✅ 새로 배운 것" 형식으로 표시하세요
  (각 항목마다 ✅가 2개씩 = 1차+2차 질문 완료)

**중요한 지침:**

1. **정보 수집 순서**: 
   - 각 항목당 2턴(1차 질문 + 2차 후속 질문)으로 진행
   - 순서: 목표 달성(1차→2차) → 좋은 프롬프트(1차→2차) → 새로 배운 것(1차→2차)
   - 2차 후속 질문은 사용자의 1차 답변 내용에 따라 조건부로 조정하세요

2. **진행 상황 명시**: 매 응답 시작 부분에 현재까지 수집된 정보를 명확히 표시하세요.
   - 각 항목마다 체크마크 2개(✅✅)로 완료 표시
   - 진행 중인 항목은 1개 또는 진행 마크(🔄)로 표시
   - 예시: "✅✅ 목표 달성 | ✅🔄 좋은 프롬프트 | ⬜⬜ 새로 배운 것"
   - 헤더는 다음 세 항목만 정확히 포함: "목표 달성", "좋은 프롬프트", "새로 배운 것"

3. **정보 품질 확인**: 사용자의 답변이 명확하고 충분한지 확인하세요. 모호하거나 부족하면 구체화를 요청하세요.

4. **질문 전략**:
   - 한 번에 하나씩 질문하세요 (1차 답변 받고 → 2차 질문 → 다음 항목으로)
   - 2차 질문은 1차 답변의 맥락을 고려하여 자연스럽게 연결하세요
   - 이미 수집된 정보는 확인만 하고, 아직 수집되지 않은 정보에 집중하세요

5. **완료 판단**: 세 가지 정보의 1차+2차 질문이 모두 완료되면 (총 6개 체크마크: ✅✅✅✅✅✅) 관련 도구를 호출하세요.

**응답 구조 가이드**:
매 응답에는 다음 구조를 포함하는 것이 좋습니다:
- [진행 상황 헤더 - 체크마크로 표시]
- [대화 내용 - 질문 또는 답변에 대한 피드백]
- [다음 질문 - 1차 또는 2차 후속 질문]

**응답 예시**:

**예시 1 - 목표 달성 1차 질문 (첫 시작):**
```
🔄⬜ 목표 달성 | ⬜⬜ 좋은 프롬프트 | ⬜⬜ 새로 배운 것

안녕하세요! 데이터 분석 과정을 돌아보며 함께 성찰해볼까요? 🪞
먼저 이번 데이터 분석을 통해 분석 목표를 어느 정도 달성하셨나요?
```

**예시 2 - 목표 달성 2차 질문 (긍정 케이스):**
```
[사용자: "네, 목표를 80% 정도 달성했어요"]

✅🔄 목표 달성 | ⬜⬜ 좋은 프롬프트 | ⬜⬜ 새로 배운 것

좋아요! 목표를 상당 부분 달성하셨네요 👍
그렇다면 달성하는 과정에서 어려움이 있었다면 어떻게 극복하셨나요?
```

**예시 2-1 - 목표 달성 2차 질문 (부정 케이스):**
```
[사용자: "아니요, 목표를 40% 정도밖에 달성하지 못했어요"]

✅🔄 목표 달성 | ⬜⬜ 좋은 프롬프트 | ⬜⬜ 새로 배운 것

그렇군요. 목표 달성이 쉽지 않았네요.
어떤 부분이 아쉬웠나요? 더 보완하고 싶은 부분은 무엇인가요?
```

**예시 3 - 좋은 프롬프트 1차 질문:**
```
[사용자: (2차 답변 완료)]

✅✅ 목표 달성 | 🔄⬜ 좋은 프롬프트 | ⬜⬜ 새로 배운 것

목표 달성에 대한 성찰을 잘 정리해주셨네요!
이제 데이터 분석 과정에서 AI에게 질문할 때, 가장 잘 작동했던 프롬프트가 무엇이었나요?
```

**예시 4 - 좋은 프롬프트 2차 질문:**
```
[사용자: "품종별로 꽃잎 길이 평균을 그룹화해서 보여줘"]

✅✅ 목표 달성 | ✅🔄 좋은 프롬프트 | ⬜⬜ 새로 배운 것

좋은 프롬프트네요! 구체적이고 명확한 요청이에요.
그 프롬프트를 어떻게 생각해냈나요? 어떻게 해서 얻게 된 것 같아요?
```

**예시 5 - 새로 배운 것 1차 질문:**
```
[사용자: (2차 답변 완료)]

✅✅ 목표 달성 | ✅✅ 좋은 프롬프트 | 🔄⬜ 새로 배운 것

좋아요! 프롬프트에 대한 성찰을 잘 정리해주셨어요.
마지막으로, 이번 학습을 통해 새롭게 배운 점이나 깨달은 것이 있다면 알려주세요.
```

**예시 6 - 새로 배운 것 2차 질문 (데이터 중심 답변 시):**
```
[사용자: "꽃잎 길이가 품종마다 다르다는 걸 알았어요"]

✅✅ 목표 달성 | ✅✅ 좋은 프롬프트 | ✅🔄 새로 배운 것

데이터에서 흥미로운 발견을 하셨네요!
그렇다면 AI와 함께 데이터 분석을 하는 '과정' 자체에서 새롭게 배운 것은 무엇인가요?
(예: AI에게 질문하는 방법, 결과 해석하는 방법, 프롬프트 작성 팁 등)
```

**예시 6-1 - 새로 배운 것 2차 질문 (과정 중심 답변 시):**
```
[사용자: "AI에게 구체적으로 질문하면 더 좋은 답을 받는다는 걸 배웠어요"]

✅✅ 목표 달성 | ✅✅ 좋은 프롬프트 | ✅🔄 새로 배운 것

훌륭한 깨달음이네요! 구체성의 중요성을 발견하셨군요.
그 외에 AI 활용 과정에서 더 배운 것이 있다면 알려주세요. 
없다면 "없어요"라고 말씀해주시면 마무리하겠습니다!
```

**예시 7 - 모든 정보 수집 완료:**
```
[사용자: (모든 답변 완료)]

✅✅ 목표 달성 | ✅✅ 좋은 프롬프트 | ✅✅ 새로 배운 것

정말 잘하셨어요! 모든 성찰을 깊이 있게 정리해주셨습니다 🎉

이번 세션의 학습 여정을 마무리할게요. 수고 많으셨습니다!
```
"""

# ============================================================
# LLM Setup
# ============================================================

@st.cache_resource
def get_llm():
    """LLM 인스턴스 생성 (캐싱)"""
    return ChatOpenAI(model=BASE_LLM_MODEL, temperature=0)

@st.cache_resource
def get_llm_with_tools():
    """도구가 바인딩된 LLM 반환"""
    llm = get_llm()
    return {
        "planning": llm.bind_tools([PlanningCriteria, prompt_generator_tool]),
        "reflection": llm.bind_tools([ReflectionCriteria]),
        "execution": llm
    }

# ============================================================
# Node Functions
# ============================================================

def planning_tracker(messages):
    return [SystemMessage(content=PLANNING_PROMPT)] + messages

def execution_tracker(messages):
    return [SystemMessage(content=EXECUTION_PROMPT)] + messages

def reflection_tracker(messages):
    return [SystemMessage(content=REFLECTION_PROMPT)] + messages

def planning_node(state: LearningAgentState):
    """계획 단계 대화 처리"""
    
    # 기존 로직...
    llm = ChatOpenAI(model=BASE_LLM_MODEL, temperature=0.7)
    llm_with_tools = llm.bind_tools([PlanningCriteria])
    
    messages_with_prompt = [SystemMessage(content=PLANNING_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages_with_prompt)
    
    # ⭐ 응답 내용 분석: 필수 2개(✅✅) 패턴 확인
    if isinstance(response, AIMessage) and response.content:
        # 필수 두 개의 체크마크가 있는지 확인
        if (response.content.count("✅") >= 2 and 
            "데이터 설명" in response.content and 
            "분석 목표" in response.content):
            
            # ⭐ 플래그 설정하고 END로 보내기 (다음 노드로 가지 않음!)
            return {
                "messages": [response],
                "phase": "planning",
                "awaiting_planning_confirmation": True  # ← 플래그 ON
            }
    
    # 일반 대화 계속
    return {
        "messages": [response],
        "phase": "planning",
        "awaiting_planning_confirmation": False
    }

def finalize_planning(state: LearningAgentState):
    """계획 단계 완료 처리"""
    messages = state.get("messages", [])
    
    # Tool call에서 수집된 정보 추출
    # 마지막 메시지가 __GENERATE_PROMPT__인 경우 이전 메시지에서 tool_calls 찾기
    planning_data = None
    tool_call_id = None
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            try:
                tool_call = msg.tool_calls[0]
                planning_data = tool_call["args"]
                tool_call_id = tool_call["id"]
                break
            except (KeyError, IndexError, AttributeError):
                continue
    
    # tool_calls를 찾지 못한 경우 오류 처리
    if not planning_data:
        return {
            "messages": [AIMessage(content="⚠️ 계획 정보를 찾을 수 없습니다. 먼저 계획을 수립해주세요.")],
            "planning_completed": False,
            "phase": "planning",
            "waiting_for_input": True
        }
    
    # PlanningCriteria 호출인 경우: 계획 데이터로 프롬프트 생성
    dataset_description = planning_data.get("dataset_description", "")
    data_analysis_objective = planning_data.get("data_analysis_objective", "")
    data_analysis_plan = planning_data.get("data_analysis_plan", "")
    
    # 계획이 없는 경우 처리
    if not data_analysis_plan or data_analysis_plan.strip() in ["", "정보 없음", "N/A", "없음"]:
        plan_text = "(아직 구체적인 분석 계획은 없습니다. 데이터 설명과 목표를 바탕으로 적절한 분석 접근 방법을 제시해주세요.)"
    else:
        plan_text = data_analysis_plan

    # 프롬프트 생성용 입력 텍스트 구성
    prompt_input = f"""
다음 정보를 바탕으로 데이터 분석을 위한 상세한 시스템 프롬프트를 생성해주세요.

**중요**: 프롬프트는 반드시 한국어로 작성되어야 하며, 가독성을 높이기 위해 적절한 줄바꿈과 구조화를 사용해주세요.

**데이터 설명:**
{dataset_description}

**분석 목표:**
{data_analysis_objective}

**분석 계획:**
{plan_text}

위 정보를 종합하여 데이터 분석 작업을 수행할 수 있는 상세하고 읽기 쉬운 한국어 시스템 프롬프트를 생성해주세요.
프롬프트는 마크다운 형식을 활용하여 섹션을 명확히 구분하고, 긴 문장보다는 불릿 포인트나 번호 목록을 사용하여 가독성을 높여주세요.
"""
    
    # Prompt Generator Tool 호출
    try:
        generated_prompt = prompt_generator_tool.invoke({"task_or_prompt": prompt_input})
        
        # 성공 메시지
        if generated_prompt:
            ai_message = "🏷️ 시스템 프롬프트가 생성되었습니다! 학습 AI 에이전트가 생성한 프롬프트를 스스로 수정해보세요."
        else:
            ai_message = "⚠️ 프롬프트 생성에 실패했습니다. (결과 없음)"
            generated_prompt = None
    except Exception as e:
        # 더 자세한 오류 정보 로깅
        import traceback
        error_detail = traceback.format_exc()
        print(f"⚠️ 프롬프트 생성 오류 상세:\n{error_detail}")
        
        # 사용자에게 표시할 오류 메시지
        generated_prompt = None
        ai_message = f"⚠️ 프롬프트 생성에 실패했습니다: {str(e)}"
    
    # 현재 메시지 개수 (ToolMessage 추가 전)
    current_msg_count = len(messages)
    
    # phase_end_indices 업데이트 - 새로운 딕셔너리 생성
    existing_indices = state.get("phase_end_indices") or {}
    phase_end_indices = {**existing_indices, "planning": current_msg_count}  # 새 딕셔너리 생성

    # -------------------------------------------------
    # DynamoDB 로그 저장 (Stage 1 - 계획 결과 중심)
    # -------------------------------------------------
    try:
        conversation_id = st.session_state.get("learning_conversation_id")
        message_manager = st.session_state.get("learning_message_manager")

        if conversation_id and message_manager:
            # 1) 계획 정보(planning_data)를 JSON으로 저장
            try:
                planning_json = json.dumps(planning_data, ensure_ascii=False)
                message_manager.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content_type="learning_stage1_planning_info",
                    content=planning_json,
                    llm_model=BASE_LLM_MODEL,
                )
            except Exception as e:
                print(f"[learning_agent.finalize_planning] 계획 정보 저장 실패: {e}")

            # 2) 생성된 시스템 프롬프트가 있다면 함께 저장
            if generated_prompt:
                try:
                    message_manager.save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content_type="learning_stage1_generated_prompt",
                        content=generated_prompt,
                        llm_model=TOOL_LLM_MODEL,
                    )
                except Exception as e:
                    print(f"[learning_agent.finalize_planning] 생성 프롬프트 저장 실패: {e}")
    except Exception as e:
        # DynamoDB 연동 오류는 학습 플로우에 영향을 주지 않도록 로그만 남김
        print(f"[learning_agent.finalize_planning] DynamoDB 로깅 중 오류: {e}")

    return {
        "messages": [
            ToolMessage(
                content=ai_message,
                tool_call_id=tool_call_id or "unknown",
            ),
        ],
        "planning_info": planning_data,
        "generated_prompt": generated_prompt,
        "planning_completed": True,
        "phase": "execution",
        "waiting_for_input": True,
        "phase_end_indices": phase_end_indices,  # 새 딕셔너리 반환
    }


def planning_complete_confirmation(state: LearningAgentState):
    """
    정보 수집 완료 상태에서 사용자 선택 처리
    - 완료: finalize_planning으로 tool_calls 트리거
    - 계획 확인: 수집된 정보 출력
    - 계획 수정: planning 재시작 플래그 설정
    """
    messages = state["messages"]
    # 마지막 메시지만 확인 (옵션 제시 이후 실제 사용자 입력만 처리)
    last_msg = messages[-1] if messages else None
    if not isinstance(last_msg, HumanMessage):
        # 사용자 입력이 아직 없으므로 조용히 대기
        return {
            "messages": [],
            "phase": "planning",
            "awaiting_planning_confirmation": True,
            "waiting_for_input": True
        }

    last_human_msg = None
    
    # 마지막 사용자 메시지 찾기
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content.lower().strip()
            break

    # ⭐ 이제 항상 사용자 입력이 있음!
    if not last_human_msg:
        # 이 경우는 발생하지 않아야 함
        raise ValueError("planning_complete_confirmation called without user input")
    
    # 사용자 의도 파악
    complete_keywords = ["완료", "계획 완료", "다음 단계", "진행", "넘어가"]
    confirm_keywords = ["계획 확인", "정보 확인", "확인", "정리", "보여줘"]
    reset_keywords = ["계획 수정", "다시", "초기화", "처음부터", "재시작"]
    no_plan_keywords = ["없어요", "없습니다", "계획 없", "없음", "건너뛸", "건너뛰", "스킵", "패스"]
    
    # 1. 완료 의도 → PlanningCriteria tool_calls 생성
    if any(keyword in last_human_msg for keyword in complete_keywords):
        # 대화 내역에서 정보 추출 (간단한 예시)
        planning_info = extract_planning_info_from_conversation(state)
        
        # ✅ 1. 정보 추출 실패 시 명확한 오류 처리
        if any(v == "추출 실패" for v in planning_info.values()):
            return {
                "messages": [AIMessage(content="""
    ⚠️ 계획 정보를 자동으로 추출할 수 없습니다.

    먼저 **'계획 확인'** 을 입력해서 정보가 제대로 수집되었는지 확인해주세요.
    정보가 부족하다면 추가로 입력해주시고, 모두 확인되면 다시 **'완료'** 를 입력해주세요.
    """)],
                "phase": "planning",
                "awaiting_planning_confirmation": True  # 계속 대기
            }

        # LLM을 사용하여 tool_calls 생성하거나, 직접 생성
        llm = ChatOpenAI(model=BASE_LLM_MODEL, temperature=0)
        llm_with_tools = llm.bind_tools([PlanningCriteria])
        
        system_msg = SystemMessage(content=f"""
사용자가 계획 단계 완료를 확인했습니다.
다음 정보를 PlanningCriteria 도구로 제출하세요:

데이터 설명: {planning_info.get('dataset_description', '정보 없음')}
분석 목표: {planning_info.get('data_analysis_objective', '정보 없음')}
분석 계획: {planning_info.get('data_analysis_plan', '정보 없음')}
""")
        
        response = llm_with_tools.invoke([system_msg] + messages[-5:])
        
        # ✅ 4. tool_calls 생성 실패 시 강제 진행 (무한 루프 방지)
        if not response.tool_calls:
            print("⚠️ LLM이 tool_calls를 생성하지 못함. 수동으로 진행합니다.")
            # 수동으로 AIMessage with tool_calls 생성
            from langchain_core.messages import AIMessage
            import uuid
            
            tool_call_id = str(uuid.uuid4())
            forced_response = AIMessage(
                content="계획 정보를 확인했습니다. 실행 단계로 진행하겠습니다.",
                tool_calls=[{
                    "name": "PlanningCriteria",
                    "args": {
                        "dataset_description": planning_info.get('dataset_description', ''),
                        "data_analysis_objective": planning_info.get('data_analysis_objective', ''),
                        "data_analysis_plan": planning_info.get('data_analysis_plan', '')
                    },
                    "id": tool_call_id
                }]
            )
            response = forced_response

        return {
            "messages": [response],
            "phase": "planning",  # 여전히 planning (finalize로 가기 전)
            "awaiting_planning_confirmation": False  # 플래그 OFF
        }
    
    # 1-1. 계획 없음 의도 → 옵션 제시로 전환 (안전망)
    elif any(keyword in last_human_msg for keyword in no_plan_keywords):
        response = AIMessage(content="""
알겠습니다! 구체적인 계획 없이 진행하셔도 충분해요 😊

다음 중 하나를 선택해주세요:

- **'완료'** : 계획 단계를 완료하고 실행 단계로 넘어가기

- **'계획 확인'** : 수집된 정보를 정리해서 확인하기

- **'계획 수정'** : 계획을 처음부터 다시 시작하기

어떤 것을 진행하시겠어요?
""")
        return {
            "messages": [response],
            "phase": "planning",
            "awaiting_planning_confirmation": True  # 계속 대기
        }
    
    # 2. 계획 확인 의도 → 정보 정리 출력
    elif any(keyword in last_human_msg for keyword in confirm_keywords):
        planning_info = extract_planning_info_from_conversation(state)
        
        # 계획이 없는 경우 처리
        plan = planning_info.get('data_analysis_plan', '')
        if not plan or plan.strip() in ["", "정보 없음", "N/A", "없음", "추출 실패"]:
            plan_display = "(아직 구체적인 계획은 세우지 않았습니다)"
        else:
            plan_display = plan
        
        summary = f"""
📋 수집된 계획 정보

📊 데이터 설명:
{planning_info.get('dataset_description', '정보가 수집되지 않았습니다.')}

🎯 분석 목표:
{planning_info.get('data_analysis_objective', '정보가 수집되지 않았습니다.')}

📋 분석 계획 (선택):
{plan_display}

---

정보가 맞나요? 
- **'완료'** 를 입력하면 실행 단계로 넘어갑니다
- **'계획 수정'** 을 입력하면 처음부터 다시 시작합니다
"""
        response = AIMessage(content=summary)
        return {
            "messages": [response],
            "phase": "planning",
            "awaiting_planning_confirmation": False  # 플래그 OFF
        }
    
    # 3. 계획 수정 의도 → 재시작 확인
    elif any(keyword in last_human_msg for keyword in reset_keywords):
        response = AIMessage(content="""
계획을 처음부터 다시 시작하시겠어요? 

- **'예'** 또는 **'확인'**: 처음부터 다시 시작
- **'아니오'** 또는 **'취소'**: 현재 계획 유지

어떻게 하시겠어요?
""")
        return {
            "messages": [response],
            "phase": "planning",
            "awaiting_planning_confirmation": True
        }
    
    # 4. 의도 불명확 → 다시 옵션 제시
    else:
        response = AIMessage(content="""
선택을 명확히 이해하지 못했어요. 다시 선택해주세요:

- **'완료'** : 실행 단계로 넘어가기
- **'계획 확인'** : 수집된 정보 보기
- **'계획 수정'** : 처음부터 다시 시작하기
""")
        return {
            "messages": [response],
            "phase": "planning",
            "awaiting_planning_confirmation": True
        }


def extract_planning_info_from_conversation(state: LearningAgentState) -> dict:
    """대화 내역에서 planning 정보 추출"""
    messages = state["messages"]
    
    # ✅ planning 단계 메시지만 필터링 (더 정확한 추출)
    phase_end_indices = state.get("phase_end_indices", {})
    planning_messages = get_phase_messages(messages, "planning", phase_end_indices)
    if not planning_messages:
        planning_messages = messages
    
    llm = ChatOpenAI(model=TOOL_LLM_MODEL, temperature=0)
    
    extraction_prompt = """
다음 대화 내역에서 세 가지 정보를 추출하세요:
1. 데이터 설명 (dataset_description) - 필수
2. 분석 목표 (data_analysis_objective) - 필수
3. 분석 계획 (data_analysis_plan) - 선택 (명확히 언급되지 않았으면 빈 문자열로 설정)

JSON 형식으로만 반환하세요. 다른 텍스트는 포함하지 마세요:
{
  "dataset_description": "...",
  "data_analysis_objective": "...",
  "data_analysis_plan": "..." (없으면 "")
}
"""
    
    # 대화를 텍스트로 변환
    conversation_text = "\n\n".join([
        f"{msg.__class__.__name__}: {msg.content}" 
        for msg in planning_messages 
        if hasattr(msg, 'content') and msg.content
    ])
    
    response = llm.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=f"대화 내역:\n{conversation_text}")
    ])
    
    try:
        # JSON 파싱
        content = response.content.strip()
        # 코드 블록 제거
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"정보 추출 실패: {e}")
        return {
            "dataset_description": "추출 실패",
            "data_analysis_objective": "추출 실패",
            "data_analysis_plan": "추출 실패"
        }


def ensure_serializable(value):
    """
    msgpack 직렬화 가능한 형태로 변환
    
    Phase 1: 나중에 사용할 준비용 함수
    Enum 객체를 문자열로 변환하여 State 저장 시 직렬화 오류 방지
    
    Args:
        value: 변환할 값 (Enum, list, dict 등)
    
    Returns:
        msgpack 직렬화 가능한 형태 (기본 타입만)
    """
    from enum import Enum
    
    if isinstance(value, Enum):
        return value.name  # Enum → str (예: TutorAct.SE_ACKNOWLEDGE → "SE_ACKNOWLEDGE")
    
    if isinstance(value, list):
        return [ensure_serializable(v) for v in value]
    
    if isinstance(value, dict):
        return {k: ensure_serializable(v) for k, v in value.items()}
    
    # str, int, float, bool, None 등 기본 타입은 그대로 반환
    return value


def call_agency_tutor_safely(
    conversation_history: List[Dict[str, str]],
    student_message: str,
    intent_history: List,
    assessment_history: List,
    phase_context: str = "execution"
):
    """
    AgencyTutor를 안전하게 호출하는 헬퍼 함수
    
    LangGraph를 사용하지 않는 기존 AgencyTutor를 사용하여
    직렬화 문제를 완전히 회피
    
    Args:
        conversation_history: 대화 이력 (List[Dict[str, str]])
        student_message: 학생 메시지
        intent_history: 튜터 행동 이력 (문자열 리스트)
        assessment_history: 평가 이력 (문자열 리스트)
        phase_context: 단계 컨텍스트
    
    Returns:
        Tuple[str, List[str], str]: (응답, 직렬화된 intent, assessment)
    """
    from src.agency.Tutor import AgencyTutor
    from src.agency.prompts.TutorActTaxonomy import TutorAct
    from enum import Enum
    
    # Enum으로 역직렬화 (AgencyTutor는 Enum 리스트를 기대)
    def deserialize_intent(value):
        if isinstance(value, Enum):
            return value
        if isinstance(value, str):
            try:
                return TutorAct[value]
            except (KeyError, AttributeError):
                return value
        if isinstance(value, list):
            return [deserialize_intent(v) for v in value]
        return value
    
    intent_history_for_tutor = [deserialize_intent(item) for item in intent_history]
    
    # OpenAI 클라이언트 가져오기 (함수 스코프 내에서만 존재)
    client = get_openai_client()
    if not client:
        raise ValueError("OpenAI 클라이언트를 가져올 수 없습니다.")
    
    # AgencyTutor 인스턴스 생성 (LangGraph 없음 = 직렬화 문제 없음)
    tutor = AgencyTutor(
        client=client,
        model="gpt-4.1",
        intent_history=intent_history_for_tutor,
        assessment_history=assessment_history,
        open=True
    )
    
    # conversation_history에 student_message를 추가하여 전체 대화 구성
    full_conversation = conversation_history + [{"role": "user", "content": student_message}]
    
    # 응답 생성 (Phase 3: phase_context 전달)
    response, _, _, _, intent, assessment = tutor.get_response(
        conversation_history=full_conversation,
        messages_student=student_message,
        max_tokens=1500,
        phase_context=phase_context
    )
    
    # intent 직렬화 (Enum → str)
    serialized_intent = ensure_serializable(intent)
    
    return response, serialized_intent, assessment


def execution_node(state: LearningAgentState):
    """실행 단계 대화 처리"""
    llm_tools = get_llm_with_tools()
    
    # 처리되지 않은 tool_calls가 있으면 오류 방지
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_call_ids = {tc["id"] for tc in msg.tool_calls}
            existing_tool_call_ids = {
                m.tool_call_id 
                for m in messages 
                if isinstance(m, ToolMessage)
            }
            if tool_call_ids - existing_tool_call_ids:
                return {
                    "messages": [AIMessage(content="⚠️ 이전 단계가 완료되지 않았습니다. 먼저 계획 단계를 완료해주세요.")],
                    "waiting_for_input": True
                }
    
    # ⭐ finalize_planning에서 온 직후인지 확인
    # 마지막 메시지가 ToolMessage(계획 완료)이고, 그 이전에 AIMessage(프롬프트 생성)가 있으면 실행 단계 대화 시작
    if messages:
        last_msg = messages[-1]
        
        # finalize_planning에서 온 직후 패턴:
        # - 마지막 메시지: ToolMessage (프롬프트 생성 완료)
        if isinstance(last_msg, ToolMessage) and "프롬프트가 생성되었습니다" in last_msg.content:
            # 실행 단계 첫 대화 시작 - 기존 LLM 사용 (AgencyTutor는 사용자 입력 이후부터)
            messages_with_tracker = execution_tracker(state["messages"])
            response = llm_tools["execution"].invoke(messages_with_tracker)
            return {
                "messages": [response],
                "waiting_for_input": True,
                "execution_intent_history": state.get("execution_intent_history"),
                "execution_assessment_history": state.get("execution_assessment_history"),
            }
    
    # 마지막 사용자 메시지가 '분석 완료'인지 확인
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.strip()
            break

    # '분석 완료' 입력 확인 (실행 단계에서 입력된 경우만)
    # 단, finalize_planning 직후가 아닌 경우만
    if last_user_msg and last_user_msg == "분석 완료":
        messages = state.get("messages", [])
        current_msg_count = len(messages)
        
        # phase_end_indices 업데이트 - 새로운 딕셔너리 생성
        existing_indices = state.get("phase_end_indices") or {}
        phase_end_indices = {**existing_indices, "execution": current_msg_count}

        return {
            "messages": [AIMessage(content="실행 단계를 완료했습니다! 이제 Stage 3 성찰 단계로 넘어가주세요.")],
            "execution_completed": True,
            "phase": "reflection",
            "waiting_for_input": False,
            "phase_end_indices": phase_end_indices,
            "execution_intent_history": state.get("execution_intent_history"),
            "execution_assessment_history": state.get("execution_assessment_history"),
        }
    
    # 일반적인 실행 단계 대화 - AgencyTutorLG 사용
    try:
        # execution 단계 메시지만 추출
        execution_messages = get_phase_messages(
            messages,
            "execution",
            state.get("phase_end_indices")
        )
        
        # conversation_history로 변환
        conversation_history = convert_langchain_to_history(execution_messages)
        
        # 마지막 사용자 메시지 추출
        student_message = extract_last_user_message(execution_messages)
        
        # student_message가 없으면 대기 상태
        if not student_message:
            return {
                "messages": [],
                "waiting_for_input": True,
                "phase_end_indices": state.get("phase_end_indices"),
                "execution_intent_history": state.get("execution_intent_history"),
                "execution_assessment_history": state.get("execution_assessment_history"),
            }
        
        # State에서 히스토리 가져오기
        raw_intent_history = state.get("execution_intent_history") or []
        raw_assessment_history = state.get("execution_assessment_history") or []
        
        # AgencyTutorLG 안전하게 호출
        response, serialized_intent, assessment = call_agency_tutor_safely(
            conversation_history=conversation_history,
            student_message=student_message,
            intent_history=raw_intent_history,
            assessment_history=raw_assessment_history,
            phase_context="execution"
        )

        # Stage2 실행 단계 intent/assessment를 Agency 테이블에 바로 저장
        try:
            conversation_id = st.session_state.get("learning_conversation_id")
            agency_manager = st.session_state.get("learning_stage2_agency_manager")
            if conversation_id and agency_manager:
                agency_manager.save_agency_response(
                    conversation_id=conversation_id,
                    output=response,
                    intent=serialized_intent,
                    assessment=assessment,
                    turn=st.session_state.get("learning_turn"),
                )
        except Exception as e:
            print(f"[execution_node] Stage2 Agency 로그 저장 실패: {e}")

        # 히스토리 업데이트
        new_intent_history = list(raw_intent_history)
        new_intent_history.append(serialized_intent)
        new_assessment_history = list(raw_assessment_history)
        new_assessment_history.append(assessment)
        
        return {
            "messages": [AIMessage(content=response)],
            "waiting_for_input": True,
            "phase_end_indices": state.get("phase_end_indices"),
            "execution_intent_history": new_intent_history,
            "execution_assessment_history": new_assessment_history,
        }
        
    except Exception as e:
        # 에러 발생 시 기존 방식으로 폴백
        print(f"AgencyTutorLG 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        messages_with_tracker = execution_tracker(state["messages"])
        response = llm_tools["execution"].invoke(messages_with_tracker)
        return {
            "messages": [response],
            "waiting_for_input": True,
            "phase_end_indices": state.get("phase_end_indices"),
            "execution_intent_history": state.get("execution_intent_history"),
            "execution_assessment_history": state.get("execution_assessment_history"),
        }

def summarize_execution(state: LearningAgentState):
    """실행 단계 대화 요약 생성 (CSV Agent 대화만 요약)"""
    # CSV Agent와 사용자 간의 데이터 분석 대화만 요약 대상
    conversation_text = (state.get("csv_execution_history") or "").strip()
    
    # CSV 대화가 없으면 요약 불가 안내
    if not conversation_text:
        conversation_text = "실행 단계에서 CSV 데이터 분석 대화 기록이 저장되지 않았습니다."
        summary = "데이터 분석 대화 기록이 없어, 요약을 생성할 수 없습니다."
    else:
        # 요약 생성
        try:
            summary = conv_history_summary_tool.invoke({"conversation_history": conversation_text})
        except Exception as e:
            print(f"⚠️ 대화 요약 생성 오류: {e}")
            summary = "대화 요약 생성에 실패했습니다."
    
    # 메시지 및 state 업데이트
    summary_message = AIMessage(
        content="📊 실행 단계 대화 요약이 완료되었습니다!\n\n👈 왼쪽 사이드바에서 요약 내용을 확인해주세요!"
    )

    # -------------------------------------------------
    # DynamoDB에 실행 단계 대화 요약(summary) 내용 저장
    # -------------------------------------------------
    try:
        import streamlit as st  # 이미 상단에서 import 되어 있지만, 안전 차원에서 로컬 import

        conversation_id = st.session_state.get("learning_conversation_id")
        # Stage3에서 호출되므로 우선 Stage3용 매니저를 사용, 없으면 Stage1 순으로 폴백
        message_manager = (
            st.session_state.get("learning_stage3_message_manager")
            or st.session_state.get("learning_message_manager")
        )

        if conversation_id and message_manager and summary:
            current_turn = st.session_state.get("learning_turn")
            message_manager.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content_type="learning_stage3_execution_summary",
                content=summary,
                llm_model=TOOL_LLM_MODEL,
                turn=current_turn,
            )
    except Exception as e:
        print(f"[summarize_execution] 실행 단계 대화 요약 저장 실패: {e}")
    
    return {
        "messages": [summary_message],
        "execution_summary": summary,
        "summary_generated": True,
        "awaiting_reflection_start_choice": False
    }

def reflection_node(state: LearningAgentState):
    """성찰 단계 대화 처리"""
    llm_tools = get_llm_with_tools()

    # 요약 없으면 먼저 요약 유도(대화형)
    if not state.get("summary_generated", False):
        # 마지막 사용자 입력 추출
        last_human = None
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                last_human = m.content.strip().lower()
                break

        # 아직 선택 질문을 안 보냈다면 질문 제시
        if not state.get("awaiting_reflection_start_choice", False):
            return {
                "messages": [AIMessage(content=(
                    "🪞 Stage 3를 시작해요. 아래 중 하나를 선택해주세요:\n\n"
                    "1) 성찰 진행 (요약 없이 바로 시작)\n"
                    "2) 실행 단계 대화 요약 먼저 보기\n\n"
                    "원하는 선택을 '성찰' 또는 '요약'으로 답해주세요."
                ))],
                "phase": "reflection",
                "waiting_for_input": True,
                "awaiting_reflection_start_choice": True
            }

        # 선택 질문을 보낸 상태: "성찰 진행" 선택 시 요약 스킵하고 바로 진행
        if last_human and any(k in last_human for k in ["성찰", "성찰 진행", "바로", "진행", "시작", "성찰 시작"]):
            # summary_generated를 True로 설정하여 요약 분기를 스킵
            # 바로 성찰 질문 시작
            return {
                "messages": [AIMessage(content=(
                    "🔄⬜ 목표 달성 | ⬜⬜ 좋은 프롬프트 | ⬜⬜ 새로 배운 것\n\n"
                    "안녕하세요! 데이터 분석 과정을 돌아보며 함께 성찰해볼까요? 🪞\n"
                    "먼저, 이번 데이터 분석을 통해 분석 목표를 어느 정도 달성하셨나요?"
                ))],
                "phase": "reflection",
                "waiting_for_input": True,
                "summary_generated": True,  # 요약 없이 진행하는 것으로 표시
                "awaiting_reflection_start_choice": False
            }
        else:
            # 요약을 원하거나 아직 명확한 선택 안함 → route_initial이 처리할 것
            return {
                "messages": [AIMessage(content=(
                    "원하시면 '요약'이라고 답해 실행 대화를 정리해드릴게요. "
                    "바로 진행하려면 '성찰'이라고 답해주세요."
                ))],
                "phase": "reflection",
                "waiting_for_input": True,
                "awaiting_reflection_start_choice": True
            }

    # 요약 완료 후 또는 요약 스킵 후: 성찰 수집 진행
    # reflection 단계 메시지만 필터링해서 전달
    phase_end_indices = state.get("phase_end_indices")
    reflection_messages = get_phase_messages(state["messages"], "reflection", phase_end_indices)
    
    # reflection 단계 메시지가 없으면 전체 메시지 사용 (초기 상태)
    if not reflection_messages:
        reflection_messages = state["messages"]
    
    # 요약 메시지 이후의 메시지만 사용
    for i, msg in enumerate(reflection_messages):
        if isinstance(msg, AIMessage) and "📊 실행 단계 대화 요약이 완료되었습니다" in msg.content:
            reflection_messages = reflection_messages[i + 1:]
            break
    
    messages = reflection_tracker(reflection_messages)
    response = llm_tools["reflection"].invoke(messages)

    # 잘못된 체크리스트 항목 제거 및 완료 여부 확인
    if isinstance(response, AIMessage) and isinstance(response.content, str):
        # 선두의 '✅ 실행 완료 | ' 패턴 제거
        response.content = re.sub(
            r"^\s*✅\s*실행\s*완료\s*\|\s*", 
            "", 
            response.content.strip(), 
            flags=re.IGNORECASE
        )
        
        # ✅ 6개 수집 완료 패턴 시 자동 완료 처리 (3개 항목 × 2턴)
        if response.content.count("✅") >= 6:
            # 대화 내역에서 성찰 정보 추출
            info = extract_reflection_info_from_conversation(state)

            # ReflectionCriteria 도구 호출을 위한 AIMessage 생성
            llm = ChatOpenAI(model=BASE_LLM_MODEL, temperature=0)
            llm_with_tools = llm.bind_tools([ReflectionCriteria])
            sys = SystemMessage(content=f"""사용자가 성찰 단계를 완료했습니다. 다음 정보를 ReflectionCriteria로 제출하세요:
목표 달성 여부: {info.get('goal_achievement','정보 없음')}
좋은 프롬프트: {info.get('best_prompt','정보 없음')}
새로 배운 것: {info.get('new_learning','정보 없음')}""")
            tool_call_msg = llm_with_tools.invoke([sys] + state.get("messages", [])[-5:])

            # -------------------------------------------------
            # DynamoDB에 최종 성찰 안내/정리 메시지(response) 저장
            # -------------------------------------------------
            try:
                import streamlit as st
                conversation_id = st.session_state.get("learning_conversation_id")
                message_manager = (
                    st.session_state.get("learning_stage3_message_manager")
                    or st.session_state.get("learning_message_manager")
                )
                if conversation_id and message_manager:
                    current_turn = st.session_state.get("learning_turn")
                    message_manager.save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content_type="learning_stage3_reflection_complete",
                        content=response.content,
                        llm_model=BASE_LLM_MODEL,
                        turn=current_turn,
                    )
            except Exception as e:
                print(f"[reflection_node] 최종 성찰 메시지 저장 실패: {e}")

            return {
                "messages": [response, tool_call_msg],
                "phase": "reflection",
                "waiting_for_input": False,
                "awaiting_reflection_start_choice": False,
            }

    return {
        "messages": [response],
        "phase": "reflection",
        "waiting_for_input": True,
        "awaiting_reflection_start_choice": False
    }

def extract_reflection_info_from_conversation(state: LearningAgentState) -> dict:
    # planning과 유사한 추출기(간단 LLM 기반)
    llm = ChatOpenAI(model=TOOL_LLM_MODEL, temperature=0)
    conversation_text = "\n\n".join(
        f"{m.__class__.__name__}: {m.content}" for m in state.get("messages", [])
        if hasattr(m, "content") and m.content
    )
    sys = SystemMessage(content="""
다음 대화 내역에서 세 가지 정보를 추출하세요.
각 항목은 1차 질문과 2차 후속 질문에 대한 답변이 모두 포함되어 있으므로, 이를 통합하여 하나의 답변으로 추출하세요:

1) goal_achievement: 목표 달성 여부 + 극복 방법 or 아쉬운 점을 통합
2) best_prompt: 좋은 프롬프트 + 그 프롬프트를 생각해낸 방법을 통합
3) new_learning: 새롭게 배운 것 + AI 활용 과정에서 배운 것을 통합

JSON으로만 반환:
{"goal_achievement":"...", "best_prompt":"...", "new_learning":"..."}
""")
    resp = llm.invoke([sys, HumanMessage(content=f"대화 내역:\n{conversation_text}")]).content.strip()
    import json
    if resp.startswith("```"):
        resp = resp.split("```")[1]
        if resp.startswith("json"): resp = resp[4:]
        resp = resp.strip()
    try:
        return json.loads(resp)
    except Exception:
        return {}

def finalize_reflection(state: LearningAgentState):
    """성찰 단계 완료 처리"""
    messages = state.get("messages", [])
    
    # 안전성 검사
    if not messages:
        return {
            "reflection_info": {},
            "reflection_completed": False,
            "phase": "reflection",
            "waiting_for_input": True
        }
    
    last_message = messages[-1]
    
    # 마지막 메시지가 AIMessage이고 tool_calls가 있는지 확인
    if not isinstance(last_message, AIMessage):
        return {
            "reflection_info": {},
            "reflection_completed": False,
            "phase": "reflection",
            "waiting_for_input": True
        }
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {
            "reflection_info": {},
            "reflection_completed": False,
            "phase": "reflection",
            "waiting_for_input": True
        }
    
    # Tool call에서 수집된 정보 추출
    try:
        tool_call = last_message.tool_calls[0]
        reflection_data = tool_call.get("args", {})

        messages = state.get("messages", [])
        current_msg_count = len(messages)
        
        # phase_end_indices 업데이트 - 새로운 딕셔너리 생성
        existing_indices = state.get("phase_end_indices") or {}
        phase_end_indices = {**existing_indices, "reflection": current_msg_count}

        # -------------------------------------------------
        # DynamoDB에 성찰 완료 안내 메시지 저장
        # -------------------------------------------------
        try:
            import streamlit as st
            conversation_id = st.session_state.get("learning_conversation_id")
            message_manager = (
                st.session_state.get("learning_stage3_message_manager")
                or st.session_state.get("learning_message_manager")
            )
            if conversation_id and message_manager:
                current_turn = st.session_state.get("learning_turn")
                completion_text = "학습 성찰이 완료되었습니다! 수고하셨습니다."
                message_manager.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content_type="learning_stage3_reflection_complete",
                    content=completion_text,
                    llm_model=BASE_LLM_MODEL,
                    turn=current_turn,
                )
        except Exception as e:
            print(f"⚠️ finalize_reflection DB 저장 실패: {e}")
        
        return {
            "messages": [
                ToolMessage(
                    content="학습 성찰이 완료되었습니다! 수고하셨습니다.",
                    tool_call_id=tool_call.get("id", "unknown"),
                )
            ],
            "reflection_info": reflection_data,
            "reflection_completed": True,
            "phase": "complete",
            "waiting_for_input": False,
            "phase_end_indices": phase_end_indices,

        }
        
    except (IndexError, KeyError, AttributeError) as e:
        print(f"⚠️ finalize_reflection 오류: {e}")
        return {
            "reflection_info": {},
            "reflection_completed": False,
            "phase": "reflection",
            "waiting_for_input": True
        }

def generate_summary(state: LearningAgentState):
    """전체 학습 과정 요약"""
    # State에서 직접 정보 가져오기
    planning_info = state.get("planning_info", {})
    reflection_info = state.get("reflection_info", {})
    
    # 계획이 없는 경우 처리
    plan = planning_info.get('data_analysis_plan', '')
    if not plan or plan.strip() in ["", "N/A", "정보 없음", "없음"]:
        plan_display = "(구체적인 계획 없이 진행)"
    else:
        plan_display = plan
    
    summary_prompt = f"""
**학습 과정 요약**:

【계획 단계】
- 📊 데이터 설명: {planning_info.get('dataset_description', 'N/A')}
- 🎯 데이터 분석 목표: {planning_info.get('data_analysis_objective', 'N/A')}
- 📋 데이터 분석 계획: {plan_display}

【성찰 단계】
- 🎯 목표 달성 여부: {reflection_info.get('goal_achievement', 'N/A')}
- 👍 좋은 프롬프트: {reflection_info.get('best_prompt', 'N/A')}
- 💡 새롭게 배운 것: {reflection_info.get('new_learning', 'N/A')}
    """

    # -------------------------------------------------
    # DynamoDB에 최종 학습 과정 요약(summary_prompt) 저장
    # -------------------------------------------------
    try:
        import streamlit as st
        conversation_id = st.session_state.get("learning_conversation_id")
        message_manager = (
            st.session_state.get("learning_stage3_message_manager")
            or st.session_state.get("learning_message_manager")
        )
        if conversation_id and message_manager:
            current_turn = st.session_state.get("learning_turn")
            message_manager.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content_type="learning_stage3_final_summary",
                content=summary_prompt,
                llm_model=BASE_LLM_MODEL,
                turn=current_turn,
            )
    except Exception as e:
        print(f"[generate_summary] 최종 학습 요약 저장 실패: {e}")
    
    return {
        "messages": [AIMessage(content=summary_prompt)],
        "phase": "complete"
    }

# ============================================================
# Routing Functions
# ============================================================

def route_initial(state) -> Literal["planning", "execution", "reflection", "summarize_execution", "finalize_planning",  "planning_complete_confirmation"]:
    """초기 라우팅 - 현재 phase에 따라 적절한 노드로"""
    phase = state.get("phase", "planning")
    
    # 완료 상태면 종료
    if phase == "complete":
        return END
    
    # 메시지 확인 (특별한 메시지 우선 처리)
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        
        # __GENERATE_PROMPT__ 메시지는 phase와 관계없이 우선 처리
        if isinstance(last_msg, HumanMessage) and last_msg.content == "__GENERATE_PROMPT__":
            # tool_calls가 있는지 확인 (아직 처리되지 않은)
            tool_calls_pending = False
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_call_ids = {tc["id"] for tc in msg.tool_calls}
                    existing_tool_call_ids = {
                        m.tool_call_id 
                        for m in messages 
                        if isinstance(m, ToolMessage)
                    }
                    if tool_call_ids - existing_tool_call_ids:
                        tool_calls_pending = True
                        break
            
            if tool_calls_pending:
                return "finalize_planning"
        
        # __GENERATE_SUMMARY__ 메시지 처리
        if isinstance(last_msg, HumanMessage) and last_msg.content == "__GENERATE_SUMMARY__":
            if phase == "reflection":
                return "summarize_execution"

        # AIMessage의 tool_calls 처리 (planning 단계에서)
        elif isinstance(last_msg, AIMessage):
            if phase == "planning" and last_msg.tool_calls:
                # ToolMessage가 아직 없는지 확인
                tool_call_ids = {tc["id"] for tc in last_msg.tool_calls}
                existing_tool_call_ids = {
                    msg.tool_call_id 
                    for msg in messages 
                    if isinstance(msg, ToolMessage)
                }
                # 아직 처리되지 않은 tool_calls가 있으면 finalize_planning으로
                if tool_call_ids - existing_tool_call_ids:
                    return "finalize_planning"

    # Planning 완료 확인 대기 상태인지 확인
    if state.get("awaiting_planning_confirmation", False):
        # 마지막 메시지가 HumanMessage인지 확인 (사용자가 입력했는지)
        if messages and isinstance(messages[-1], HumanMessage):
            return "planning_complete_confirmation"

    # route_initial에 자연어 요약 의도 감지 추가
    if phase == "reflection" and messages:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            t = last.content.strip().lower()
            if state.get("awaiting_reflection_start_choice", False):
                # 선택 대기 상태에서 요약 선택 → summarize_execution으로
                if any(k in t for k in ["요약", "요약해줘", "요약 시작", "요약 생성", "정리", "보여줘"]):
                    return "summarize_execution"
                # 선택 대기 상태에서 성찰 진행 선택 → reflection으로 (요약 스킵)
                if any(k in t for k in ["성찰", "성찰 진행", "바로", "진행", "시작", "성찰 시작"]):
                    # summary_generated를 True로 설정하여 요약 단계 스킵
                    # (하지만 route_initial에서는 state를 직접 수정할 수 없으므로 reflection_node에서 처리)
                    return "reflection"
            # (선택 여부와 무관) 일반 요약 의도도 허용
            if any(k in t for k in ["요약", "요약해줘", "요약 시작", "요약 생성", "정리", "보여줘"]):
                return "summarize_execution"

    # planning 단계
    if phase == "planning":
        return "planning"
    
    # execution 단계
    if phase == "execution":
        return "execution"
    
    # reflection 단계 - 요약은 수동으로 생성 (버튼 클릭)
    if phase == "reflection":
        return "reflection"

    # 기본값: planning
    return "planning"

def route_from_planning(state) -> Literal["planning_complete_confirmation", "finalize_planning"]:
    """
    개선된 planning 라우팅
    1. tool_calls 감지 → finalize_planning
    2. awaiting_planning_confirmation 플래그 확인 → planning_complete_confirmation
    3. "✅✅✅" 패턴 감지 → planning_complete_confirmation
    4. 그 외 → END
    """
    messages = state["messages"]
    
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # 1. Tool calls 있으면 즉시 finalize로
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "finalize_planning"

    # 2. awaiting_planning_confirmation 플래그 확인 (⭐ 추가!)
    if state.get("awaiting_planning_confirmation", False):
        # 사용자 입력이 들어온 경우에만 confirmation으로
        if messages and isinstance(messages[-1], HumanMessage):
            return "planning_complete_confirmation"
        # 사용자 입력이 아직 없으면 대기
        return END

    # 3. "✅✅" 패턴 감지 (필수 2개)
    if isinstance(last_message, AIMessage):
        content = last_message.content
        # 필수 두 개의 체크마크가 있는지 확인
        if content.count("✅") >= 2:
            # 특히 "✅ 데이터 설명 | ✅ 분석 목표" 패턴
            if "데이터 설명" in content and "분석 목표" in content:
                return "planning_complete_confirmation"
    
    # 4. planning_info_collected 플래그 확인 (⭐ 추가!)
    if state.get("planning_info_collected", False):
        return "planning_complete_confirmation"
    
    return END

def route_from_planning_complete_confirmation(state) -> Literal["finalize_planning"]:
    """
    planning_complete_confirmation 노드에서의 라우팅
    - tool_calls가 있으면 finalize_planning으로
    - 그 외에는 END (사용자 입력 대기)
    """
    messages = state["messages"]
    
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # Tool calls가 있으면 finalize_planning으로
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "finalize_planning"
    
    # 그 외에는 사용자 입력 대기
    return END


def route_from_execution(state):
    """실행 단계에서의 라우팅 - 항상 END (다음 invoke에서 reflection으로)"""
    # execution_completed가 True여도 여기서는 END
    # 다음 invoke 시 route_initial이 reflection으로 라우팅
    return END

def route_from_reflection(state) -> Literal["finalize_reflection"]:
    """성찰 단계에서의 라우팅"""
    messages = state.get("messages", [])
    
    if not messages:
        return END
    
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "finalize_reflection"
    return END

# ============================================================
# Graph Creation
# ============================================================

@st.cache_resource
def create_learning_agent_graph():
    """Learning Agent 그래프 생성"""
    
    workflow = StateGraph(LearningAgentState)
    
    # 노드 추가
    workflow.add_node("planning", planning_node)
    workflow.add_node("planning_complete_confirmation", planning_complete_confirmation)
    workflow.add_node("finalize_planning", finalize_planning)
    workflow.add_node("execution", execution_node)
    workflow.add_node("summarize_execution", summarize_execution)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("finalize_reflection", finalize_reflection)
    workflow.add_node("generate_summary", generate_summary)
    
    # 엣지 연결
    # START에서 현재 phase에 따라 라우팅
    workflow.add_conditional_edges(
        START,
        route_initial,
        {
            "planning": "planning",
            "execution": "execution",
            "reflection": "reflection",
            "summarize_execution": "summarize_execution",
            "finalize_planning": "finalize_planning",
            "planning_complete_confirmation": "planning_complete_confirmation",  # ⭐ 추가!
            END: END
        }
    )

    # Planning에서는 tool_calls만 확인
    workflow.add_conditional_edges(
        "planning",
        route_from_planning,
        {
            "finalize_planning": "finalize_planning",
            "planning_complete_confirmation": "planning_complete_confirmation",  # 추가!
            END: END
        }
    )
    
    # Planning complete confirmation에서도 tool_calls 확인
    workflow.add_conditional_edges(
        "planning_complete_confirmation",
        route_from_planning_complete_confirmation,
        {
            "finalize_planning": "finalize_planning",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "reflection",
        route_from_reflection,
        {
            "finalize_reflection": "finalize_reflection",
            END: END
        }
    )

    # finalize_planning에서 execution으로 이동
    workflow.add_edge("finalize_planning", "execution")

    # execution은 항상 END (다음 invoke에서 route_initial이 reflection으로 라우팅)
    workflow.add_edge("execution", END)

    # summarize_execution 후 END로 이동 (사용자 입력 대기)
    workflow.add_edge("summarize_execution", "reflection")
    workflow.add_edge("finalize_reflection", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    # Checkpointer
    conn = sqlite3.connect("learning_agent.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

# ============================================================
# Runner Class
# ============================================================

class LearningAgentRunner:
    """Learning Agent 실행 관리"""
    
    def __init__(self):
        if "learning_graph" not in st.session_state:
            st.session_state.learning_graph = create_learning_agent_graph()
        
        self.graph = st.session_state.learning_graph
        
        # 스레드 목록 초기화 (없으면 DynamoDB에서 한번 동기화 시도)
        if "learning_thread_list" not in st.session_state:
            st.session_state.learning_thread_list = []
            try:
                self.sync_thread_list_from_db()
            except Exception as e:
                print(f"[LearningAgentRunner] 초기 thread 동기화 실패: {e}")
        
        # 현재 스레드 초기화
        thread_list = st.session_state.get("learning_thread_list", [])
        if "learning_thread_id" not in st.session_state:
            # 기존 세션이 있으면 가장 최근 세션을 기본 선택, 없으면 새 세션 생성
            if thread_list:
                st.session_state.learning_thread_id = thread_list[0]["id"]
            else:
                self.create_new_thread()
        
        self.config = {
            "configurable": {"thread_id": st.session_state.learning_thread_id}
        }

    def sync_thread_list_from_db(self):
        """
        DynamoDB의 learning_conversations 테이블에서
        현재 사용자 기준 학습 세션 목록을 불러와 session_state에 저장.
        """
        try:
            from src.graph.learning_logging import load_learning_threads_for_user

            threads = load_learning_threads_for_user()
            if threads:
                st.session_state.learning_thread_list = threads
        except Exception as e:
            print(f"[LearningAgentRunner] sync_thread_list_from_db 오류: {e}")

    def create_new_thread(self, title: Optional[str] = None):
        """새로운 학습 세션 시작 (LangGraph thread + DynamoDB conversation 동시 생성)"""
        import datetime

        new_thread_id = str(uuid.uuid4())

        # 1) LangGraph thread 설정
        st.session_state.learning_thread_id = new_thread_id
        self.config = {
            "configurable": {"thread_id": new_thread_id}
        }

        # 2) DynamoDB conversation 생성/연결
        try:
            from src.graph.learning_logging import init_learning_managers

            username = st.session_state.get("username", "anonymous")
            _, conversation_manager, message_manager = init_learning_managers()

            if conversation_manager and message_manager:
                conv = conversation_manager.get_or_create_conversation(
                    session_uuid=new_thread_id,
                    user_id=username,
                )
                if conv:
                    st.session_state["learning_conversation_id"] = conv["session_uuid"]
                    st.session_state["learning_message_manager"] = message_manager
                    st.session_state["learning_conversation_created_at"] = conv.get(
                        "created_at"
                    )
        except Exception as e:
            print(f"[LearningAgentRunner] create_new_thread Dynamo 연동 실패: {e}")

        # 3) sidebar에서 사용할 스레드 정보 저장 / 갱신
        thread_info = {
            "id": new_thread_id,
            "title": title
            or f"학습 세션 {len(st.session_state.get('learning_thread_list', [])) + 1}",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "planning",
        }

        st.session_state.setdefault("learning_thread_list", []).append(thread_info)

        return new_thread_id

    def switch_thread(self, thread_id: str):
        """다른 학습 세션으로 전환 (LangGraph thread + DynamoDB conversation 연결)"""
        st.session_state.learning_thread_id = thread_id
        self.config = {
            "configurable": {"thread_id": thread_id}
        }

        # �� 기존 conversation 매핑 초기화 (새 thread용으로 다시 맵핑하기 위함)
        st.session_state.pop("learning_conversation_id", None)
        st.session_state.pop("learning_message_manager", None)
        st.session_state.pop("learning_conversation_created_at", None)

        # 해당 thread_id용 DynamoDB conversation 보장
        try:
            from src.graph.learning_logging import ensure_learning_conversation
            ensure_learning_conversation()
        except Exception as e:
            print(f"[LearningAgentRunner] switch_thread ensure_learning_conversation 실패: {e}")
    
    def update_thread_info(self, thread_id: str, **kwargs):
        """스레드 정보 업데이트"""
        for thread in st.session_state.learning_thread_list:
            if thread["id"] == thread_id:
                thread.update(kwargs)
                break

        # phase 변경이 있으면 DynamoDB learning_conversations.phase 도 동기화
        phase = kwargs.get("phase")
        if phase:
            try:
                from src.graph.learning_logging import init_learning_managers

                _, conversation_manager, _ = init_learning_managers()
                conversation_id = st.session_state.get("learning_conversation_id")
                if conversation_id and conversation_manager:
                    conversation_manager.conversations_table.update_item(
                        Key={"session_uuid": conversation_id},
                        UpdateExpression="SET phase = :phase",
                        ExpressionAttributeValues={":phase": phase},
                    )
            except Exception as e:
                print(f"[LearningAgentRunner] phase 동기화 실패: {e}")
    
    def get_thread_info(self, thread_id: str):
        """스레드 정보 조회"""
        for thread in st.session_state.learning_thread_list:
            if thread["id"] == thread_id:
                return thread
        return None
    
    def send_message(self, user_input: str):
        """사용자 메시지 전송 및 응답 받기"""
        # invoke 이전 상태에서 메시지 개수 기록 (AI 응답 로그용)
        prev_snapshot = self.graph.get_state(self.config)
        prev_count = 0
        if prev_snapshot and prev_snapshot.values:
            prev_messages = prev_snapshot.values.get("messages", [])
            if isinstance(prev_messages, list):
                prev_count = len(prev_messages)

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.config
        )
        
        # 스레드 정보 업데이트
        if result:
            phase = result.get("phase", "planning")
            self.update_thread_info(st.session_state.learning_thread_id, phase=phase)

        # DynamoDB에 AI 응답 로그 저장
        self._log_ai_messages_to_dynamo(prev_count)

        return result
    
    def get_state(self):
        """현재 State 조회"""
        snapshot = self.graph.get_state(self.config)
        return snapshot.values if snapshot else None
    
    def initialize(self):
        """초기 실행 (planning 시작)"""
        result = self.graph.invoke(
            {"messages": []},
            config=self.config
        )
        return result
    
    def reset(self):
        """워크플로우 초기화 (새 스레드 생성)"""
        self.create_new_thread()
    
    def generate_summary(self):
        """실행 단계 대화 요약 생성 (수동 실행)"""
        # 특별한 메시지를 보내서 요약 생성 트리거
        # 라우팅 함수에서 이를 감지하여 summarize_execution 노드로 라우팅
        result = self.graph.invoke(
            {"messages": [HumanMessage(content="__GENERATE_SUMMARY__")]},
            config=self.config
        )
        return result
    
    def generate_prompt(self):
        """계획 단계 프롬프트 생성 (수동 실행)"""
        # 특별한 메시지를 보내서 프롬프트 생성 트리거
        # 라우팅 함수에서 이를 감지하여 finalize_planning 노드로 라우팅
        result = self.graph.invoke(
            {"messages": [HumanMessage(content="__GENERATE_PROMPT__")]},
            config=self.config
        )
        return result
    
    def reset_planning(self):
        """계획 단계를 처음부터 다시 시작 (전체 초기화)"""
        # 새 스레드 생성 (기존 메시지/상태 완전 초기화)
        self.create_new_thread()
        # 초기화 메시지 전송하여 planning 단계 시작
        self.initialize()
        return self.get_state()
    
    def set_csv_execution_history(self, history_text: str):
        """CSV Agent TEXT 대화를 state에 저장"""
        self.graph.update_state(
            self.config,
            {"csv_execution_history": history_text},
            as_node="execution"
        )
    
    def setup_test_execution_stage(self):
        """Stage 2 테스트를 위한 (Stage 1 완료 상태) 더미 state 설정"""
        # 더미 메시지 및 state 생성 - Stage 1이 끝난 직후라고 가정
        dummy_messages = [
            SystemMessage(content="시스템 시작"),
            AIMessage(content="안녕하세요! 학습을 시작해볼까요?"),
            HumanMessage(content="네, 시작할게요"),
        ]
        
        dummy_state = {
            "messages": dummy_messages,
            "phase": "execution",
            "planning_completed": True,
            "execution_completed": False,
            "reflection_completed": False,
            "waiting_for_input": True,
            "summary_generated": False,
            "awaiting_planning_confirmation": False,
            "planning_info": {
                "dataset_description": "테스트 데이터셋 - Iris 품종별 꽃잎 특징 데이터",
                "data_analysis_objective": "품종별 꽃잎 길이와 너비의 차이를 시각화하고 분석하기",
                "data_analysis_plan": "산점도와 박스플롯을 활용하여 품종별 특징 비교"
            },
            # Stage 2에서 사용할 더미 시스템 프롬프트
            "generated_prompt": (
                "이것은 Stage 2 실행 단계를 바로 테스트하기 위한 더미 시스템 프롬프트입니다.\n"
                "실제 학습에서는 Stage 1 계획 단계에서 생성된 프롬프트가 여기에 들어갑니다."
            ),
            "reflection_info": None,
            # planning 단계 메시지는 dummy_messages 전체라고 가정 (0~2 인덱스)
            "phase_end_indices": {
                "planning": len(dummy_messages) - 1
            },
            "execution_intent_history": [],
            "execution_assessment_history": []
        }
        
        # State 업데이트
        self.graph.update_state(self.config, dummy_state, as_node="__start__")
        
        return self.get_state()
    
    def setup_test_reflection_stage(self):
        """Stage 3 테스트를 위한 더미 state 설정"""
        # 더미 메시지 및 state 생성
        dummy_messages = [
            SystemMessage(content="시스템 시작"),
            AIMessage(content="안녕하세요! 학습을 시작해볼까요?"),
            HumanMessage(content="네, 시작할게요"),
        ]
        
        dummy_state = {
            "messages": dummy_messages,
            "phase": "reflection",
            "planning_completed": True,
            "execution_completed": True,
            "reflection_completed": False,
            "waiting_for_input": True,
            "summary_generated": True,
            "awaiting_planning_confirmation": False,
            "planning_info": {
                "dataset_description": "테스트 데이터셋 - Iris 품종별 꽃잎 특징 데이터",
                "data_analysis_objective": "품종별 꽃잎 길이와 너비의 차이를 시각화하고 분석하기",
                "data_analysis_plan": "산점도와 박스플롯을 활용하여 품종별 특징 비교"
            },
            "reflection_info": None,
            "phase_end_indices": {
                "planning": 3,
                "execution": 3
            },
            "execution_intent_history": [],
            "execution_assessment_history": []
        }
        
        # State 업데이트
        self.graph.update_state(self.config, dummy_state, as_node="__start__")
        
        return self.get_state()

    # -------------------------------------------------
    # 내부 유틸: AI 메시지 DynamoDB 로깅
    # -------------------------------------------------
    def _log_ai_messages_to_dynamo(self, previous_count: int):
        """graph.invoke 이후 추가된 AIMessage들을 DynamoDB에 저장."""
        try:
            conversation_id = st.session_state.get("learning_conversation_id")
            if not conversation_id:
                return

            snapshot = self.graph.get_state(self.config)
            if not snapshot or not snapshot.values:
                return

            messages = snapshot.values.get("messages", [])
            if not isinstance(messages, list) or previous_count >= len(messages):
                return

            # 새로 추가된 메시지들만 확인
            new_messages = messages[previous_count:]

            # 현재 phase에 따라 사용할 메시지 매니저 및 content_type 결정
            phase = snapshot.values.get("phase", "planning")
            if phase == "planning":
                message_manager = st.session_state.get("learning_message_manager")  # Stage1
                content_type = "learning_stage1_assistant_text"
            elif phase == "execution":
                # Stage2는 별도 dataanalysis/agency 테이블로 로깅하므로 여기서는 추가 로깅하지 않음
                return
            elif phase == "reflection":
                message_manager = st.session_state.get("learning_stage3_message_manager")  # Stage3
                content_type = "learning_stage3_assistant_text"
            else:
                return

            if not message_manager:
                return

            current_turn = st.session_state.get("learning_turn")

            for msg in new_messages:
                # HumanMessage는 Stage별 페이지에서 따로 로깅하므로 여기선 AI만 처리
                if isinstance(msg, AIMessage):
                    content = getattr(msg, "content", "") or ""
                    if not content.strip():
                        continue

                    try:
                        message_manager.save_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content_type=content_type,
                            content=content,
                            llm_model=BASE_LLM_MODEL,
                            turn=current_turn,
                        )
                    except Exception as e:
                        print(f"[LearningAgentRunner] AI 메시지 로그 저장 실패: {e}")
        except Exception as e:
            print(f"[LearningAgentRunner] DynamoDB 로깅 중 오류: {e}")
