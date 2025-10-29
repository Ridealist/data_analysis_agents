# LangGraph 기반 교육 에이전트 아키텍처

## 📐 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                    AgencyTutorLG                        │
│                  (메인 인터페이스)                        │
│  - get_response()                                       │
│  - get_response_stream()                                │
│  - update_client(), update_model()                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Workflow                         │
│            (create_agency_graph)                        │
│                                                         │
│  ┌─────────────┐                                        │
│  │   START     │                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────┐                              │
│  │   assess_node        │  ← IntentAssessor            │
│  │  (학생 상태 평가)      │                              │
│  └──────┬───────────────┘                              │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────┐                              │
│  │   select_node        │  ← TutorActSelector          │
│  │  (튜터 행동 선택)      │                              │
│  └──────┬───────────────┘                              │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────┐                              │
│  │ generate_prompt_node │  ← PromptGenerator           │
│  │  (프롬프트 생성)        │                              │
│  └──────┬───────────────┘                              │
│         │                                               │
│    ┌────┴────┐                                          │
│    │  조건부   │                                          │
│    │ 라우팅    │                                          │
│    └────┬────┘                                          │
│         │                                               │
│    ┌────┴──────────┐                                    │
│    ▼               ▼                                    │
│  ┌──────┐      ┌────────┐                              │
│  │respond│      │ stream │                              │
│  │_node  │      │ _node  │                              │
│  └───┬──┘      └───┬────┘                              │
│      │             │                                    │
│      └─────┬───────┘                                    │
│            ▼                                            │
│         ┌─────┐                                         │
│         │ END │                                         │
│         └─────┘                                         │
└─────────────────────────────────────────────────────────┘
```

## 🔄 데이터 흐름

### AgencyState 변환 과정

```python
# 1. 초기 입력
{
    "conversation_history": [...],
    "student_message": "질문입니다",
    "config": {
        "stream_mode": False,
        "open_mode": True,
        "max_tokens": 1500,
        "model": "gpt-4o",
        "client": <OpenAI>,
        ...
    }
}

# 2. assess_node 후
{
    ...기존 필드...,
    "assessment": '{"justification": "...", "selection": "abc"}',
    "tokens": {"assessor": 230}
}

# 3. select_node 후
{
    ...기존 필드...,
    "tutor_acts": [TutorAct.MP_GOAL, TutorAct.SE_CONFIDENCE]
}

# 4. generate_prompt_node 후
{
    ...기존 필드...,
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        ...
    ]
}

# 5. respond_node 후 (최종)
{
    ...기존 필드...,
    "output": "튜터의 응답 텍스트",
    "tokens": {
        "assessor": 230,
        "final": 1100,
        "total": 1330
    }
}
```

## 🧩 컴포넌트 상세

### 1. State Layer (상태 계층)

**파일:** `graph/state.py`

```python
class AgencyState(TypedDict):
    # 입력
    conversation_history: List[Dict[str, str]]
    student_message: str
    config: Dict[str, Any]
    
    # 중간 처리
    assessment: Optional[str]
    tutor_acts: Optional[List[TutorAct]]
    messages: Optional[List[Dict[str, str]]]
    
    # 출력
    output: Optional[Union[str, Any]]
    tokens: Optional[Dict[str, int]]
```

**역할:**
- 워크플로우 전체의 상태 정의
- 노드 간 데이터 전달 표준화
- 타입 안정성 보장

### 2. Agent Layer (에이전트 계층)

**파일:** `agents/assessor.py`, `agents/selector.py`, `agents/generator.py`

```python
# assessor.py
def assess_student(...) -> Tuple[str, int]:
    assessor = IntentAssessor(client, model, assessment_history)
    return assessor.assess(...)

# selector.py
def select_tutor_acts(...) -> List[TutorAct]:
    selector = LLMIntentSelector(client, intent_history)
    return selector.get_intent(...)

# generator.py
def generate_messages(...) -> List[Dict[str, str]]:
    generator = RigidPromptGenerator()
    return generator.get_prompt(...)
```

**역할:**
- 기존 `agency` 모듈 로직 래핑
- 순수 함수 형태로 변환
- 재사용성 향상

### 3. Node Layer (노드 계층)

**파일:** `graph/nodes.py`

```python
def assess_node(state: AgencyState) -> Dict[str, Any]:
    assessment, tokens = assess_student(
        conversation_history=state["conversation_history"],
        student_message=state["student_message"],
        ...
    )
    return {
        "assessment": assessment,
        "tokens": {"assessor": tokens}
    }
```

**역할:**
- State를 받아 일부 필드 업데이트
- Agent 함수 호출 및 결과 변환
- 워크플로우의 실행 단위

### 4. Workflow Layer (워크플로우 계층)

**파일:** `graph/workflow.py`

```python
def create_agency_graph():
    workflow = StateGraph(AgencyState)
    
    # 노드 추가
    workflow.add_node("assess", assess_node)
    workflow.add_node("select", select_node)
    ...
    
    # 엣지 연결
    workflow.set_entry_point("assess")
    workflow.add_edge("assess", "select")
    ...
    
    return workflow.compile()
```

**역할:**
- 노드 연결 및 실행 순서 정의
- 조건부 라우팅 설정
- 그래프 컴파일

### 5. Interface Layer (인터페이스 계층)

**파일:** `tutor.py`

```python
class AgencyTutorLG:
    def __init__(self, client, model, ...):
        self.graph = create_agency_graph()
    
    def get_response(self, ...):
        initial_state = {...}
        result = self.graph.invoke(initial_state)
        return ...
```

**역할:**
- 기존 API 호환성 유지
- State 구성 및 그래프 실행
- 결과 변환 및 히스토리 관리

## 🔀 조건부 라우팅

```python
def route_output(state: AgencyState) -> str:
    return "stream" if state["config"]["stream_mode"] else "respond"

workflow.add_conditional_edges(
    "generate",
    route_output,
    {
        "respond": "respond",
        "stream": "stream"
    }
)
```

**흐름:**
```
generate_prompt_node
       │
       ▼
  route_output()
       │
   ┌───┴───┐
   ▼       ▼
respond   stream
 _node     _node
   │       │
   └───┬───┘
       ▼
      END
```

## 🗂️ 디렉토리 구조 상세

```
src/agency_langgraph/
│
├── __init__.py                 # 모듈 초기화, AgencyTutorLG 노출
├── tutor.py                    # AgencyTutorLG 클래스 (메인 인터페이스)
│
├── graph/                      # 워크플로우 정의
│   ├── __init__.py
│   ├── state.py                # AgencyState 타입 정의
│   ├── nodes.py                # 노드 함수들 (assess, select, generate, respond, stream)
│   └── workflow.py             # 그래프 구성 및 컴파일
│
├── agents/                     # 기존 로직 래핑
│   ├── __init__.py
│   ├── assessor.py             # assess_student() - IntentAssessor 래핑
│   ├── selector.py             # select_tutor_acts() - TutorActSelector 래핑
│   └── generator.py            # generate_messages() - PromptGenerator 래핑
│
└── utils/                      # 유틸리티
    ├── __init__.py
    └── helpers.py              # postprocess_response() 등
```

## 🔌 의존성 관계

```
tutor.py
  └── workflow.py
      └── nodes.py
          ├── assessor.py → IntentAssessor (기존)
          ├── selector.py → TutorActSelector (기존)
          ├── generator.py → PromptGenerator (기존)
          └── helpers.py
```

**기존 모듈 재사용:**
- ✅ `IntentAssessor` - 학생 상태 평가 로직
- ✅ `TutorActSelector` - 튜터 행동 선택 로직
- ✅ `PromptGenerator` - 프롬프트 생성 로직
- ✅ `TutorActTaxonomy` - 튜터 행동 분류체계

**새로운 구조:**
- ✨ `AgencyState` - 중앙화된 상태 관리
- ✨ LangGraph 워크플로우 - 명시적 데이터 흐름
- ✨ 노드 기반 설계 - 확장 가능한 아키텍처

## 🎯 설계 원칙

### 1. 관심사 분리 (Separation of Concerns)

- **State**: 데이터 구조만 정의
- **Agent**: 비즈니스 로직 (기존 재사용)
- **Node**: State 변환 로직
- **Workflow**: 실행 흐름 제어
- **Interface**: 외부 API

### 2. 단방향 데이터 흐름

```
Input → State → Node1 → State → Node2 → State → ... → Output
```

각 노드는 State를 읽고 업데이트만 수행

### 3. 불변성 (Immutability)

```python
# 노드는 새로운 dict를 반환 (State 직접 수정 X)
def node(state):
    return {"field": new_value}  # ✅
    # state["field"] = new_value  # ❌
```

### 4. 합성 가능성 (Composability)

```python
# 새 노드 추가가 쉬움
workflow.add_node("new_feature", new_node)
workflow.add_edge("existing", "new_feature")
```

## 📊 실행 흐름 예시

### 예시 1: 일반 응답 (Non-streaming)

```python
# 입력
tutor.get_response(
    conversation_history=[],
    messages_student="데이터 분석 어떻게 시작하죠?",
    max_tokens=1500
)

# 내부 실행 흐름
START
  → assess_node
      → IntentAssessor.assess()
      → assessment: '{"selection": "ace"}'
  → select_node
      → TutorActSelector.get_intent()
      → tutor_acts: [TutorAct.MP_GOAL, TutorAct.MEL_EXPLORE]
  → generate_prompt_node
      → PromptGenerator.get_prompt()
      → messages: [{role: system, ...}, ...]
  → route_output() → "respond"
  → respond_node
      → OpenAI API call
      → output: "좋은 질문이에요! 먼저..."
END

# 반환
("좋은 질문이에요! ...", 1330, 230, 1100, [...], '{"..."}')
```

### 예시 2: 스트리밍 응답

```python
# 입력
tutor.get_response_stream(
    conversation_history=[...],
    messages_student="ChatGPT에 다 물어보면 안 돼요?",
    max_tokens=1500
)

# 내부 실행 흐름
START
  → assess_node → ...
  → select_node → ...
  → generate_prompt_node → ...
  → route_output() → "stream"
  → stream_node
      → OpenAI API call (stream=True)
      → output: <Stream object>
END

# 반환
(<Stream>, [...], '{"..."}')
```

## 🚀 확장 가능성

### 추가 가능한 노드 예시

```python
# 1. 응답 검증 노드
def validation_node(state):
    response = state["output"]
    if contains_inappropriate_content(response):
        response = sanitize(response)
    return {"output": response}

# 2. 로깅 노드
def logging_node(state):
    log_to_database(state)
    return {}  # State 변경 없음

# 3. 피드백 수집 노드
def feedback_node(state):
    feedback = collect_user_feedback()
    return {"feedback": feedback}

# 4. A/B 테스트 노드
def ab_test_node(state):
    variant = select_variant(state["student_message"])
    return {"variant": variant}
```

### 조건부 분기 확장

```python
def advanced_routing(state):
    codes = json.loads(state["assessment"])["selection"]
    
    if 'a' in codes:  # 과제 전체 위임
        return "strict_response"
    elif 'f' in codes or 'g' in codes:  # 학습 지원 요청
        return "guided_response"
    else:
        return "normal_response"

workflow.add_conditional_edges(
    "generate",
    advanced_routing,
    {
        "strict_response": "strict_node",
        "guided_response": "guided_node",
        "normal_response": "respond"
    }
)
```

## 🔍 디버깅 및 모니터링

### 중간 상태 확인

```python
from src.agency_langgraph.graph.workflow import create_agency_graph

graph = create_agency_graph()
result = graph.invoke(initial_state)

# 각 단계 결과
print("Assessment:", result.get("assessment"))
print("Tutor Acts:", result.get("tutor_acts"))
print("Messages:", result.get("messages"))
print("Output:", result.get("output"))
print("Tokens:", result.get("tokens"))
```

### 그래프 시각화

```python
# Mermaid 다이어그램
mermaid = graph.get_graph().draw_mermaid()
print(mermaid)

# 또는 PNG 이미지
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

## 📈 성능 특성

| 측면 | 기존 | LangGraph | 설명 |
|------|------|-----------|------|
| **LLM 호출** | 3회 | 3회 | 동일 (assess, select, respond) |
| **메모리 사용** | 낮음 | 약간 높음 | State 객체 오버헤드 |
| **실행 시간** | 기준 | +10ms | 그래프 실행 오버헤드 |
| **확장성** | 제한적 | 우수 | 노드 추가 용이 |
| **디버깅** | 어려움 | 쉬움 | 상태 추적 가능 |

**결론:** 약간의 오버헤드로 큰 이득 (유지보수성, 확장성)

## 🎓 학습 곡선

1. **LangGraph 기본 개념** (30분)
   - State, Node, Edge
   - Workflow 구성

2. **현재 구조 이해** (1시간)
   - AgencyState 구조
   - 각 노드 역할
   - 데이터 흐름

3. **기존 코드 마이그레이션** (30분)
   - 임포트 변경
   - 테스트

4. **커스터마이징** (상황에 따라)
   - 새 노드 추가
   - 조건부 라우팅 수정

**총 소요 시간:** 2-3시간

## 🔗 관련 문서

- [사용 가이드](./README.md)
- [마이그레이션 가이드](../../LANGGRAPH_MIGRATION.md)
- [LangGraph 공식 문서](https://python.langchain.com/docs/langgraph)

