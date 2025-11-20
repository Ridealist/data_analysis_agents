# LangGraph 기반 교육 에이전트 (AgencyTutorLG)

기존 `agency` 모듈을 LangGraph 프레임워크로 재구조화한 버전입니다.

## 📋 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [사용법](#사용법)
- [마이그레이션 가이드](#마이그레이션-가이드)
- [고급 기능](#고급-기능)

## 개요

### 주요 개선사항

✅ **중앙화된 상태 관리**: 모든 상태를 `AgencyState`로 통합 관리  
✅ **명확한 데이터 흐름**: 노드 기반의 명시적 파이프라인  
✅ **향상된 디버깅**: 각 단계별 상태 추적 가능  
✅ **확장 가능성**: 새로운 노드 추가 및 조건부 흐름 구현 용이  
✅ **메모리 관리**: 체크포인터를 통한 영구 대화 저장 지원  
✅ **완벽한 호환성**: 기존 `AgencyTutor` API와 100% 호환  

### 워크플로우

```
입력 (학생 메시지)
    ↓
[assess_node] - 학생 상태 평가 (IntentAssessor)
    ↓
[select_node] - 튜터 행동 선택 (TutorActSelector)
    ↓
[generate_prompt_node] - 프롬프트 생성 (PromptGenerator)
    ↓
[respond_node OR stream_node] - 응답 생성
    ↓
출력 (튜터 응답)
```

## 아키텍처

### 디렉토리 구조

```
src/agency_langgraph/
├── __init__.py
├── tutor.py                    # 메인 인터페이스 (AgencyTutorLG)
├── graph/
│   ├── __init__.py
│   ├── state.py                # AgencyState 정의
│   ├── nodes.py                # 워크플로우 노드 함수들
│   └── workflow.py             # 그래프 구성 및 컴파일
├── agents/
│   ├── __init__.py
│   ├── assessor.py             # 학생 평가 로직
│   ├── selector.py             # 튜터 행동 선택 로직
│   └── generator.py            # 프롬프트 생성 로직
└── utils/
    ├── __init__.py
    └── helpers.py              # 유틸리티 함수
```

### 상태 스키마 (AgencyState)

```python
class AgencyState(TypedDict):
    # 입력
    conversation_history: List[Dict[str, str]]
    student_message: str
    
    # 설정
    config: Dict[str, Any]  # stream_mode, open_mode, max_tokens, model, client 등
    
    # 파이프라인 (노드별 업데이트)
    assessment: Optional[str]
    tutor_acts: Optional[List[TutorAct]]
    messages: Optional[List[Dict[str, str]]]
    
    # 출력
    output: Optional[Union[str, Any]]
    
    # 메타데이터
    tokens: Optional[Dict[str, int]]
```

## 설치

### 1. 의존성 설치

```bash
pip install langgraph langchain-core
```

### 2. 기존 환경 확인

기존 `agency` 모듈이 정상 작동하는지 확인:

```bash
python -c "from src.agency.Tutor import AgencyTutor; print('✅ 기존 모듈 정상')"
```

## 사용법

### 기본 사용

```python
from src.agency_langgraph.tutor import AgencyTutorLG
from src.agency.utils.get_key import get_client_openAI

# 클라이언트 생성
client = get_client_openAI()

# 튜터 초기화
tutor = AgencyTutorLG(
    client=client,
    model="gpt-4o",
    open=True
)

# 응답 생성
response, total_tokens, prompt_tokens, completion_tokens, intent, assessment = tutor.get_response(
    conversation_history=[],
    messages_student="데이터 분석을 어떻게 시작해야 할지 모르겠어요.",
    max_tokens=1500
)

print(f"튜터: {response}")
print(f"선택된 행동: {intent}")
```

### 스트리밍 응답

```python
# 스트리밍 모드
stream, intent, assessment = tutor.get_response_stream(
    conversation_history=[],
    messages_student="ChatGPT에게 전부 물어보면 되는 거 아닌가요?",
    max_tokens=1500
)

# 스트리밍 출력
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 대화 히스토리 관리

```python
conversation_history = []

# Turn 1
response_1, _, _, _, _, _ = tutor.get_response(
    conversation_history=conversation_history,
    messages_student="데이터 분석 과제가 있어요."
)

# 히스토리 업데이트
conversation_history.append({"role": "user", "content": "데이터 분석 과제가 있어요."})
conversation_history.append({"role": "assistant", "content": response_1})

# Turn 2
response_2, _, _, _, _, _ = tutor.get_response(
    conversation_history=conversation_history,
    messages_student="어떻게 시작하죠?"
)
```

## 마이그레이션 가이드

### 기존 코드에서 교체

**Before (기존 AgencyTutor):**
```python
from src.agency.Tutor import AgencyTutor

tutor = AgencyTutor(client=client, model="gpt-4o", open=True)
```

**After (새로운 AgencyTutorLG):**
```python
from src.agency_langgraph.tutor import AgencyTutorLG

tutor = AgencyTutorLG(client=client, model="gpt-4o", open=True)
```

API가 완전히 동일하므로 **나머지 코드는 수정 불필요**합니다!

### 점진적 마이그레이션

```python
# 1단계: 별칭 사용
from src.agency_langgraph.tutor import AgencyTutorLG as AgencyTutor

# 2단계: A/B 테스트
USE_LANGGRAPH = True

if USE_LANGGRAPH:
    from src.agency_langgraph.tutor import AgencyTutorLG as AgencyTutor
else:
    from src.agency.Tutor import AgencyTutor

# 3단계: 완전 교체
from src.agency_langgraph.tutor import AgencyTutorLG
```

## 고급 기능

### 1. 그래프 시각화

```python
from src.agency_langgraph.graph.workflow import create_agency_graph

graph = create_agency_graph()

# Mermaid 다이어그램 생성
mermaid_graph = graph.get_graph().draw_mermaid()
print(mermaid_graph)

# PNG 이미지 생성 (Jupyter에서)
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 2. 영구 메모리 (Checkpointer)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agency_langgraph.graph.workflow import create_agency_graph_with_memory

# SQLite 기반 메모리 생성
memory = SqliteSaver.from_conn_string("agency_conversations.db")

# 메모리가 포함된 그래프 생성
graph_with_memory = create_agency_graph_with_memory(memory)

# 세션별 대화 관리
config = {"configurable": {"thread_id": "student_123"}}

result = graph_with_memory.invoke(
    {
        "conversation_history": [],
        "student_message": "안녕하세요!",
        "config": {...}
    },
    config=config
)

# 같은 thread_id로 재호출하면 이전 상태 복원
result_2 = graph_with_memory.invoke(
    {
        "conversation_history": [],
        "student_message": "이어서 질문이에요.",
        "config": {...}
    },
    config=config  # 동일한 thread_id
)
```

### 3. 커스텀 노드 추가

```python
# graph/nodes.py에 새 노드 추가
def validation_node(state: AgencyState) -> Dict[str, Any]:
    """응답 검증 노드"""
    response = state["output"]
    
    # 검증 로직
    if "부적절한단어" in response:
        response = response.replace("부적절한단어", "***")
    
    return {"output": response}

# graph/workflow.py에 노드 연결
workflow.add_node("validate", validation_node)
workflow.add_edge("respond", "validate")
workflow.add_edge("validate", END)
```

### 4. 조건부 라우팅 확장

```python
def advanced_routing(state: AgencyState) -> str:
    """복잡한 조건부 라우팅"""
    assessment_codes = json.loads(state["assessment"])["selection"]
    
    if 'a' in assessment_codes:  # 과제 전체 위임
        return "strict_guidance"
    elif 'i' in assessment_codes:  # 단계적 학습
        return "supportive_response"
    else:
        return "normal_response"

workflow.add_conditional_edges(
    "generate",
    advanced_routing,
    {
        "strict_guidance": "strict_node",
        "supportive_response": "supportive_node",
        "normal_response": "respond"
    }
)
```

## 테스트

```bash
# 전체 테스트 실행
python test_langgraph_agency.py

# 개별 테스트
python -c "from test_langgraph_agency import test_basic_response; test_basic_response()"
```

## 성능 비교

| 항목 | 기존 AgencyTutor | AgencyTutorLG |
|------|-----------------|---------------|
| LLM 호출 횟수 | 3회 | 3회 (동일) |
| 상태 관리 | 분산 (각 클래스) | 중앙화 (State) |
| 디버깅 | 어려움 | 단계별 추적 가능 |
| 확장성 | 제한적 | 노드 추가 용이 |
| 메모리 관리 | 수동 | 자동 (Checkpointer) |
| 시각화 | 불가 | 가능 (Mermaid) |

## 문제 해결

### 1. 임포트 에러

```bash
ModuleNotFoundError: No module named 'langgraph'
→ pip install langgraph langchain-core
```

### 2. 기존 모듈 의존성

```python
# 기존 모듈 재사용
from src.agency.prompts.IntentAssessor import IntentAssessor
from src.agency.prompts.TutorActSelector import LLMIntentSelector
from src.agency.prompts.PromptGenerator import RigidPromptGenerator
```

모두 정상 작동합니다!

## 기여

새로운 노드나 기능 추가 시:
1. `graph/nodes.py`에 노드 함수 추가
2. `graph/workflow.py`에 노드 연결
3. 테스트 작성

## 라이선스

기존 `agency` 모듈과 동일

