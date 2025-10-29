"""
Learning Agent 공통 모듈
Learning_Agents.ipynb의 LangGraph 구조를 Streamlit 멀티페이지로 구현
"""

import streamlit as st
import sqlite3
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

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
    data_analysis_plan: str       # 데이터 분석 계획


class ReflectionCriteria(BaseModel):
    """반성 단계에서 수집할 정보"""
    goal_achievement: str   # 목표 달성 여부
    best_prompt: str        # 좋은 프롬프트
    worst_prompt: str       # 나쁜 프롬프트
    new_learning: str       # 새롭게 배운 것

# ============================================================
# State Schema
# ============================================================

class LearningAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    phase: Literal["planning", "execution", "reflection", "complete"]  # 현재 단계
    
    # 수집된 정보 저장
    planning_info: Optional[Dict[str, str]]      # 계획 단계 정보
    reflection_info: Optional[Dict[str, str]]    # 반성 단계 정보
    
    # 완료 상태
    planning_completed: bool
    execution_completed: bool
    reflection_completed: bool
    waiting_for_input: bool  # 사용자 입력 대기 상태

# ============================================================
# System Prompts
# ============================================================

PROMPT_PLANNING = f"""{BASE_SYSTEM_PROMPT}

당신은 데이터 분석 계획 수립을 도와주는 어시스턴트입니다.
사용자로부터 다음 정보를 수집해야 합니다:

- 데이터 설명: 지금 분석하려는 데이터에 대한 설명 (어떤 데이터인지, 어떤 열(컬럼)들이 있는지 등)
- 분석 목표: 이번 데이터 분석에서 달성 하고 싶은 목표
- 분석 계획: 이번 데이터 분석 단계별 계획

정보가 명확하지 않으면 구체화를 위한 질문을 하세요.
한 번에 너무 많은 질문을 하지 말고, 사용자가 편하게 답할 수 있도록 하세요.
모든 정보가 수집되면 관련 도구를 호출하세요."""

PROMPT_EXECUTION = f"""{BASE_SYSTEM_PROMPT}

사용자가 학습 계획을 실행하는 단계입니다.
질문에 답하고, 도움을 제공하세요.
사용자가 '완료'라고 입력하면 반성 단계로 넘어갑니다."""

PROMPT_REFLECTION = f"""{BASE_SYSTEM_PROMPT}

당신은 데이터 분석 과정에 대한 반성을 도와주는 어시스턴트입니다.
사용자로부터 다음 정보를 수집해야 합니다:

- 목표 달성 여부: 이번 데이터 분석을 통해 분석 목표를 어느 정도 달성했는지
- 좋은 프롬프트: 데이터 분석 과정에서 가장 잘 AI에게 질문했다 생각하는 프롬프트
- 나쁜 프롬프트: 데이터 분석 과정에서 개선이 필요하고 느끼는 프롬프트
- 새롭게 배운 것: 이번 학습을 통해 얻은 새로운 지식이나 깨달음

정보가 명확하지 않으면 구체화를 위한 질문을 하세요.
한 번에 너무 많은 질문을 하지 말고, 사용자가 편하게 답할 수 있도록 한 번에 하나씩 질문 하세요.
각 항목에 대해 구체적이고 실질적인 내용을 얻을 수 있도록 도와주세요.
모든 정보가 수집되면 관련 도구를 호출하세요."""

# ============================================================
# LLM Setup
# ============================================================

@st.cache_resource
def get_llm():
    """LLM 인스턴스 생성 (캐싱)"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def get_llm_with_tools():
    """도구가 바인딩된 LLM 반환"""
    llm = get_llm()
    return {
        "planning": llm.bind_tools([PlanningCriteria]),
        "reflection": llm.bind_tools([ReflectionCriteria]),
        "execution": llm
    }

# ============================================================
# Node Functions
# ============================================================

def planning_tracker(messages):
    return [SystemMessage(content=PROMPT_PLANNING)] + messages

def execution_tracker(messages):
    return [SystemMessage(content=PROMPT_EXECUTION)] + messages

def reflection_tracker(messages):
    return [SystemMessage(content=PROMPT_REFLECTION)] + messages

def planning_node(state: LearningAgentState):
    """계획 단계 대화 처리"""
    llm_tools = get_llm_with_tools()
    messages = planning_tracker(state["messages"])
    response = llm_tools["planning"].invoke(messages)
    return {"messages": [response]}

def finalize_planning(state: LearningAgentState):
    """계획 단계 완료 처리"""
    # Tool call에서 수집된 정보 추출
    planning_data = state["messages"][-1].tool_calls[0]["args"]
    
    return {
        "messages": [
            ToolMessage(
                content="계획이 완성되었습니다! 이제 실행 단계로 넘어갑니다.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ],
        "planning_info": planning_data,  # State에 저장
        "planning_completed": True,
        "phase": "execution",
        "waiting_for_input": True
    }

def execution_node(state: LearningAgentState):
    """실행 단계 대화 처리"""
    llm_tools = get_llm_with_tools()
    
    # 마지막 메시지가 '완료'인지 확인
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.strip()
            break

    # '완료' 입력 확인
    if last_user_msg and last_user_msg == "완료":
        return {
            "messages": [AIMessage(content="실행 단계를 완료했습니다! 이제 학습을 반성해보겠습니다.")],
            "execution_completed": True,
            "phase": "reflection",
            "waiting_for_input": False
        }
    
    # 일반적인 실행 단계 대화
    messages = execution_tracker(state["messages"])
    response = llm_tools["execution"].invoke(messages)
    return {
        "messages": [response],
        "waiting_for_input": True
    }

def reflection_node(state: LearningAgentState):
    """반성 단계 대화 처리"""
    llm_tools = get_llm_with_tools()
    messages = reflection_tracker(state["messages"])
    response = llm_tools["reflection"].invoke(messages)
    return {
        "messages": [response],
        "waiting_for_input": True
    }

def finalize_reflection(state: LearningAgentState):
    """반성 단계 완료 처리"""
    # Tool call에서 수집된 정보 추출
    reflection_data = state["messages"][-1].tool_calls[0]["args"]
    
    return {
        "messages": [
            ToolMessage(
                content="학습 반성이 완료되었습니다! 수고하셨습니다.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ],
        "reflection_info": reflection_data,  # State에 저장
        "reflection_completed": True,
        "phase": "complete",
        "waiting_for_input": False
    }

def generate_summary(state: LearningAgentState):
    """전체 학습 과정 요약"""
    # State에서 직접 정보 가져오기
    planning_info = state.get("planning_info", {})
    reflection_info = state.get("reflection_info", {})
    
    summary_prompt = f"""
학습 과정 요약:

【계획 단계】
- 데이터 설명: {planning_info.get('dataset_description', 'N/A')}
- 데이터 분석 목표: {planning_info.get('data_analysis_objective', 'N/A')}
- 데이터 분석 계획: {planning_info.get('data_analysis_plan', 'N/A')}

【반성 단계】
- 목표 달성 여부: {reflection_info.get('goal_achievement', 'N/A')}
- 좋은 프롬프트: {reflection_info.get('best_prompt', 'N/A')}
- 나쁜 프롬프트: {reflection_info.get('worst_prompt', 'N/A')}
- 새롭게 배운 것: {reflection_info.get('new_learning', 'N/A')}

학습 과정이 모두 완료되었습니다!
"""
    
    return {
        "messages": [AIMessage(content=summary_prompt)],
        "phase": "complete"
    }

# ============================================================
# Routing Functions
# ============================================================

def route_initial(state) -> Literal["planning", "execution", "reflection"]:
    """초기 라우팅 - 현재 phase에 따라 적절한 노드로"""
    phase = state.get("phase", "planning")
    
    # 완료 상태면 종료
    if phase == "complete":
        return END
    
    # planning 단계
    if phase == "planning":
        return "planning"
    
    # execution 단계
    if phase == "execution":
        return "execution"
    
    # reflection 단계  
    if phase == "reflection":
        return "reflection"
    
    # 기본값: planning
    return "planning"

def route_from_planning(state) -> Literal["finalize_planning"]:
    """계획 단계에서의 라우팅"""
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "finalize_planning"
    return END

def route_from_execution(state):
    """실행 단계에서의 라우팅 - 항상 END (다음 invoke에서 reflection으로)"""
    # execution_completed가 True여도 여기서는 END
    # 다음 invoke 시 route_initial이 reflection으로 라우팅
    return END

def route_from_reflection(state) -> Literal["finalize_reflection"]:
    """반성 단계에서의 라우팅"""
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
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
    workflow.add_node("finalize_planning", finalize_planning)
    workflow.add_node("execution", execution_node)
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
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "planning",
        route_from_planning,
        {"finalize_planning": "finalize_planning", END: END}
    )
    workflow.add_edge("finalize_planning", "execution")
    
    # execution은 항상 END (다음 invoke에서 route_initial이 reflection으로 라우팅)
    workflow.add_edge("execution", END)
    
    workflow.add_conditional_edges(
        "reflection",
        route_from_reflection,
        {"finalize_reflection": "finalize_reflection", END: END}
    )
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
        
        # 스레드 목록 초기화
        if "learning_thread_list" not in st.session_state:
            st.session_state.learning_thread_list = []
        
        # 현재 스레드 초기화
        if "learning_thread_id" not in st.session_state:
            self.create_new_thread()
        
        self.config = {
            "configurable": {"thread_id": st.session_state.learning_thread_id}
        }
    
    def create_new_thread(self, title=None):
        """새로운 학습 세션 시작"""
        new_thread_id = str(uuid.uuid4())
        
        # 스레드 정보 저장
        import datetime
        thread_info = {
            "id": new_thread_id,
            "title": title or f"학습 세션 {len(st.session_state.learning_thread_list) + 1}",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "planning"
        }
        
        st.session_state.learning_thread_list.append(thread_info)
        st.session_state.learning_thread_id = new_thread_id
        
        self.config = {
            "configurable": {"thread_id": new_thread_id}
        }
        
        return new_thread_id
    
    def switch_thread(self, thread_id: str):
        """다른 학습 세션으로 전환"""
        st.session_state.learning_thread_id = thread_id
        self.config = {
            "configurable": {"thread_id": thread_id}
        }
    
    def update_thread_info(self, thread_id: str, **kwargs):
        """스레드 정보 업데이트"""
        for thread in st.session_state.learning_thread_list:
            if thread["id"] == thread_id:
                thread.update(kwargs)
                break
    
    def get_thread_info(self, thread_id: str):
        """스레드 정보 조회"""
        for thread in st.session_state.learning_thread_list:
            if thread["id"] == thread_id:
                return thread
        return None
    
    def send_message(self, user_input: str):
        """사용자 메시지 전송 및 응답 받기"""
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=self.config
        )
        
        # 스레드 정보 업데이트
        if result:
            phase = result.get("phase", "planning")
            self.update_thread_info(st.session_state.learning_thread_id, phase=phase)
        
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


# ============================================================
# UI Helper Functions
# ============================================================

def render_thread_manager(agent: LearningAgentRunner):
    """스레드 관리 UI (사이드바용)"""
    
    st.header("📚 학습 세션 관리")
    
    thread_list = st.session_state.learning_thread_list
    current_thread_id = st.session_state.learning_thread_id
    
    if thread_list:
        # 현재 세션 표시
        # current_thread = agent.get_thread_info(current_thread_id)
        # if current_thread:
        #     st.write(f"**현재 세션:** {current_thread['title']}")
        #     st.caption(f"생성: {current_thread['created_at']}")
        #     st.caption(f"단계: {current_thread['phase']}")
        
        # st.divider()
        
        # 세션 선택
        # st.write("**저장된 세션:**")
        
        for i, thread in enumerate(reversed(thread_list)):  # 최신순
            is_current = thread["id"] == current_thread_id
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 세션 정보
                phase_emoji = {
                    "planning": "📝",
                    "execution": "🚀", 
                    "reflection": "🪞",
                    "complete": "✅"
                }
                emoji = phase_emoji.get(thread["phase"], "❓")
                
                label = f"{emoji} {thread['title']}"
                if is_current:
                    label = f"**🔵 {label}** (현재)"
                
                st.write(label)
                st.caption(f"{thread['created_at']} | {thread['phase']}")
            
            with col2:
                if not is_current:
                    if st.button("열기", key=f"switch_{thread['id']}", use_container_width=True):
                        agent.switch_thread(thread["id"])
                        st.rerun()
        
        st.divider()
    
    # 새 세션 시작
    if st.button("➕ 새 학습 세션 시작", type="primary", use_container_width=True):
        agent.create_new_thread()
        st.rerun()

