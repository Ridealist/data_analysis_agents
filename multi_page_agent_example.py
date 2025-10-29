"""
멀티 페이지 LangGraph 에이전트 구현 예시

Streamlit 여러 페이지에 걸쳐 실행되는 LangGraph 에이전트
AgentState가 페이지 간 공유되며 영구 저장됨
"""

import streamlit as st
from typing import TypedDict, Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
import sqlite3


# ============================================================
# 1. State 정의 (페이지 정보 포함)
# ============================================================

class MultiPageAgentState(TypedDict):
    """멀티 페이지 에이전트 상태"""
    
    # 페이지 추적
    current_page: Literal["page1", "page2", "page3"]
    completed_steps: List[str]
    
    # 비즈니스 데이터
    user_input: str
    conversation_history: List[Dict[str, str]]
    
    # 각 페이지의 중간 결과
    page1_result: Optional[Dict[str, Any]]
    page2_result: Optional[Dict[str, Any]]
    page3_result: Optional[Dict[str, Any]]
    
    # 최종 출력
    final_output: Optional[str]
    
    # 메타데이터
    config: Dict[str, Any]


# ============================================================
# 2. 각 페이지의 노드 함수들
# ============================================================

def page1_step1_node(state: MultiPageAgentState) -> Dict:
    """Page 1 - 단계 1: 초기 분석"""
    print("🔵 Page 1 - Step 1 실행 중...")
    
    user_input = state["user_input"]
    
    # 실제 로직 (예: 데이터 분석)
    analysis_result = {
        "analyzed_text": user_input,
        "word_count": len(user_input.split()),
        "sentiment": "positive"  # 예시
    }
    
    return {
        "page1_result": analysis_result,
        "completed_steps": state.get("completed_steps", []) + ["page1_step1"]
    }


def page1_step2_node(state: MultiPageAgentState) -> Dict:
    """Page 1 - 단계 2: 추가 처리"""
    print("🔵 Page 1 - Step 2 실행 중...")
    
    page1_result = state["page1_result"]
    
    # 추가 처리
    enhanced_result = {
        **page1_result,
        "enhanced": True,
        "keywords": ["data", "analysis"]  # 예시
    }
    
    return {
        "page1_result": enhanced_result,
        "completed_steps": state.get("completed_steps", []) + ["page1_step2"]
    }


def page2_step1_node(state: MultiPageAgentState) -> Dict:
    """Page 2 - 단계 1: 변환"""
    print("🟢 Page 2 - Step 1 실행 중...")
    
    page1_data = state["page1_result"]
    
    # Page 1의 결과를 활용
    transformed = {
        "source": page1_data,
        "transformed_format": "processed",
        "timestamp": "2024-10-10"
    }
    
    return {
        "page2_result": transformed,
        "completed_steps": state.get("completed_steps", []) + ["page2_step1"]
    }


def page2_step2_node(state: MultiPageAgentState) -> Dict:
    """Page 2 - 단계 2: 검증"""
    print("🟢 Page 2 - Step 2 실행 중...")
    
    page2_result = state["page2_result"]
    
    validated = {
        **page2_result,
        "validated": True,
        "confidence": 0.95
    }
    
    return {
        "page2_result": validated,
        "completed_steps": state.get("completed_steps", []) + ["page2_step2"]
    }


def page3_final_node(state: MultiPageAgentState) -> Dict:
    """Page 3 - 최종 출력 생성"""
    print("🟣 Page 3 - Final Step 실행 중...")
    
    page1_data = state["page1_result"]
    page2_data = state["page2_result"]
    
    # 모든 결과 통합
    final_output = f"""
    분석 완료!
    
    📊 Page 1 분석:
    - 단어 수: {page1_data['word_count']}
    - 감정: {page1_data['sentiment']}
    - 키워드: {', '.join(page1_data['keywords'])}
    
    🔄 Page 2 처리:
    - 형식: {page2_data['transformed_format']}
    - 검증: {page2_data['validated']}
    - 신뢰도: {page2_data['confidence']}
    
    ✅ 전체 처리 완료
    """
    
    return {
        "final_output": final_output,
        "completed_steps": state.get("completed_steps", []) + ["page3_final"]
    }


# ============================================================
# 3. 라우팅 함수들
# ============================================================

def route_after_page1(state: MultiPageAgentState) -> str:
    """Page 1 완료 후 라우팅"""
    current_page = state.get("current_page", "page1")
    
    if current_page == "page2":
        return "page2_step1"
    elif current_page == "page3":
        return "page3_final"
    else:
        # Page 1 내에서 계속
        return END


def route_after_page2(state: MultiPageAgentState) -> str:
    """Page 2 완료 후 라우팅"""
    current_page = state.get("current_page", "page2")
    
    if current_page == "page3":
        return "page3_final"
    else:
        return END


# ============================================================
# 4. Graph 생성
# ============================================================

def create_multi_page_graph():
    """멀티 페이지 에이전트 그래프 생성"""
    
    workflow = StateGraph(MultiPageAgentState)
    
    # ===== Page 1 노드 =====
    workflow.add_node("page1_step1", page1_step1_node)
    workflow.add_node("page1_step2", page1_step2_node)
    
    # ===== Page 2 노드 =====
    workflow.add_node("page2_step1", page2_step1_node)
    workflow.add_node("page2_step2", page2_step2_node)
    
    # ===== Page 3 노드 =====
    workflow.add_node("page3_final", page3_final_node)
    
    # ===== 엣지 연결 =====
    # Page 1 흐름
    workflow.set_entry_point("page1_step1")
    workflow.add_edge("page1_step1", "page1_step2")
    
    # Page 1 → Page 2/3 조건부 라우팅
    workflow.add_conditional_edges(
        "page1_step2",
        route_after_page1,
        {
            "page2_step1": "page2_step1",
            "page3_final": "page3_final",
            END: END
        }
    )
    
    # Page 2 흐름
    workflow.add_edge("page2_step1", "page2_step2")
    
    # Page 2 → Page 3 조건부 라우팅
    workflow.add_conditional_edges(
        "page2_step2",
        route_after_page2,
        {
            "page3_final": "page3_final",
            END: END
        }
    )
    
    # Page 3 종료
    workflow.add_edge("page3_final", END)
    
    # Checkpointer와 함께 컴파일 (State 영구 저장)
    # checkpointer = SqliteSaver.from_conn_string(":memory:")  # 실제로는 "agent.db" 사용

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


# ============================================================
# 5. Streamlit 페이지별 실행 래퍼
# ============================================================

class MultiPageAgentRunner:
    """Streamlit 페이지에서 사용할 에이전트 실행 클래스"""
    
    def __init__(self):
        # 그래프 초기화 (한 번만)
        if "graph" not in st.session_state:
            st.session_state.graph = create_multi_page_graph()
        
        self.graph = st.session_state.graph
        
        # Thread ID 초기화
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        
        self.config = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }
    
    def initialize_state(self, user_input: str):
        """최초 State 초기화 (Page 1 진입 시)"""
        initial_state: MultiPageAgentState = {
            "current_page": "page1",
            "completed_steps": [],
            "user_input": user_input,
            "conversation_history": [],
            "page1_result": None,
            "page2_result": None,
            "page3_result": None,
            "final_output": None,
            "config": {}
        }
        
        # Page 1 실행
        result = self.graph.invoke(initial_state, config=self.config)
        return result
    
    def run_page(self, page_name: Literal["page1", "page2", "page3"]):
        """특정 페이지 실행"""
        
        # 현재 저장된 State 가져오기
        current_state = self.graph.get_state(self.config)
        
        if current_state is None:
            st.error("⚠️ 초기화되지 않은 상태입니다. Page 1부터 시작하세요.")
            return None
        
        # State 업데이트 (페이지 변경)
        updated_state = {
            **current_state.values,
            "current_page": page_name
        }
        
        # 그래프 재실행
        result = self.graph.invoke(updated_state, config=self.config)
        return result
    
    def get_current_state(self):
        """현재 State 조회"""
        state_snapshot = self.graph.get_state(self.config)
        if state_snapshot:
            return state_snapshot.values
        return None
    
    def reset(self):
        """State 초기화"""
        st.session_state.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }


# ============================================================
# 6. Streamlit 페이지 예시
# ============================================================

def streamlit_page1():
    """Streamlit Page 1"""
    st.title("📝 Page 1: 데이터 입력")
    
    agent = MultiPageAgentRunner()
    
    user_input = st.text_area("분석할 텍스트를 입력하세요:")
    
    if st.button("분석 시작"):
        with st.spinner("분석 중..."):
            result = agent.initialize_state(user_input)
            
        st.success("✅ Page 1 분석 완료!")
        st.json(result["page1_result"])
        
        st.info("👉 Page 2로 이동해서 계속 진행하세요.")
    
    # 현재 상태 확인
    if st.checkbox("현재 State 보기"):
        current_state = agent.get_current_state()
        if current_state:
            st.json(current_state)


def streamlit_page2():
    """Streamlit Page 2"""
    st.title("🔄 Page 2: 데이터 변환")
    
    agent = MultiPageAgentRunner()
    current_state = agent.get_current_state()
    
    if current_state is None:
        st.warning("⚠️ Page 1을 먼저 완료하세요.")
        return
    
    # Page 1 결과 표시
    st.subheader("Page 1 결과:")
    st.json(current_state["page1_result"])
    
    if st.button("Page 2 처리 시작"):
        with st.spinner("처리 중..."):
            result = agent.run_page("page2")
        
        st.success("✅ Page 2 처리 완료!")
        st.json(result["page2_result"])
        
        st.info("👉 Page 3로 이동해서 최종 결과를 확인하세요.")


def streamlit_page3():
    """Streamlit Page 3"""
    st.title("📊 Page 3: 최종 결과")
    
    agent = MultiPageAgentRunner()
    current_state = agent.get_current_state()
    
    if current_state is None or "page2_step2" not in current_state.get("completed_steps", []):
        st.warning("⚠️ Page 2를 먼저 완료하세요.")
        return
    
    if st.button("최종 결과 생성"):
        with st.spinner("결과 생성 중..."):
            result = agent.run_page("page3")
        
        st.success("✅ 모든 처리 완료!")
        st.markdown(result["final_output"])
        
        # 초기화 옵션
        if st.button("🔄 처음부터 다시 시작"):
            agent.reset()
            st.rerun()


# ============================================================
# 7. 사용 예시 (단일 파일 테스트)
# ============================================================

if __name__ == "__main__":
    """터미널에서 직접 테스트"""
    
    print("=" * 60)
    print("멀티 페이지 에이전트 테스트")
    print("=" * 60)
    
    # 그래프 생성
    graph = create_multi_page_graph()
    thread_id = "test_user_123"
    config = {"configurable": {"thread_id": thread_id}}
    
    # === Page 1 실행 ===
    print("\n🔵 Page 1 실행...")
    state1: MultiPageAgentState = {
        "current_page": "page1",
        "completed_steps": [],
        "user_input": "이것은 테스트 데이터입니다. 분석을 진행해주세요.",
        "conversation_history": [],
        "page1_result": None,
        "page2_result": None,
        "page3_result": None,
        "final_output": None,
        "config": {}
    }
    
    result1 = graph.invoke(state1, config=config)
    print(f"Page 1 결과: {result1['page1_result']}")
    print(f"완료된 단계: {result1['completed_steps']}")
    
    # === Page 2 실행 ===
    print("\n🟢 Page 2 실행...")
    state2 = {
        **result1,
        "current_page": "page2"
    }
    
    result2 = graph.invoke(state2, config=config)
    print(f"Page 2 결과: {result2['page2_result']}")
    print(f"완료된 단계: {result2['completed_steps']}")
    
    # === Page 3 실행 ===
    print("\n🟣 Page 3 실행...")
    state3 = {
        **result2,
        "current_page": "page3"
    }
    
    result3 = graph.invoke(state3, config=config)
    print(f"\n최종 출력:\n{result3['final_output']}")
    print(f"완료된 단계: {result3['completed_steps']}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

