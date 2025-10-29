"""
Phase 3: 그래프 구조 (단순화)

LangGraph 워크플로우 정의 및 컴파일
노드 연결, 조건부 라우팅, 실행 흐름 관리
"""

from langgraph.graph import StateGraph, END
from .state import AgencyState
from .nodes import (
    assess_node,
    select_node,
    generate_prompt_node,
    respond_node,
    stream_node
)


def create_agency_graph():
    """
    교육 에이전트 워크플로우 그래프 생성
    
    워크플로우 흐름:
    START → assess → select → generate → [respond OR stream] → END
    
    Returns:
        CompiledGraph: 실행 가능한 LangGraph 워크플로우
    """
    
    # 그래프 초기화
    workflow = StateGraph(AgencyState)
    
    # ========== 노드 추가 ==========
    workflow.add_node("assess", assess_node)
    workflow.add_node("select", select_node)
    workflow.add_node("generate", generate_prompt_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("stream", stream_node)
    
    # ========== 조건부 라우팅 함수 ==========
    def route_output(state: AgencyState) -> str:
        """
        스트리밍 여부에 따라 응답 노드 선택
        
        Args:
            state: 현재 상태
        
        Returns:
            str: 다음 노드 이름 ("respond" 또는 "stream")
        """
        return "stream" if state["config"]["stream_mode"] else "respond"
    
    # ========== 엣지 연결 ==========
    # 시작점 설정
    workflow.set_entry_point("assess")
    
    # 순차 실행 엣지
    workflow.add_edge("assess", "select")
    workflow.add_edge("select", "generate")
    
    # 조건부 엣지: 스트리밍 여부에 따라 분기
    workflow.add_conditional_edges(
        "generate",
        route_output,
        {
            "respond": "respond",
            "stream": "stream"
        }
    )
    
    # 종료 엣지
    workflow.add_edge("respond", END)
    workflow.add_edge("stream", END)
    
    # 그래프 컴파일
    return workflow.compile()


def create_agency_graph_with_memory(checkpointer):
    """
    영구 메모리를 포함한 교육 에이전트 워크플로우 생성
    
    대화 히스토리를 데이터베이스에 저장하고 세션별로 관리 가능
    
    Args:
        checkpointer: LangGraph 체크포인터 (예: SqliteSaver)
    
    Returns:
        CompiledGraph: 메모리 기능이 추가된 실행 가능한 워크플로우
    
    사용 예:
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        memory = SqliteSaver.from_conn_string("agency.db")
        graph = create_agency_graph_with_memory(memory)
        
        # 세션별 실행
        config = {"configurable": {"thread_id": "student_123"}}
        result = graph.invoke(state, config=config)
    """
    
    # 그래프 초기화
    workflow = StateGraph(AgencyState)
    
    # 노드 추가 (동일)
    workflow.add_node("assess", assess_node)
    workflow.add_node("select", select_node)
    workflow.add_node("generate", generate_prompt_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("stream", stream_node)
    
    # 라우팅 함수
    def route_output(state: AgencyState) -> str:
        return "stream" if state["config"]["stream_mode"] else "respond"
    
    # 엣지 연결 (동일)
    workflow.set_entry_point("assess")
    workflow.add_edge("assess", "select")
    workflow.add_edge("select", "generate")
    workflow.add_conditional_edges(
        "generate",
        route_output,
        {"respond": "respond", "stream": "stream"}
    )
    workflow.add_edge("respond", END)
    workflow.add_edge("stream", END)
    
    # 체크포인터와 함께 컴파일
    return workflow.compile(checkpointer=checkpointer)

