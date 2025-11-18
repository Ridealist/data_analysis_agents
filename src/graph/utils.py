from typing import Optional, Dict

def get_phase_messages(messages: list, phase: str, phase_end_indices: Optional[Dict[str, int]]) -> list:
    """
    특정 단계의 메시지만 필터링하여 반환
    
    Args:
        messages: 전체 메시지 리스트
        phase: 단계 이름 ("planning", "execution", "reflection")
        phase_end_indices: 각 단계의 끝 인덱스 딕셔너리
    
    Returns:
        해당 단계의 메시지 리스트
    """
    if not phase_end_indices:
        # 인덱스가 없으면 전체 메시지 반환 (초기 상태)
        return messages
    
    # 이전 단계의 끝 인덱스 찾기
    phase_order = ["planning", "execution", "reflection"]
    phase_idx = phase_order.index(phase) if phase in phase_order else -1
    
    if phase_idx == 0:
        # Planning: 처음부터 planning_end_idx까지
        end_idx = phase_end_indices.get("planning")
        if end_idx is not None:
            return messages[:end_idx + 1]  # 끝 인덱스 포함
    elif phase_idx == 1:
        # Execution: planning_end + 1부터 execution_end_idx까지
        planning_end = phase_end_indices.get("planning")
        execution_end = phase_end_indices.get("execution")
        if planning_end is not None:
            start_idx = planning_end + 1
            if execution_end is not None:
                return messages[start_idx:execution_end + 1]
            else:
                return messages[start_idx:]  # 진행 중이면 현재까지
    elif phase_idx == 2:
        # Reflection: execution_end + 1부터 reflection_end_idx까지
        execution_end = phase_end_indices.get("execution")
        reflection_end = phase_end_indices.get("reflection")
        if execution_end is not None:
            start_idx = execution_end + 1
            if reflection_end is not None:
                return messages[start_idx:reflection_end + 1]
            else:
                return messages[start_idx:]  # 진행 중이면 현재까지
    
    # 기본값: 전체 메시지
    return messages