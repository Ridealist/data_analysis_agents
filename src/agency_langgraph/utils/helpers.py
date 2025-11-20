"""
유틸리티 헬퍼 함수들
"""


def postprocess_response(response: str) -> str:
    """
    LLM 응답의 LaTeX 수식 표기를 변환
    
    Args:
        response: LLM의 원본 응답 텍스트
    
    Returns:
        str: 변환된 응답 텍스트
        - \\( \\) → $ $ (인라인 수식)
        - \\[ \\] → $$ $$ (블록 수식)
    """
    return response.replace(
        "\\(", "$"
    ).replace(
        "\\)", "$"
    ).replace(
        "\\[", "$$"
    ).replace(
        "\\]", "$$"
    )

