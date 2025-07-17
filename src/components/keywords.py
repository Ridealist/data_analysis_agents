import streamlit as st
import os
from typing import List, Tuple

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt


selected_prompt = "prompts/keyword-extract.yaml"

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# 체인 생성
def create_chain(prompt_filepath):
    """
    데이터 분석 학습을 위한 LangChain 체인을 생성합니다.
    
    이 함수는 학생들이 실제로 성공할 수 있는 질문들을 생성하기 위해
    GPT 모델과 프롬프트를 연결하는 파이프라인을 구성합니다.
    
    Args:
        prompt_filepath (str): 데이터 분석 교육용 프롬프트 파일 경로
    
    Returns:
        chain: 실행 가능한 LangChain 체인 객체
    """

    # 데이터 분석 교육에 특화된 프롬프트를 로드
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # GPT 모델 설정 (학생용이므로 비용 효율적인 mini 모델 사용)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)  # 약간의 창의성을 위해 temperature 증가

    # 문자열 출력 파서 (깔끔한 질문 형태로 반환)
    output_parser = StrOutputParser()

    # 체인 생성: 프롬프트 → LLM → 파서 순서로 실행
    chain = prompt | llm | output_parser
    return chain



def parsing_messages(conv_history: List[ChatMessage]) -> str:
    """
    대화 기록을 분석하여 맥락을 파악하고 적절한 다음 질문을 생성하기 위해
    대화 내용을 문자열로 변환합니다.
    
    이 함수는 학생과 AI 튜터 간의 대화 흐름을 이해하여
    현재 학습 단계에 맞는 질문을 추천할 수 있도록 도움을 줍니다.
    
    Args:
        conv_history (List[ChatMessage]): 대화 기록 리스트
    
    Returns:
        str: 파싱된 대화 내용 문자열
    """

    conv_context = []

    for chat in conv_history:
        if chat.role == "user":
            conv_context.append(f"사용자: {chat.content}")
        elif chat.role == "assistant":
            conv_context.append(f"AI챗봇: {chat.content}")
        else:
            continue
    
    return "\n".join(conv_context)



def create_keyword(conv_history: List[ChatMessage], analysis_stage='data_exploration') -> List[str]:
    """
    학생의 현재 학습 단계에 맞는 데이터 분석 질문 2개를 생성합니다.
    
    이 함수는 학생이 실제로 LLM에게 물어봤을 때 성공적인 결과를 얻을 수 있는
    구체적이고 실행 가능한 질문들을 추천합니다.
    
    Args:
        conv_history (List[ChatMessage]): 학생과의 대화 기록
        analysis_stage (str): 현재 분석 단계 ('data_exploration', 'pattern_finding', 'deep_analysis')
    
    Returns:
        List[str]: 추천 질문 2개가 담긴 리스트
    """

    # 데이터 분석 체인 생성
    chain = create_chain(selected_prompt)

    # 대화 기록이 없는 경우: 분석 단계별 초기 질문 제공
    if not conv_history:
        if analysis_stage == 'data_exploration':
            # 첫 단계: 데이터 파악하기 (100% 성공 보장)
            message_context = """대화 기록이 아직 없습니다. 학생들이 데이터 분석을 시작할 수 있도록 다음과 같은 형태의 실행 가능한 질문 2개를 제시하세요:
            
            1. '첨부한 [파일명] 데이터의 구조를 파악하고, 처음 5줄을 보여주세요. 각 컬럼이 무엇을 의미하는지도 설명해주세요.'
            2. '이 데이터로 가장 기본적인 시각화(막대그래프나 선그래프) 하나를 만들어주세요. 코드와 결과를 모두 보여주세요.'
            
            학생들이 LLM에게 바로 복사해서 붙여넣을 수 있는 형태로 만들어주세요."""
            
        elif analysis_stage == 'pattern_finding':
            # 중간 단계: 패턴 찾기
            message_context = """학생들이 기본적인 데이터 파악을 마쳤다고 가정하고, 패턴을 찾는 단계의 질문 2개를 제시하세요:
            
            1. 시간대별/지역별/카테고리별 비교 분석
            2. 특이한 값이나 흥미로운 패턴 발견하기
            
            실제로 LLM이 실행할 수 있는 구체적인 질문 형태로 만들어주세요."""
            
        else:  # deep_analysis
            # 고급 단계: 깊은 분석
            message_context = """학생들이 심화 분석을 할 수 있도록 다음 형태의 질문 2개를 제시하세요:
            
            1. 여러 데이터셋을 연결한 종합 분석
            2. 가설 설정 및 검증을 위한 분석
            
            학생들이 성공적으로 실행할 수 있는 수준의 질문으로 만들어주세요."""
    else:
        # 대화 기록이 있는 경우: 맥락을 바탕으로 다음 단계 질문 생성
        message_context = parsing_messages(conv_history)

    # LLM을 통해 추천 질문 생성
    response = chain.invoke({"context": message_context})

    # 응답에서 두 개의 질문 추출
    # 예상 형태: "추천 질문 1: ... // 추천 질문 2: ..."
    try:
        q1 = response.split("//")[0].split(":")[-1].strip()
        q2 = response.split("//")[-1].split(":")[-1].strip()
    except IndexError:
        # 파싱 실패 시 기본 질문 제공
        q1 = "이 데이터에서 가장 흥미로운 부분은 어디인가요? 구체적으로 분석해주세요."
        q2 = "방금 결과를 바탕으로 더 자세히 살펴보고 싶은 부분이 있다면 어떻게 분석하면 좋을까요?"

    return [q1, q2]


def create_keyword_practical(conv_history: List[ChatMessage]) -> List[str]:
    """
    실전 데이터 분석 프로젝트를 위한 실용적인 질문들을 생성합니다.
    
    이 함수는 학생들이 실제 프로젝트에서 마주칠 수 있는 상황들을 고려하여
    문제 해결 중심의 질문들을 제공합니다.
    
    Args:
        conv_history (List[ChatMessage]): 대화 기록
    
    Returns:
        List[str]: 실용적인 추천 질문 2개
    """
    chain = create_chain(selected_prompt)

    if not conv_history:
        message_context = """학생들이 실전 데이터 분석에서 자주 마주치는 상황을 고려한 실용적인 질문 2개를 제시하세요:
        
        예시 상황들:
        - 데이터가 예상과 다를 때 어떻게 대처할지
        - 분석 결과를 다른 사람에게 설명하는 방법
        - 오류가 발생했을 때의 해결 방법
        - 추가 데이터가 필요한 상황 판단
        
        학생들이 실제로 활용할 수 있는 구체적인 질문으로 만들어주세요."""
    else:
        message_context = parsing_messages(conv_history)

    response = chain.invoke({"context": message_context})

    try:
        q1 = response.split("//")[0].split(":")[-1].strip()
        q2 = response.split("//")[-1].split(":")[-1].strip()
    except IndexError:
        # 실전 중심의 기본 질문
        q1 = "현재 분석에서 놓치고 있는 중요한 관점이 있을까요? 어떻게 확인해볼 수 있을까요?"
        q2 = "이 분석 결과를 실제 상황에 적용한다면 어떤 추가 고려사항이 필요할까요?"

    return [q1, q2]


def create_keyword_llm_collaboration(conv_history: List[ChatMessage]) -> List[str]:
    """
    학생들이 LLM과 효과적으로 협업할 수 있도록 돕는 전략적 질문들을 생성합니다.
    
    이 함수는 학생들이 AI 도구를 더 효율적으로 활용할 수 있도록
    프롬프트 작성 기법과 문제 해결 전략을 포함한 질문들을 제공합니다.
    
    Args:
        conv_history (List[ChatMessage]): 대화 기록
    
    Returns:
        List[str]: LLM 협업 전략 질문 2개
    """
    chain = create_chain(selected_prompt)

    if not conv_history:
        message_context = """학생들이 LLM과 더 효과적으로 협업할 수 있도록 돕는 전략적 질문 2개를 제시하세요:
        
        포함할 요소들:
        - 더 정확한 결과를 얻기 위한 프롬프트 작성법
        - 오류 발생 시 LLM에게 도움 요청하는 방법
        - 복잡한 분석을 단계별로 나누어 진행하는 방법
        - LLM의 답변을 검증하고 개선하는 방법
        
        실제로 학습에 도움이 되는 구체적인 질문으로 만들어주세요."""
    else:
        message_context = parsing_messages(conv_history)

    response = chain.invoke({"context": message_context})

    try:
        q1 = response.split("//")[0].split(":")[-1].strip()
        q2 = response.split("//")[-1].split(":")[-1].strip()
    except IndexError:
        # LLM 협업 중심의 기본 질문
        q1 = "지금까지의 분석에서 LLM에게 더 구체적으로 질문했다면 어떤 부분이 개선될 수 있을까요?"
        q2 = "다음 분석 단계를 위해 LLM에게 어떻게 질문하면 가장 도움이 될까요?"

    return [q1, q2]