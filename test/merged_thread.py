import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAIError, InternalServerError
from typing import Dict, Any, Optional

# MessageType과 MessageRole 정의 (원본 파일에서 import 대신 직접 정의)
from enum import Enum

class MessageType(Enum):
    TEXT = "text"
    CODE = "code"
    DATAFRAME = "dataframe"
    FIGURE = "figure"
    ERROR = "error"

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def stream_ask_to_queue(query: str, agent, python_tool, queue: Queue, error_queue: Queue):
    """
    ask 함수를 스트리밍 가능한 형태로 변환하여 Queue에 전송
    
    Args:
        query: 사용자 질문
        agent: DataAnalysisAgent 인스턴스
        python_tool: Python 실행 도구
        queue: 결과를 담을 Queue
        error_queue: 에러를 담을 Queue
    """
    try:
        # 이전 결과 초기화
        if 'result_df' in python_tool.locals:
            del python_tool.locals['result_df']
        
        # 임시 저장 변수
        temp_last_result_df = None
        temp_last_figure = None
        
        response_items = []
        executed_code = ""
        
        try:
            # Agent 스트림 실행 및 결과 수집
            for step in agent.stream(query, "abc123"):
                response_items.append(step)
                
                # 코드 실행 감지
                if "actions" in step:
                    for action in step["actions"]:
                        if action.tool == "python_repl_tool":
                            executed_code = action.tool_input.get("code", "")
                            queue.put(("CODE", executed_code))
                            
                            # 시각화 코드 감지 및 실행
                            vis_keywords = ["plt.", "sns.", "matplotlib", "seaborn", "plt.show()"]
                            if any(keyword in executed_code for keyword in vis_keywords):
                                try:
                                    safe_code = executed_code.replace("plt.show()", "")
                                    python_tool.invoke({"query": safe_code})
                                    
                                    # figure가 생성되었는지 확인
                                    if hasattr(plt, 'gcf'):
                                        fig = plt.gcf()
                                        if fig and fig.get_axes():
                                            queue.put(("FIGURE", fig))
                                            temp_last_figure = fig
                                except Exception as e:
                                    error_msg = f"시각화 처리 중 오류: {str(e)}"
                                    queue.put(("ERROR", error_msg))
                
                # DataFrame 결과 확인
                if 'result_df' in python_tool.locals:
                    result_df = python_tool.locals['result_df']
                    if isinstance(result_df, (pd.DataFrame, pd.Series)):
                        queue.put(("DATAFRAME", result_df))
                        temp_last_result_df = result_df
                        del python_tool.locals['result_df']  # 중복 방지
                
                # 텍스트 출력 스트리밍
                if "output" in step:
                    queue.put(("TEXT", step["output"]))
                    
        except OpenAIError as e:
            # OpenAI 에러 처리
            if isinstance(e, InternalServerError):
                error_queue.put(f"서버 문제 발생: {str(e)}")
            else:
                error_queue.put(f"API 오류: {str(e)}")
            queue.put(None)
            return
            
        except Exception as e:
            error_queue.put(f"예상치 못한 오류: {str(e)}")
            queue.put(None)
            return
        
        # 최종 결과 처리
        final_text = ""
        for item in response_items:
            if "output" in item:
                final_text += item["output"]
        
        if final_text and not any(item[0] == "TEXT" for item in []):
            queue.put(("TEXT", final_text))
        
        # 스트림 종료 신호
        queue.put(None)
        
    except Exception as e:
        error_queue.put(str(e))
        queue.put(None)


def stream_thinking_to_queue(query: str, thinking_chain, queue: Queue, error_queue: Queue):
    """
    생각할 거리 체인을 스트리밍하여 Queue에 전송
    
    Args:
        query: 사용자 질문
        thinking_chain: 생각할 거리 생성 체인
        queue: 결과를 담을 Queue
        error_queue: 에러를 담을 Queue
    """
    try:
        for chunk in thinking_chain.stream({"input": query}):
            if hasattr(chunk, 'content'):
                queue.put(chunk.content)
            else:
                queue.put(str(chunk))
        queue.put(None)  # 스트림 종료 신호
    except Exception as e:
        error_queue.put(str(e))
        queue.put(None)


def parallel_ask_streaming_process(user_input: str, agent, python_tool, thinking_chain):
    """
    ask 함수와 thinking chain을 병렬로 스트리밍 처리
    
    Args:
        user_input: 사용자 입력
        agent: DataAnalysisAgent 인스턴스
        python_tool: Python 실행 도구
        thinking_chain: 생각할 거리 생성 체인
    
    Returns:
        tuple: (main_queue, sidebar_queue, error_queue)
    """
    # 큐 생성
    main_queue = Queue()
    sidebar_queue = Queue()
    error_queue = Queue()
    
    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=2) as executor:
        # ask 함수를 큐로 스트리밍
        executor.submit(
            stream_ask_to_queue, 
            user_input, 
            agent, 
            python_tool, 
            main_queue, 
            error_queue
        )
        
        # thinking chain을 큐로 스트리밍
        executor.submit(
            stream_thinking_to_queue,
            user_input,
            thinking_chain,
            sidebar_queue,
            error_queue
        )
    
    return main_queue, sidebar_queue, error_queue


def display_ask_streaming_response(queue: Queue, container, error_queue: Queue, 
                                  add_message_func=None):
    """
    ask 함수의 복잡한 출력(코드, DataFrame, Figure, 텍스트)을 스트리밍 표시
    
    Args:
        queue: 데이터를 가져올 Queue
        container: Streamlit 컨테이너
        error_queue: 에러 Queue
        add_message_func: 메시지 저장 함수 (선택사항)
    
    Returns:
        dict: 처리된 모든 결과
    """
    results = {
        "text": "",
        "code": None,
        "dataframe": None,
        "figure": None,
        "errors": []
    }
    
    text_placeholder = None
    
    while True:
        try:
            # 에러 체크
            if not error_queue.empty():
                error = error_queue.get_nowait()
                container.error(f"오류 발생: {error}")
                results["errors"].append(error)
                break
            
            # 데이터 가져오기
            item = queue.get(timeout=0.1)
            
            if item is None:  # 스트림 종료
                break
            
            # 튜플 형태의 데이터 처리
            if isinstance(item, tuple) and len(item) == 2:
                msg_type, content = item
                
                if msg_type == "TEXT":
                    results["text"] += content
                    if text_placeholder is None:
                        text_placeholder = container.empty()
                    text_placeholder.markdown(results["text"])
                    
                elif msg_type == "CODE":
                    with container.expander("🔍 실행된 코드 보기", expanded=False):
                        st.code(content, language="python")
                    results["code"] = content
                    if add_message_func:
                        add_message_func(MessageRole.ASSISTANT, [MessageType.CODE, content])
                
                elif msg_type == "DATAFRAME":
                    container.success("📊 데이터 분석 결과:")
                    container.dataframe(content, use_container_width=True)
                    results["dataframe"] = content
                    if add_message_func:
                        add_message_func(MessageRole.ASSISTANT, [MessageType.DATAFRAME, content])
                
                elif msg_type == "FIGURE":
                    container.pyplot(content)
                    results["figure"] = content
                    if add_message_func:
                        add_message_func(MessageRole.ASSISTANT, [MessageType.FIGURE, content])
                    plt.close(content)  # 메모리 정리
                
                elif msg_type == "ERROR":
                    results["errors"].append(content)
                    # 에러는 화면에 표시하지 않고 로그만 남김
            
            # 단순 텍스트 스트리밍 (thinking chain 용)
            elif isinstance(item, str):
                results["text"] += item
                if text_placeholder is None:
                    text_placeholder = container.empty()
                text_placeholder.markdown(results["text"])
                
        except Empty:
            continue
        except Exception as e:
            container.error(f"표시 중 오류: {str(e)}")
            results["errors"].append(str(e))
            break
    
    # 최종 텍스트 메시지 저장
    if results["text"] and add_message_func:
        add_message_func(MessageRole.ASSISTANT, [MessageType.TEXT, results["text"]])
    
    return results


# 사용 예제 함수
def example_usage():
    """통합된 병렬 스트리밍 사용 예제"""
    
    # 필요한 컴포넌트들이 준비되었다고 가정
    agent = st.session_state.get("agent_with_UploadData")
    python_tool = st.session_state.get("python_tool")
    
    # thinking chain 생성
    thinking_chain = ChatPromptTemplate.from_messages([
        ("system", "깊이 생각해볼 질문을 제시하세요."),
        ("human", "{input}")
    ]) | ChatOpenAI(model="gpt-4o-mini", streaming=True)
    
    # 사용자 입력 받기
    user_input = st.chat_input("질문을 입력하세요")
    
    if user_input and agent and python_tool:
        # 사용자 메시지 표시
        st.chat_message("user").write(user_input)
        
        # 병렬 스트리밍 시작
        main_queue, sidebar_queue, error_queue = parallel_ask_streaming_process(
            user_input, agent, python_tool, thinking_chain
        )
        
        # 메인 컨테이너에서 ask 결과 표시
        with st.container():
            with st.chat_message("assistant"):
                main_container = st.container()
                main_results = display_ask_streaming_response(
                    main_queue, 
                    main_container, 
                    error_queue,
                    add_message_func=None  # 실제 사용시 메시지 저장 함수 전달
                )
        
        # 사이드바에서 thinking 결과 표시
        with st.sidebar:
            with st.chat_message("assistant"):
                sidebar_container = st.container()
                sidebar_results = display_ask_streaming_response(
                    sidebar_queue,
                    sidebar_container,
                    error_queue
                )
        
        # 결과 활용
        if main_results["text"]:
            st.success("응답 완료!")

# 앱 실행
if __name__ == "__main__":
    example_usage()
