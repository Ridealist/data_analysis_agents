import os
import streamlit as st
from openai import RateLimitError, APITimeoutError, InternalServerError

#-----------------------------#
#      API 키 설정 함수        #
#-----------------------------#
# API 키 설정 함수
def set_openai_api_key():
    """OpenAI API 키를 설정하는 함수"""
    # 기본값: secrets에서 가져오기
    default_api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    # 사이드바에서 입력된 API 키가 있으면 해당 키 사용
    if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
        api_key = st.session_state["openai_api_key"]
    else:
        api_key = default_api_key
    
    # 환경 변수에 설정
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    else:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        return False

#--------------------------------#
#      OpenAI 에러 처리 함수      #
#--------------------------------#
def handle_openai_error(error, current_model):
    """OpenAI 에러를 처리하고 더 경량 모델로 전환하는 함수"""
    
    # 모델 전환 순서 (무거운 모델 → 가벼운 모델)
    model_fallback_chain = [
        "gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"
    ]
    
    # 현재 모델의 인덱스 찾기
    try:
        current_index = model_fallback_chain.index(current_model)
        # 더 가벼운 모델들만 선택
        fallback_models = model_fallback_chain[current_index + 1:]
    except ValueError:
        # 현재 모델이 체인에 없으면 기본값 사용
        fallback_models = ["gpt-4o-mini"]
    
    # 에러 타입별 메시지 생성
    if isinstance(error, RateLimitError):
        # RateLimitError의 세부 내용 분석
        error_str = str(error)
        if "insufficient_quota" in error_str or "You exceeded your current quota" in error_str:
            # 1번 케이스: Quota 초과 (결제/크레딧 부족)
            error_msg = f"💰 **크레딧 부족**: {current_model} 모델의 사용 가능한 크레딧이 부족합니다."
            error_msg += f"\n\n📋 **에러 상세**: {error_str}"
            error_msg += f"\n\n💳 **해결 방법**: OpenAI 계정에 크레딧을 추가하거나 결제 정보를 확인해주세요."
        elif "rate_limit_exceeded" in error_str or "Rate limit reached" in error_str:
            # 2번 케이스: Rate Limit 초과 (요청 제한)
            error_msg = f"⏱️ **요청 제한 초과**: {current_model} 모델의 요청 제한에 도달했습니다."
            error_msg += f"\n\n📋 **에러 상세**: {error_str}"
            error_msg += f"\n\n🔄 **해결 방법**: 잠시 후 다시 시도하거나, 더 가벼운 모델로 전환합니다."
        else:
            # 기타 RateLimitError
            error_msg = f"🚨 **API 요청 한도 초과**: {current_model} 모델의 요청 한도를 초과했습니다."
            error_msg += f"\n\n📋 **에러 상세**: {error_str}"
    elif isinstance(error, APITimeoutError):
        error_msg = f"⏰ **API 응답 시간 초과**: {current_model} 모델의 응답이 너무 느립니다."
    elif isinstance(error, InternalServerError):
        error_msg = f"🔧 **서버 내부 오류**: OpenAI 서버에서 처리 중 오류가 발생했습니다."
    else:
        error_msg = f"❌ **OpenAI API 오류**: {current_model} 모델에서 오류가 발생했습니다."
    
    # 500 에러는 서버 문제이므로 모델 전환 없이 재시도 권장
    if isinstance(error, InternalServerError):
        error_msg += f"\n\n🔄 **재시도 권장**: 잠시 후 다시 시도해주세요. (서버 문제는 일시적입니다)"
        return False, None, error_msg
    
    # RateLimitError의 세부 처리
    if isinstance(error, RateLimitError):
        error_str = str(error)
        if "insufficient_quota" in error_str or "You exceeded your current quota" in error_str:
            # Quota 초과는 모델 전환으로 해결할 수 없음
            error_msg += f"\n\n🔑 **API 키 교체 필요**: 크레딧 부족 문제는 새로운 API 키로 교체해야 합니다."
            return False, None, error_msg
        elif "rate_limit_exceeded" in error_str or "Rate limit reached" in error_str:
            # Rate Limit 초과는 모델 전환으로 해결 가능
            if fallback_models:
                next_model = fallback_models[0]
                error_msg += f"\n\n🔄 **자동 모델 전환**: {next_model} 모델로 전환하여 재시도합니다."
                
                # 모델 전환
                st.session_state["model_option_general"] = next_model
                
                # 성공 메시지 표시
                st.success(f"✅ 모델이 {next_model}로 자동 전환되었습니다!")
                
                return True, next_model, error_msg
            else:
                error_msg += f"\n\n⚠️ **모델 전환 불가**: 더 이상 사용할 수 있는 모델이 없습니다."
                return False, None, error_msg
        else:
            # 기타 RateLimitError는 모델 전환 시도
            if fallback_models:
                next_model = fallback_models[0]
                error_msg += f"\n\n🔄 **자동 모델 전환**: {next_model} 모델로 전환하여 재시도합니다."
                
                # 모델 전환
                st.session_state["model_option_general"] = next_model
                
                # 성공 메시지 표시
                st.success(f"✅ 모델이 {next_model}로 자동 전환되었습니다!")
                
                return True, next_model, error_msg
            else:
                error_msg += f"\n\n⚠️ **모델 전환 불가**: 더 이상 사용할 수 있는 모델이 없습니다."
                return False, None, error_msg
    
    # 기타 에러에 대한 모델 전환 시도
    if fallback_models:
        next_model = fallback_models[0]
        error_msg += f"\n\n🔄 **자동 모델 전환**: {next_model} 모델로 전환하여 재시도합니다."
        
        # 모델 전환
        st.session_state["model_option_general"] = next_model
        
        # 성공 메시지 표시
        st.success(f"✅ 모델이 {next_model}로 자동 전환되었습니다!")
        
        return True, next_model, error_msg
    else:
        error_msg += f"\n\n⚠️ **모델 전환 불가**: 더 이상 사용할 수 있는 모델이 없습니다."
        return False, None, error_msg
