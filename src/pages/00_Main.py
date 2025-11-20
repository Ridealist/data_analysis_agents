import streamlit as st
import time
import uuid

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None


st.title("🌱 Learning AI Agent")
st.write("**학습 AI 에이전트**는 생성형 AI를 주도성을 갖고 사용할 수 있도록 도와주는 학습 파트너입니다.")

# st.write("Please log in to continue (username `test`, password `test`).")
# secrets.toml에서 사용자 정보 가져오기
users_info = st.secrets["users"]

# 로그인 UI
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Log in", type="primary"):
    if username not in users_info:
        st.error("Username을 확인해주세요.")
    else:
        if password != users_info[username]["password"]:
            st.error("Password를 확인해주세요.")
        else:
            # 로그인 성공 시 모든 필요한 session state 업데이트
            st.session_state.logged_in = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users_info[username]["userID"]
            
            # 01_Prepro_Agent.py에서 필요한 session state들 초기화
            st.session_state["session_uuid"] = str(uuid.uuid4())
            st.session_state["conversation_id"] = None
            
            st.success(f"로그인 성공! 환영합니다, {username}님!")
            time.sleep(0.5)
            # 로그인 후 Stage 1: 계획 페이지로 이동
            st.switch_page("src/pages/01_Learning_Stage1_Planning.py")
