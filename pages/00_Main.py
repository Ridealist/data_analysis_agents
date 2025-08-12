import streamlit as st
import time

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None


# # Logo
# st.logo(
#     "https://media.licdn.com/dms/image/v2/C511BAQFHW_naY__2Fg/company-background_10000/company-background_10000/0/1583927014937/iled_lighting_systems_pvt_ltd__cover?e=2147483647&v=beta&t=Y1x2WJMstxhMwG8RDFgTgTQbhYyn6Us6rRGDRtsiaoA",
#     link="https://iled.snu.ac.kr/",
#     size="large"
# )

st.title("📊 Data Analysis Agent")
st.write("**데이터 분석 에이전트**는 데이터를 분석하고 시각화할 수 있는 ChatGPT 기반의 도구입니다.")

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
            st.session_state.logged_in = True
            st.session_state["session_id"] = str(users_info[username]["userID"])
            st.success("로그인 성공!")
            time.sleep(0.5)
            st.switch_page("pages/03_CSV_Agent_With_School_Food.py")
