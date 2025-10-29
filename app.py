import streamlit as st


pages = {
    # "📊 Data Analysis Agent": [
    #     st.Page("src/pages/00_Main.py", title="🚪 메인 페이지"),
    #     st.Page("src/pages/02_CSV_Agent.py", title="🕵🏻‍♂️ 데이터 분석"),
    # ],
    "🌱 Learning Agent (계획→실행→반성)": [
        st.Page("src/pages/06_Learning_Stage1_Planning.py", title="📝 Stage 1: 계획"),
        st.Page("src/pages/07_Learning_Stage2_Execution.py", title="🚀 Stage 2: 실행"),
        st.Page("src/pages/08_Learning_Stage3_Reflection.py", title="🪞 Stage 3: 반성"),
    ],
}

pg = st.navigation(pages)
pg.run()
