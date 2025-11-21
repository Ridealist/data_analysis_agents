import streamlit as st


pages = {
    "🌱 Learning AI Agent": [
        st.Page("src/pages/00_Main.py", title="🚪 메인 페이지"),
        st.Page("src/pages/01_Learning_Stage1_Planning.py", title="📝 Stage 1: 계획"),
        st.Page("src/pages/02_Learning_Stage2_Execution.py", title="🚀 Stage 2: 실행"),
        st.Page("src/pages/03_Learning_Stage3_Reflection.py", title="🪞 Stage 3: 성찰"),
    ],
}

pg = st.navigation(pages)
pg.run()
