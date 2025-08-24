import streamlit as st


pages = {
    "📊 Data Analysis Agent": [
        st.Page("src/pages/00_Main.py", title="🚪 메인 페이지"),
        st.Page("src/pages/01_CSV_Agent.py", title="🍜 학교 급식 분석"),
    ],
}

pg = st.navigation(pages)
pg.run()
