import streamlit as st


pages = {
    "📊 Data Analysis Agent": [
        st.Page("src/pages/00_Main.py", title="🚪 메인 페이지"),
        st.Page("src/pages/01_Prepro_Agent.py", title="🏗️ 데이터 전처리"),
        st.Page("src/pages/02_CSV_Agent.py", title="🕵🏻‍♂️ 데이터 분석"),
    ],
}

pg = st.navigation(pages)
pg.run()
