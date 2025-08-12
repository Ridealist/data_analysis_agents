import streamlit as st


pages = {
    "📊 Data Analysis Agent": [
        st.Page("pages/00_Main.py", title="🚪 메인 페이지"),
        # st.Page("pages/01_CSV_Agent_With_History_SeoulTrain.py", title="🚆 서울지하철 승하차 분석"),
        # st.Page("pages/02_CSV_Agent_With_History_KpopRanking.py", title="🎤 Kpop 음원순위 분석"),
        st.Page("pages/03_CSV_Agent_With_School_Food.py", title="🍜 학교 급식 분석"),
    ],
}

pg = st.navigation(pages)
pg.run()
