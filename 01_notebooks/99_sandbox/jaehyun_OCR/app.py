"""
app.py
------
대시보드 진입점. 레이아웃 조립만 담당하며 모든 로직은 utils.py 에 있습니다.

실행 방법
---------
    streamlit run app.py

의존성 설치
-----------
    pip install streamlit plotly pandas
"""

import streamlit as st

import utils

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix 이탈 예측 분석 대시보드",
    page_icon="🎬",
    layout="wide",
)

# ── 전역 스타일 · 헤더 · 툴바 ────────────────────────────────────────────────
utils.inject_css()
utils.render_header()
utils.render_toolbar()

# ── Section 1: KPI 카드 ────────────────────────────────────────────────────────
cols = st.columns(3)
for col, item in zip(cols, utils.KPI_DATA):
    utils.render_kpi_card(col, item)

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)

# ── Section 2: 월별 추세 + 위험 세분화 ────────────────────────────────────────
col_trend, col_donut = st.columns([2, 1])

with col_trend:
    with st.container(border=True):
        st.markdown('<p class="card-title">📈 월별 이탈 추세</p>', unsafe_allow_html=True)
        st.plotly_chart(
            utils.make_trend_chart(utils.MONTHLY_TREND),
            use_container_width=True,
        )

with col_donut:
    utils.render_risk_donut(utils.RISK_SEGMENTS)

st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

# ── Section 3: OTT 빈도 · 장르 · 이탈 요인 ────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    utils.render_ott_usage(utils.OTT_USAGE)

with c2:
    utils.render_genre_chart(utils.GENRES)

with c3:
    utils.render_churn_drivers(utils.CHURN_DRIVERS)

st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

# ── Section 4: 높은 이탈 위험 사용자 ──────────────────────────────────────────
utils.render_high_risk_users(utils.HIGH_RISK_USERS)
