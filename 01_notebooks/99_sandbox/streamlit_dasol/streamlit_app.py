"""
streamlit_app.py  —  DermaLens 진입점

역할: 설정 · CSS 주입 · 네비게이션 바 · 페이지 라우팅만 담당한다.
비즈니스 로직은 각 모듈로 분리되어 있다.

    state/session.py      세션 상태 초기화 / 리셋
    services/api.py       FastAPI 백엔드 클라이언트
    ui/styles.py          전역 CSS
    ui/navbar.py          상단 네비게이션 바
    ui/components.py      재사용 UI 컴포넌트
    viz/hero.py           히어로 비주얼 (플로팅 아이콘)
    views/home.py         홈 페이지
    views/analysis.py     성분 분석 페이지
    views/scanner.py      성분 스캐너 페이지
    views/recommendation.py  제품 추천 페이지

실행 방법:
    uvicorn app.main:app --reload        # 백엔드 먼저 실행
    streamlit run streamlit_app.py       # 프론트엔드 실행
"""

import streamlit as st

# ── 페이지 기본 설정 (반드시 첫 번째 st 호출)
st.set_page_config(
    page_title="DermaLens | AI 스킨케어",
    page_icon="\U0001f33f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 모듈 임포트
from state import session as sess
from ui import styles, navbar
from views import home, analysis, scanner, recommendation

# ── 세션 상태 초기화
sess.init()

# ── 전역 CSS 주입
styles.inject()

# ── 현재 페이지 (URL query param 기반)
page = st.query_params.get("page", "home")

# ── 상단 네비게이션 바
navbar.render(page)

# ── 페이지 라우터
_ROUTES: dict = {
    "home":           home.render,
    "analysis":       analysis.render,
    "scanner":        scanner.render,
    "recommendation": recommendation.render,
}

renderer = _ROUTES.get(page, home.render)
renderer()
