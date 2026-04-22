"""
views/home.py
홈 페이지 — 히어로 섹션 + 핵심 서비스 카드 + 통계 배너
"""
import streamlit as st
from viz.hero import html as hero_html


_FEATURES = [
    {
        "icon": "\U0001f50d", "color": "#eff6ff",
        "title": "성분 분석",
        "desc": "어려운 INCI 명칭과 영문 성분명을 "
                "일반 소비자도 이해하기 쉽게 설명해 드립니다. "
                "EWG 등급과 함께 확인하세요.",
        "page": "analysis",
    },
    {
        "icon": "\U0001f4f7", "color": "#ecfdf5",
        "title": "성분 스캐너",
        "desc": "화장품 뒷면의 라벨을 사진으로 찍어 올리면, "
                "OCR 기술을 통해 성분을 자동 추출하고 "
                "위험도를 분석합니다.",
        "page": "scanner",
    },
    {
        "icon": "✨", "color": "#faf5ff",
        "title": "맞춤형 제품 추천",
        "desc": "피부 고민을 이야기해주세요. "
                "RAG 기술이 결합된 LLM이 수만 개의 "
                "데이터를 기반으로 최적의 제품을 추천합니다.",
        "page": "recommendation",
    },
]

_STATS = [
    ("10,000+", "분석된 제품 수"),
    ("3,000+",  "성분 데이터베이스"),
    ("EWG",     "안전성 등급 기준"),
    ("3종",   "데이터 소스 (화해·COOS·PC)"),
]


def render() -> None:
    """홈 페이지 전체를 렌더링한다."""
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    _render_hero()
    _render_features()
    _render_stats()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_hero() -> None:
    """히어로 섹션 — 좌: 텍스트·버튼  /  우: 플로팅 아이콘 비주얼
    단일 st.markdown() 호출로 합쳐야 Streamlit wrapper div 간섭 없이
    CSS flex 레이아웃이 제대로 적용된다.
    """
    html = (
        '<div class="d-hero">'
        '  <div class="d-hero-left">'
        '    <div class="d-hero-badge">✨ AI 기반 스킨케어 안전성 제품 시스템</div>'
        '    <div class="d-hero-title">내 피부에 맞는<br>안전한 성분을 찾다</div>'
        '    <div class="d-hero-sub">어려운 화장품 성분, 이제 AI가 분석해 드립니다.<br>'
        '      피부 고민에 맞는 성분을 찾고, 라벨을 스캔하여 위험도를 확인하세요.</div>'
        '    <div class="d-hero-btns">'
        '      <a href="?page=analysis" target="_self" class="d-btn-primary">성분 분석 시작하기 →</a>'
        '      <a href="?page=scanner"  target="_self" class="d-btn-secondary">성분 스캐너 ↻</a>'
        '    </div>'
        '  </div>'
        + hero_html()
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_features() -> None:
    """핵심 서비스 3개 카드 섹션"""
    st.markdown(
        '<div class="d-section">'
        '  <div class="d-section-head">'
        '    <h2>핵심 서비스</h2>'
        '    <p>다양한 데이터 소스를 통합하여, 사용자 맞춤형 화장품 성분 정보를 제공합니다.</p>'
        '  </div>',
        unsafe_allow_html=True,
    )
    cards_html = '<div class="d-features">'
    for f in _FEATURES:
        cards_html += (
            f'<div class="d-feature-card">'
            f'  <div class="d-feature-icon" style="background:{f["color"]};">{f["icon"]}</div>'
            f'  <h3>{f["title"]}</h3>'
            f'  <p>{f["desc"]}</p>'
            f'  <a href="?page={f["page"]}" target="_self" class="d-feature-link">자세히 보기 →</a>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_stats() -> None:
    """통계 배너"""
    items_html = "".join(
        f'<div><div class="d-stat-val">{val}</div><div class="d-stat-lbl">{lbl}</div></div>'
        for val, lbl in _STATS
    )
    st.markdown(
        f'<div class="d-stats">{items_html}</div>',
        unsafe_allow_html=True,
    )
