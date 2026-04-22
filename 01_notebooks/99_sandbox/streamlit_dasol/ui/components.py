"""
ui/components.py
재사용 가능한 UI 컴포넌트 모음
- page_header()      : 내부 페이지 제목 + 부제
- ewg_badge()        : EWG 등급 배지 HTML
- scan_result_row()  : 스캐너 성분 결과 한 줄 HTML
- product_card()     : 추천 제품 카드 HTML
- rag_banner()       : RAG 안내 배너
- summary_box()      : 분석 요약 다크 박스
"""
from __future__ import annotations
import streamlit as st

# EWG 등급 → CSS 클래스 / 아이콘 매핑
_GRADE_CSS  = {"green": "badge-g", "yellow": "badge-y", "red": "badge-r"}
_GRADE_ICON = {"green": "✅",   "yellow": "⚠️", "red": "\U0001f6a8"}


def page_header(title: str, subtitle: str) -> None:
    """페이지 제목 + 부제를 렌더링한다."""
    st.markdown(
        f'''<div class="d-page-header">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>''',
        unsafe_allow_html=True,
    )


def ewg_badge(grade: str, score: str) -> str:
    """EWG 등급 배지 HTML 문자열을 반환한다."""
    cls  = _GRADE_CSS.get(grade, "badge-g")
    icon = _GRADE_ICON.get(grade, "")
    return f'<span class="{cls}">{icon} EWG {score}</span>'


def scan_result_row(name: str, grade: str, score: str, desc: str = "") -> None:
    """스캐너 성분 결과 행을 렌더링한다."""
    icon     = _GRADE_ICON.get(grade, "❓")
    badge_cls = _GRADE_CSS.get(grade, "badge-g")
    desc_html = (
        f'<span style="font-size:.7rem;color:#ef4444;background:#fee2e2;'
        f'padding:1px 6px;border-radius:4px;margin-right:4px;">{desc}</span>'
        if desc else ""
    )
    st.markdown(
        f'''<div class="d-scan-row">
          <span>{icon} <b style="font-size:.88rem;">{name}</b></span>
          <span>{desc_html}<span class="{badge_cls}">EWG {score}</span></span>
        </div>''',
        unsafe_allow_html=True,
    )


def product_card(name: str, brand: str, tags: list[str]) -> None:
    """추천 제품 카드를 렌더링한다."""
    tags_html = "".join(
        f'<span style="font-size:.72rem;font-weight:600;background:#ecfdf5;'
        f'color:#065f46;padding:2px 8px;border-radius:6px;margin-right:4px;">{t}</span>'
        for t in tags
    )
    st.markdown(
        f'''<div class="d-prod-card">
          <div>
            <div class="d-prod-brand">{brand}</div>
            <div class="d-prod-name">{name}</div>
          </div>
          <div>{tags_html}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def rag_banner() -> None:
    """RAG 기반 추천 안내 배너를 렌더링한다."""
    st.markdown(
        '''<div class="d-rag">
          \U0001f9ec <strong>RAG 기반 개인화 추천 시스템</strong><br>
          <span style="font-size:.8rem;">
            입력하신 고민은 수만 건의 화장품 성분
            데이터베이스(화해, COOS 등)와 교차 검증되어,
            안전하고 효과적인 성분 위주로 분석됩니다.
          </span>
        </div>''',
        unsafe_allow_html=True,
    )


def summary_box(total: int, safe: int, warn: int) -> None:
    """분석 요약 다크 박스를 렌더링한다."""
    st.markdown(
        f'''<div style="background:#111827;color:#f9fafb;border-radius:12px;
                     padding:14px 18px;margin-top:16px;font-size:.85rem;line-height:1.65;">
          <strong>\U0001f50e 분석 요약</strong><br>
          전체 {total}개 성분 중
          <span style="color:#6ee7b7;font-weight:700;">안전 {safe}개</span>,
          <span style="color:#fca5a5;font-weight:700;">주의 {warn}개</span>
          검출되었습니다.
          민감성 피부의 경우 패치 테스트를 권장합니다.
        </div>''',
        unsafe_allow_html=True,
    )
