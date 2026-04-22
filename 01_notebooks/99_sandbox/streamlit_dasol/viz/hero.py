"""
viz/hero.py
히어로 섹션 오른쪽 비주얼 — CSS 애니메이션 플로팅 아이콘
"""
import streamlit as st


def html() -> str:
    """히어로 오른쪽 플로팅 아이콘 영역 HTML을 반환한다.

    CSS 애니메이션(float-main / float-spark / float-search)은
    ui/styles.py의 GLOBAL_CSS에 정의되어 있다.
    """
    return (
        '<div class="d-hero-visual">'
        '  <div class="d-blob"></div>'
        '  <div class="d-icon-main">\U0001f343</div>'
        '  <div class="d-icon-spark">✨</div>'
        '  <div class="d-icon-search">\U0001f50d</div>'
        '</div>'
    )


def render() -> None:
    """히어로 오른쪽 플로팅 아이콘 영역을 렌더링한다. (단독 사용 시)"""
    st.markdown(html(), unsafe_allow_html=True)
