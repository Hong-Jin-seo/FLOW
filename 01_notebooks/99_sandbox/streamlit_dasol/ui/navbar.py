"""ui/navbar.py — 상단 네비게이션 바"""
import streamlit as st

_NAV_ITEMS = [
    ("analysis",       "🔍", "성분 분석"),
    ("scanner",        "↺",  "성분 스캐너"),
    ("recommendation", "✨", "제품 추천"),
]

def render(current_page: str) -> None:
    link_parts = []
    for pid, icon, label in _NAV_ITEMS:
        active = " active" if current_page == pid else ""
        link_parts.append(
            f'<a href="?page={pid}" target="_self" class="d-nav-link{active}">{icon}&nbsp;{label}</a>'
        )
    links_html = "\n".join(link_parts)

    html = f"""
<nav class="d-nav">
  <a href="?page=home" target="_self" class="d-nav-logo">
    <div class="d-nav-logo-icon">🌿</div>
    <span class="d-nav-logo-text">DermaLens<em>.</em></span>
  </a>
  <div class="d-nav-links">
    {links_html}
  </div>
</nav>
"""
    st.markdown(html, unsafe_allow_html=True)
