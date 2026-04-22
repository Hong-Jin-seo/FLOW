"""
ui/styles.py
전역 CSS 상수 및 주입 함수
"""
import streamlit as st

GLOBAL_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* Streamlit 기본 UI 제거 */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
[data-testid="stSidebar"]    { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stHeader"]     { display: none !important; }
.block-container { padding: 0 96px !important; max-width: 100% !important; }

/* ── 상단 네비게이션 ── */
.d-nav {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid #e5e7eb;
    margin: 0 -96px;
    padding: 0 96px; height: 64px;
    display: flex; align-items: center; justify-content: space-between;
}
.d-nav-logo { display:flex; align-items:center; gap:10px; text-decoration:none !important; color:inherit !important; }
.d-nav-logo-icon {
    width:36px; height:36px; background:#059669; border-radius:9px;
    display:flex; align-items:center; justify-content:center; font-size:1.1rem;
}
.d-nav-logo-text { font-size:1.1rem; font-weight:800; color:#111827; }
.d-nav-logo-text em { color:#059669; font-style:normal; }
.d-nav-links { display:flex; align-items:center; gap:36px; }
.d-nav-link {
    display:flex; align-items:center; gap:6px; text-decoration:none;
    font-size:1.05rem; font-weight:600; color:#111827 !important; text-decoration:none !important; transition:color 0.15s;
}
.d-nav-link:hover { color:#111827; }
.d-nav-link.active { color:#111827; font-weight:600; }

/* ── 페이지 래퍼 ── */
.d-page { max-width:1200px; margin:0 auto; padding:0 30px 100px; }

/* ── 히어로 ── */
.d-hero {
    display:grid; grid-template-columns:1fr 1fr;
    gap:80px; align-items:center; padding:88px 0 80px;
}
.d-hero-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#ecfdf5; border:1px solid #a7f3d0; color:#111827;
    padding:5px 14px; border-radius:8px; font-size:0.75rem; font-weight:600; margin-bottom:24px;
}
.d-hero-title {
    font-size:3.5rem; font-weight:800; color:#111827;
    line-height:1.15; margin-bottom:20px; letter-spacing:-0.02em;
}
.d-hero-sub { font-size:1.05rem; color:#6b7280; line-height:1.75; margin-bottom:36px; }
.d-hero-btns { display:flex; gap:12px; flex-wrap:wrap; }
.d-btn-primary {
    display:inline-flex; align-items:center; gap:8px;
    background:#059669; color:white !important;
    padding:13px 26px; border-radius:10px;
    font-size:0.95rem; font-weight:600; text-decoration:none !important;
    transition:background 0.15s, transform 0.15s;
}
.d-btn-primary:hover { background:#047857; transform:translateY(-1px); }
.d-btn-secondary {
    display:inline-flex; align-items:center; gap:8px;
    background:white; color:#111827 !important;
    padding:13px 26px; border-radius:10px;
    font-size:0.95rem; font-weight:600; text-decoration:none !important;
    border:1px solid #d1d5db; transition:background 0.15s, transform 0.15s;
}
.d-btn-secondary:hover { background:#f9fafb; transform:translateY(-1px); }

/* ── 히어로 비주얼 ── */
.d-hero-visual {
    position:relative; height:380px;
    display:flex; align-items:center; justify-content:center;
}
.d-blob {
    position:absolute; width:340px; height:340px;
    background:radial-gradient(circle,#ecfdf5 0%,#f0fdf4 55%,transparent 100%);
    border-radius:50%; filter:blur(50px); opacity:0.9;
}
.d-icon-main {
    position:absolute; width:128px; height:128px;
    background:#059669; border-radius:30px;
    display:flex; align-items:center; justify-content:center; font-size:3.2rem;
    box-shadow:0 24px 48px rgba(5,150,105,0.28);
    animation:float-main 6s ease-in-out infinite;
}
.d-icon-spark {
    position:absolute; top:72px; left:48px;
    width:64px; height:64px; background:#047857; border-radius:50%;
    display:flex; align-items:center; justify-content:center; font-size:1.5rem;
    box-shadow:0 10px 24px rgba(4,120,87,0.3);
    animation:float-spark 5s ease-in-out infinite;
}
.d-icon-search {
    position:absolute; top:54px; right:40px;
    width:76px; height:76px; background:white; border-radius:20px;
    display:flex; align-items:center; justify-content:center; font-size:1.9rem;
    box-shadow:0 10px 28px rgba(0,0,0,0.10); border:1px solid #f3f4f6;
    animation:float-search 5.5s ease-in-out infinite;
}
@keyframes float-main  { 0%,100%{transform:translateY(0) rotate(0deg);}  50%{transform:translateY(-18px) rotate(4deg);} }
@keyframes float-spark  { 0%,100%{transform:translateY(0) translateX(0);} 50%{transform:translateY(14px) translateX(8px);} }
@keyframes float-search { 0%,100%{transform:translateY(0);}               50%{transform:translateY(-12px) translateX(-5px);} }

/* ── 기능 카드 ── */
.d-section { padding:64px 0 0; }
.d-section-head { text-align:center; margin-bottom:48px; }
.d-section-head h2 { font-size:1.9rem; font-weight:800; color:#111827; margin-bottom:10px; }
.d-section-head p  { color:#6b7280; font-size:1rem; }
.d-features { display:grid; grid-template-columns:repeat(3,1fr); gap:24px; margin-bottom:60px; }
.d-feature-card {
    background:white; border:1px solid #e5e7eb; border-radius:20px; padding:32px;
    transition:box-shadow 0.2s, transform 0.2s;
}
.d-feature-card:hover { box-shadow:0 8px 32px rgba(0,0,0,.10); transform:translateY(-2px); }
.d-feature-icon { width:54px; height:54px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.6rem; margin-bottom:20px; }
.d-feature-card h3 { font-size:1.1rem; font-weight:700; color:#111827; margin-bottom:10px; }
.d-feature-card p  { font-size:.875rem; color:#6b7280; line-height:1.65; margin-bottom:20px; }
.d-feature-link { display:inline-flex; align-items:center; gap:5px; font-size:.875rem; font-weight:700; color:#111827; text-decoration:none; transition:color .15s; }
.d-feature-link:hover { color:#059669; }

/* ── 통계 배너 ── */
.d-stats {
    background:#f0fdf4; border:1px solid #a7f3d0; border-radius:20px; padding:40px 56px;
    display:grid; grid-template-columns:repeat(4,1fr); gap:24px; text-align:center;
    margin-bottom:80px;
}
.d-stat-val { font-size:2.1rem; font-weight:800; color:#059669; }
.d-stat-lbl { font-size:.875rem; color:#6b7280; margin-top:4px; }

/* ── 내부 페이지 헤더 ── */
.d-page-header { border-bottom:1px solid #f3f4f6; padding:10px 0 0px; margin-bottom:10px; }
.d-page-header h1 { font-size:1.7rem; font-weight:800; color:#111827; margin:0; }
.d-page-header p  { font-size:.875rem; color:#9ca3af; margin:6px 0 0; }

/* ── EWG 배지 ── */
.badge-g { background:#d1fae5; color:#065f46; font-size:.72rem; font-weight:700; padding:2px 9px; border-radius:99px; }
.badge-y { background:#fef3c7; color:#92400e; font-size:.72rem; font-weight:700; padding:2px 9px; border-radius:99px; }
.badge-r { background:#fee2e2; color:#991b1b; font-size:.72rem; font-weight:700; padding:2px 9px; border-radius:99px; }

/* ── 스캐너 결과 행 ── */
.d-upload-area { border:2px dashed #d1d5db; border-radius:20px; padding:52px 24px; text-align:center; background:#f9fafb; }
.d-scan-row { display:flex; align-items:center; justify-content:space-between; padding:11px 14px; border-radius:10px; background:#f9fafb; margin-bottom:8px; }

/* ── 제품 카드 ── */
.d-prod-card { background:white; border:1px solid #e5e7eb; border-radius:12px; padding:14px 18px; margin-top:8px; display:flex; align-items:center; justify-content:space-between; }
.d-prod-brand { font-size:.73rem; font-weight:700; color:#9ca3af; }
.d-prod-name  { font-size:.93rem; font-weight:700; color:#111827; margin-top:2px; }

/* ── RAG 배너 ── */
.d-rag { background:#ecfdf5; border:1px solid #a7f3d0; border-radius:14px; padding:14px 20px; margin-bottom:24px; font-size:.85rem; color:#065f46; }

/* ── 예시 칩 버튼 ── */
div[data-testid="stButton"] button {
    background: white !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 999px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
div[data-testid="stButton"] button:hover {
    background: #f0fdf4 !important;
    border-color: #6ee7b7 !important;
    color: #059669 !important;
}
</style>
"""

def inject() -> None:
    """CSS를 현재 Streamlit 페이지에 주입한다."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)