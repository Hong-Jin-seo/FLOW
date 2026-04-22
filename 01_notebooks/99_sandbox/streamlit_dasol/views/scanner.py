"""
views/scanner.py
성분 스캐너 페이지 — 이미지 업로드 + OCR 시뮬레이션 + 결과 표시
"""
import time
import streamlit as st
from ui import components
from state import session as sess

MOCK_INGREDIENTS: list[dict] = [
    {"name": "정제수",                 "grade": "green",  "score": "1",   "desc": ""},
    {"name": "부틸렌글라이콜",         "grade": "green",  "score": "1",   "desc": ""},
    {"name": "글리세린",               "grade": "green",  "score": "1-2", "desc": ""},
    {"name": "나이아신아마이드",       "grade": "green",  "score": "1",   "desc": ""},
    {"name": "판테놀",                 "grade": "green",  "score": "1",   "desc": ""},
    {"name": "센텔라아시아티카추출물", "grade": "green",  "score": "1",   "desc": ""},
    {"name": "향료",                   "grade": "red",    "score": "8",   "desc": "민감성 주의"},
    {"name": "페녹시에타놀",          "grade": "yellow", "score": "4",   "desc": "보존제"},
]

_UPLOADER_CSS = """
<style>
[data-testid="stFileUploader"] > label { display:none !important; }
[data-testid="stFileUploader"] section { border:none !important; padding:0 !important; background:transparent !important; }
[data-testid="stFileUploader"] section > div { display:none !important; }
[data-testid="stFileUploadDropzone"] { display:none !important; }
</style>
"""

_UPLOAD_CARD = (
    '<div style="border:2px dashed #d1d5db; border-radius:20px; padding:52px 24px; '
    'background:#f9fafb; text-align:center; min-height:360px; '
    'display:flex; flex-direction:column; align-items:center; justify-content:center;">'
    '  <div style="width:72px; height:72px; border-radius:50%; '
    '    background:#ecfdf5; border:1px solid #a7f3d0; '
    '    display:flex; align-items:center; justify-content:center; '
    '    font-size:2rem; margin-bottom:20px;">\u2b06\ufe0f</div>'
    '  <div style="font-size:1.1rem; font-weight:700; color:#111827; margin-bottom:8px;">'
    '    이미지 드래그 앤 드롭</div>'
    '  <div style="font-size:.875rem; color:#9ca3af; margin-bottom:24px;">'
    '    또는 클릭하여 파일 선택 (PNG, JPG)</div>'
    '</div>'
)

_EMPTY_RESULT = (
    '<div style="border:1px solid #e5e7eb; border-radius:20px; padding:80px 24px; '
    'text-align:center; min-height:360px; background:white; '
    'display:flex; flex-direction:column; align-items:center; justify-content:center;">'
    '  <div style="font-size:3.5rem; opacity:0.25; margin-bottom:16px;">\U0001f5bc\ufe0f</div>'
    '  <div style="color:#9ca3af; font-size:.9rem; line-height:1.7;">'
    '    라벨 사진을 업로드하면<br>이곳에 분석 결과가 표시됩니다.</div>'
    '</div>'
)

_READY_RESULT = (
    '<div style="border:1px solid #e5e7eb; border-radius:20px; padding:80px 24px; '
    'text-align:center; min-height:360px; background:white; '
    'display:flex; flex-direction:column; align-items:center; justify-content:center;">'
    '  <div style="font-size:3rem; opacity:0.4; margin-bottom:16px;">\U0001f446</div>'
    '  <div style="color:#9ca3af; font-size:.9rem;">'
    '    이미지가 준비되었습니다.<br>스캔 버튼을 눌러주세요.</div>'
    '</div>'
)


def render() -> None:
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; padding:32px 0 36px;">'
        '  <h1 style="font-size:2.2rem; font-weight:800; color:#111827; margin:0 0 12px;">'
        '    화장품 성분 스캐너</h1>'
        '  <p style="font-size:1rem; color:#6b7280; margin:0;">'
        '    복잡한 전성분 표기를 사진으로 찍어 올리세요. OCR(광학문자인식) 기술로 자동 분석해 드립니다.'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_upload, col_result = st.columns([1, 1], gap="large")
    with col_upload:
        _render_upload_panel()
    with col_result:
        _render_result_panel()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_upload_panel() -> None:
    """왼쪽 이미지 업로드 패널"""
    uploaded = st.file_uploader(
        "이미지 업로드 (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="scanner_upload",
    )
    if uploaded:
        new_bytes = uploaded.read()
        if new_bytes != st.session_state.scan_image:
            st.session_state.scan_image = new_bytes
            st.session_state.scan_done = False

    if st.session_state.scan_image:
        st.image(st.session_state.scan_image, use_container_width=True, caption="업로드된 이미지")
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.scan_done:
            if st.button("🔬  성분 스캔하기", use_container_width=True, type="primary", key="do_scan"):
                with st.spinner("📡 AI가 텍스트를 인식하고 성분을 분석하는 중..."):
                    time.sleep(2.2)
                st.session_state.scan_done = True
                st.rerun()
        else:
            if st.button("🔄  다른 사진 스캔", use_container_width=True, key="reset_scan"):
                sess.reset_scanner()
                st.rerun()
    else:
        st.markdown(
            '<div class="d-upload-area">'
            '  <div style="font-size:2.8rem;margin-bottom:14px;">📷</div>'
            '  <div style="font-weight:700;color:#374151;margin-bottom:6px;">화장품 성분표 사진을 올려주세요</div>'
            '  <div style="font-size:.8rem;color:#9ca3af;">PNG · JPG · JPEG 지원</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_result_panel() -> None:
    if not st.session_state.scan_image:
        st.markdown(_EMPTY_RESULT, unsafe_allow_html=True)
        return

    if not st.session_state.scan_done:
        st.markdown(_READY_RESULT, unsafe_allow_html=True)
        return

    st.markdown(
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'border-bottom:1px solid #f3f4f6; padding-bottom:12px; margin-bottom:16px;">'
        '  <span style="font-size:1rem; font-weight:700; color:#111827;">추출된 성분 목록</span>'
        '  <span style="background:#ecfdf5; color:#065f46; font-size:.75rem; font-weight:700; '
        '    padding:4px 10px; border-radius:99px;">인식률 98%</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    for item in MOCK_INGREDIENTS:
        components.scan_result_row(item["name"], item["grade"], item["score"], item["desc"])

    safe = sum(1 for i in MOCK_INGREDIENTS if i["grade"] == "green")
    warn = sum(1 for i in MOCK_INGREDIENTS if i["grade"] in ("yellow", "red"))
    components.summary_box(len(MOCK_INGREDIENTS), safe, warn)
