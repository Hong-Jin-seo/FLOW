"""
views/scanner.py
성분 스캐너 페이지 — 이미지 업로드 + OCR 실제 분석 + 결과 표시
"""
import streamlit as st
from services import api
from ui import components
from state import session as sess


def _ewg_grade(ewg: int | None) -> tuple[str, str]:
    """EWG 0-10 점수 → (grade, score_str)"""
    if ewg is None:
        return "green", "?"
    if ewg <= 2:
        return "green",  str(ewg)
    if ewg <= 6:
        return "yellow", str(ewg)
    return "red", str(ewg)


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
    uploaded = st.file_uploader(
        "이미지 업로드 (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key=f"scanner_upload_{st.session_state.scan_upload_key}",
    )
    if uploaded:
        new_bytes = uploaded.read()
        if new_bytes != st.session_state.scan_image:
            st.session_state.scan_image   = new_bytes
            st.session_state.scan_done    = False
            st.session_state.scan_results = None

    if st.session_state.scan_image:
        st.image(st.session_state.scan_image, use_container_width=True, caption="업로드된 이미지")
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.scan_done:
            if st.button("🔬  성분 스캔하기", use_container_width=True, type="primary", key="do_scan"):
                with st.spinner("📡 AI가 텍스트를 인식하고 성분을 분석하는 중..."):
                    try:
                        data = api.scan(
                            st.session_state.scan_image,
                            filename=getattr(uploaded, "name", "image.jpg") if uploaded else "image.jpg",
                        )
                        st.session_state.scan_results = data
                        st.session_state.scan_done    = True
                    except api.APIError as err:
                        st.error(str(err))
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            if st.button("🔄  다른 사진 스캔", use_container_width=True, key="reset_scan"):
                sess.reset_scanner()
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
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
        st.markdown(
            '<div style="border:1px solid #e5e7eb; border-radius:20px; padding:80px 24px; '
            'text-align:center; min-height:360px; background:white; '
            'display:flex; flex-direction:column; align-items:center; justify-content:center;">'
            '  <div style="font-size:3.5rem; opacity:0.25; margin-bottom:16px;">🖼️</div>'
            '  <div style="color:#9ca3af; font-size:.9rem; line-height:1.7;">'
            '    라벨 사진을 업로드하면<br>이곳에 분석 결과가 표시됩니다.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.scan_done or not st.session_state.scan_results:
        st.markdown(
            '<div style="border:1px solid #e5e7eb; border-radius:20px; padding:80px 24px; '
            'text-align:center; min-height:360px; background:white; '
            'display:flex; flex-direction:column; align-items:center; justify-content:center;">'
            '  <div style="font-size:3rem; opacity:0.4; margin-bottom:16px;">👆</div>'
            '  <div style="color:#9ca3af; font-size:.9rem;">'
            '    이미지가 준비되었습니다.<br>스캔 버튼을 눌러주세요.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    data        = st.session_state.scan_results
    ingredients = data["ingredients"]

    if not ingredients:
        st.warning("성분을 인식하지 못했습니다. 더 선명한 사진을 업로드해주세요.")
        return

    st.markdown(
        f'<div style="display:flex; align-items:center; justify-content:space-between; '
        f'border-bottom:1px solid #f3f4f6; padding-bottom:12px; margin-bottom:16px;">'
        f'  <span style="font-size:1rem; font-weight:700; color:#111827;">추출된 성분 목록</span>'
        f'  <span style="background:#ecfdf5; color:#065f46; font-size:.75rem; font-weight:700; '
        f'    padding:4px 10px; border-radius:99px;">{data["total"]}개 성분</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    sorted_ingredients = sorted(ingredients, key=lambda x: x["ewg"] if x["ewg"] is not None else -1)
    for item in sorted_ingredients:
        grade, score = _ewg_grade(item["ewg"])
        desc = item.get("function") or ""
        components.scan_result_row(item["ingredient"], grade, score, desc)

    safe = data["total"] - data["danger_count"] - data["caution_count"]
    components.summary_box(data["total"], safe, data["caution_count"], data["danger_count"])
    st.markdown("<div style='padding-bottom:48px;'></div>", unsafe_allow_html=True)
