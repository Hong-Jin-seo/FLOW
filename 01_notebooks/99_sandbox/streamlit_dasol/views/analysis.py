"""
views/analysis.py
성분 분석 페이지 — RAG 기반 Q&A 질의응답
"""
import streamlit as st
from services import api
from ui import components
from state import session as sess

_CHIPS = [
    "방부제 성분 화장품 성분 분석",
    "민감성 피부 성분 추천",
    "계면활성제 성분 분석",
]


def render() -> None:
    """성분 분석 페이지를 렌더링합니다."""
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    components.page_header(
        "🌿 성분 분석",
        "궁금한 성분명을 입력하면 안전도를 분석해드립니다",
    )

    search_type = st.selectbox(
        "검색 방식",
        ["dense", "bm25", "rrf", "hyde"],
        index=3,   # hyde 기본 선택
        key="search_type_select"
    )
    st.session_state.search_type = search_type

    _render_empty_state()
    _render_chat_history()
    _handle_input()
    _render_reset_button()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_empty_state() -> None:
    """대화 없을 때 빈 상태 + 예시 질문 버튼"""
    if st.session_state.qa_messages:
        return

    st.markdown(
        '''<div style="text-align:center; padding:60px 0 24px;">
          <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:72px; height:72px; border-radius:50%;
            background:#ecfdf5; border:1px solid #a7f3d0;
            font-size:2rem; margin-bottom:20px;">✨</div>
          <div style="font-size:1.2rem; font-weight:700; color:#111827; margin-bottom:8px;">
            성분 분석을 시작하세요
          </div>
          <div style="font-size:.875rem; color:#9ca3af; margin-bottom:28px;">
            예시: "방부제 성분이 포함된 화장품 성분 추천"
          </div>
        </div>''',
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1, 1, 1])
    for i, chip in enumerate(_CHIPS):
        with cols[i + 1]:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state.qa_prefill = chip
                st.rerun()


def _render_chat_history() -> None:
    """이전 대화 기록을 순서대로 렌더링합니다."""
    for msg in st.session_state.qa_messages:
        avatar = "🌿" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 참고 문서 보기"):
                    for src in msg["sources"]:
                        st.markdown(f'**📦 {src["product_name"]}**')
                        st.code(src["content"], language=None)


def _handle_input() -> None:
    """질문 입력을 받아 API 호출"""
    prefill    = st.session_state.pop("qa_prefill", None)
    user_input = st.chat_input("예시: 방부제 성분이 포함된 화장품 성분 추천", key="analysis_chat")
    if prefill and not user_input:
        user_input = prefill
    if not user_input:
        return

    st.session_state.qa_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("성분 안전도 분석 중..."):
            try:
                data = api.chat(
                    user_input,
                    st.session_state.skin_type,
                    st.session_state.get("search_type", "hyde"),
                    st.session_state.qa_messages   # ← history 추가
                )
                answer  = data["answer"]
                sources = data.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📚 참고 문서 보기"):
                        for src in sources:
                            st.markdown(f'**📦 {src["product_name"]}**')
                            st.code(src["content"], language=None)
                st.session_state.qa_messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except api.APIError as err:
                st.error(str(err))
                st.session_state.qa_messages.append(
                    {"role": "assistant", "content": str(err)}
                )
    st.rerun()


def _render_reset_button() -> None:
    """대화 초기화 버튼"""
    if not st.session_state.qa_messages:
        return
    if st.button("🗑️ 대화 초기화", key="reset_qa"):
        sess.reset_analysis()
        st.rerun()