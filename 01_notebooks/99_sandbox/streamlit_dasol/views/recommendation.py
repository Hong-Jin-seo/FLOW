"""
views/recommendation.py
제품 추천 페이지 — RAG 큐레이터 스무고개 채팅
"""
import streamlit as st
from services import api
from ui import components
from state import session as sess


def render() -> None:
    """제품 추천 페이지를 렌더링한다."""
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    components.page_header(
        "✨ 맞춤형 제품 추천",
        "피부 고민을 말씨해 주세요 — AI가 최적의 제품을 찾아드립니다",
    )
    components.rag_banner()
    _render_welcome()
    _render_history()
    _render_controls()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_welcome() -> None:
    """최초 환영 메시지 (대화 없을 때)"""
    if st.session_state.rec_messages:
        return
    with st.chat_message("assistant", avatar="✨"):
        st.markdown(
            "안녕하세요\! 저는 **AI 어드바이저**입니다. \U0001f60a\n\n"
            "어떤 피부 고민이 있으신가요? "
            "피부 타입이나 찾고 있는 성분, "
            "피하고 싶은 성분을 자유롭게 이야기해주세요."
        )


def _render_history() -> None:
    """저장된 대화 + 제품 카드 표시"""
    for msg in st.session_state.rec_messages:
        avatar = "✨" if msg["role"] == "assistant" else "\U0001f464"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            for prod in msg.get("products", []):
                components.product_card(
                    prod.get("name", ""),
                    prod.get("brand", ""),
                    prod.get("tags", []),
                )


def _render_controls() -> None:
    """최종 완료 / 선택지 버튼 / 텍스트 입력 + 초기화 버튼"""
    if st.session_state.cur_is_final:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("\U0001f504  새 고민으로 첫부터 시작", use_container_width=True, key="restart_rec"):
            sess.reset_recommendation()
            st.rerun()
        return

    if st.session_state.cur_choices:
        st.markdown(
            '<div style="font-size:.875rem;font-weight:600;color:#374151;margin:14px 0 8px;">'
            '\U0001f447 하나를 선택해주세요</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(st.session_state.cur_choices))
        for i, (col, choice) in enumerate(zip(cols, st.session_state.cur_choices)):
            if col.button(choice, key=f"choice_{i}", use_container_width=True):
                with st.spinner("분석 중..."):
                    _call_curate(choice)
                st.rerun()
    else:
        concern = st.chat_input(
            "예: 요즘 마스크 때문에 입 주변이 건조하고 트러블이 나요",
            key="rec_chat",
        )
        if concern:
            with st.spinner("데이터베이스 검색 및 답변 생성 중..."):
                _call_curate(concern)
            st.rerun()

    if st.session_state.rec_messages:
        if st.button("\U0001f5d1️ 대화 초기화", key="reset_rec"):
            sess.reset_recommendation()
            st.rerun()


def _call_curate(message: str) -> None:
    """API 호출 후 세션 상태를 업데이트한다."""
    st.session_state.rec_messages.append({"role": "user", "content": message})
    try:
        data = api.curate(message, st.session_state.cur_session)
        st.session_state.rec_messages.append(
            {"role": "assistant", "content": data["message"], "products": data.get("products", [])}
        )
        st.session_state.cur_session  = data["session"]
        st.session_state.cur_choices  = data.get("choices", [])
        st.session_state.cur_is_final = data.get("is_final", False)
    except api.APIError as err:
        st.session_state.rec_messages.append({"role": "assistant", "content": str(err)})
