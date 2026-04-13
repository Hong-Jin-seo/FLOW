import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="스킨 큐레이터", page_icon="🧴", layout="centered")


# ── 큐레이터 API 호출 공통 함수 ───────────────────────────────
def _send_curate(message: str):
    st.session_state.cur_messages.append({"role": "user", "content": message})

    with st.spinner("분석 중... 잠시만 기다려주세요 ✨"):
        try:
            resp = requests.post(
                f"{API_BASE}/curate",
                json={
                    "message": message,
                    "session": st.session_state.cur_session,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()

            st.session_state.cur_messages.append(
                {"role": "assistant", "content": data["message"]}
            )
            st.session_state.cur_session  = data["session"]
            st.session_state.cur_choices  = data.get("choices", [])
            st.session_state.cur_is_final = data.get("is_final", False)

        except requests.exceptions.ConnectionError:
            st.error("❌ 서버에 연결할 수 없습니다.")
        except Exception as e:
            st.error(f"❌ 오류: {e}")

    st.rerun()

# ── 헤더 ──────────────────────────────────────────────────────
st.title("🧴 스킨 큐레이터")
st.caption("성분 Q&A 또는 내 피부에 딱 맞는 제품 큐레이션을 받아보세요.")

# ── 탭 구성 ───────────────────────────────────────────────────
tab_qa, tab_curator = st.tabs(["💬 성분 Q&A", "✨ 스킨 큐레이터"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — 일반 성분 Q&A
# ══════════════════════════════════════════════════════════════
with tab_qa:
    with st.sidebar:
        st.header("⚙️ 설정")
        skin_type = st.selectbox(
            "내 피부 타입",
            ["선택 안 함", "지성", "건성", "복합성", "민감성", "중성"],
            key="skin_type_select",
        )
        skin_type = None if skin_type == "선택 안 함" else skin_type

        st.divider()
        st.markdown("**질문 예시**")
        examples = [
            "비플레인 캐모마일 토너 성분 알려줘",
            "지성 피부에 좋은 성분 추천해줘",
            "EWG 1등급 성분만 있는 제품 있어?",
            "알레르기 유발 성분 없는 크림 추천해줘",
            "나이아신아마이드가 뭐야?",
        ]
        for ex in examples:
            if st.button(ex, key=f"qa_ex_{ex}", use_container_width=True):
                st.session_state["qa_input"] = ex

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 참고 데이터"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['product_name']}**")
                        st.code(src["content"], language=None)

    default_qa = st.session_state.pop("qa_input", "")
    qa_input   = st.chat_input("성분이나 제품에 대해 질문하세요", key="qa_chat")
    if not qa_input and default_qa:
        qa_input = default_qa

    if qa_input:
        st.session_state.qa_messages.append({"role": "user", "content": qa_input})
        with st.chat_message("user"):
            st.markdown(qa_input)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/chat",
                        json={"question": qa_input, "skin_type": skin_type},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data    = resp.json()
                    answer  = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 참고 데이터"):
                            for src in sources:
                                st.markdown(f"**{src['product_name']}**")
                                st.code(src["content"], language=None)

                    st.session_state.qa_messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except requests.exceptions.ConnectionError:
                    st.error("❌ 서버에 연결할 수 없습니다. FastAPI 서버를 먼저 실행하세요.")
                except Exception as e:
                    st.error(f"❌ 오류: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — 스킨 큐레이터 (스무고개형)
# ══════════════════════════════════════════════════════════════
with tab_curator:
    st.markdown(
        "피부 고민을 자유롭게 입력하면 **단계별 질문**을 통해 "
        "딱 맞는 제품 **1개**를 추천해드려요. ✨"
    )
    st.divider()

    # 세션 초기화
    if "cur_messages" not in st.session_state:
        st.session_state.cur_messages = []
    if "cur_session" not in st.session_state:
        st.session_state.cur_session = {}
    if "cur_choices" not in st.session_state:
        st.session_state.cur_choices = []
    if "cur_is_final" not in st.session_state:
        st.session_state.cur_is_final = False

    # 대화 히스토리 출력
    for msg in st.session_state.cur_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 처음 시작 안내
    if not st.session_state.cur_messages:
        with st.chat_message("assistant"):
            st.markdown(
                "안녕하세요! 요즘 피부 고민이 있으신가요? 😊\n\n"
                "고민을 자유롭게 입력해주시면, 몇 가지 질문을 통해 "
                "가장 잘 맞는 제품을 추천해드릴게요."
            )

    # 최종 추천 완료 → 처음으로 버튼
    if st.session_state.cur_is_final:
        if st.button("🔄 새 고민으로 처음부터 시작", use_container_width=True):
            st.session_state.cur_messages = []
            st.session_state.cur_session  = {}
            st.session_state.cur_choices  = []
            st.session_state.cur_is_final = False
            st.rerun()

    # 선택지 버튼 (이상형 월드컵 방식)
    elif st.session_state.cur_choices:
        st.markdown("**👇 하나를 선택해주세요**")
        cols = st.columns(len(st.session_state.cur_choices))
        for i, (col, choice) in enumerate(
            zip(cols, st.session_state.cur_choices)
        ):
            if col.button(choice, key=f"choice_{i}", use_container_width=True):
                _send_curate(choice)

    # 텍스트 입력 (고민 입력 단계)
    else:
        concern = st.chat_input(
            "피부 고민을 입력하세요 (예: 요즘 마스크 때문에 입 주변이 건조하고 트러블이 나요)",
            key="cur_chat",
        )
        if concern:
            _send_curate(concern)
