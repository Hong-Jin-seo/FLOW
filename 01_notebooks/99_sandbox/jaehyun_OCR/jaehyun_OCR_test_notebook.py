# ─────────────────────────────────────────
# 터미널에서 실행:
# streamlit run jaehyun_OCR_test_notebook.py
# ─────────────────────────────────────────

import os
import re
import pickle
import numpy as np
import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import easyocr
from PIL import Image
import io
from dotenv import load_dotenv

# ─────────────────────────────────────────
# 환경변수 로드
# ─────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE  = os.path.join(BASE_DIR, "cosmetic.index")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.pkl")

# ─────────────────────────────────────────
# 캐시: 무거운 리소스는 한 번만 로드
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")


@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'], gpu=False)


@st.cache_resource
def load_index_and_chunks():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        st.error("cosmetic.index 또는 chunks.pkl 파일이 없습니다. build_index.py를 먼저 실행해주세요!")
        st.stop()

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ─────────────────────────────────────────
# RAG 검색 + LLM 답변
# ─────────────────────────────────────────
def ask(question, model, index, chunks):
    q_vec = model.encode([question], convert_to_numpy=True).astype("float32")
    _, indices = index.search(q_vec, 3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 화장품 성분 전문가입니다.\n"
                    "아래에 제공된 성분 및 제품 데이터를 기반으로만 답변하세요.\n"
                    "데이터에 없는 내용은 '해당 정보가 데이터에 없습니다'라고 솔직하게 말하세요.\n"
                    "한국어로 친절하게 답변하세요."
                ),
            },
            {
                "role": "user",
                "content": f"[참고 데이터]\n{context}\n\n[질문]\n{question}",
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────
st.set_page_config(page_title="화장품 성분 Q&A", page_icon="🧴", layout="wide")
st.title("🧴 화장품 성분 질의응답 시스템")
st.caption("coos 성분 데이터 + 화해 리뷰 데이터 기반")

# 리소스 로드
with st.spinner("모델 및 데이터 로딩 중... (최초 1회만 시간이 걸려요)"):
    model         = load_model()
    ocr_reader    = load_ocr()
    index, chunks = load_index_and_chunks()

st.success(f"로드 완료! 총 {len(chunks)}개 청크 준비됨")

# ─────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 텍스트 질문", "📷 이미지로 분석"])

# ── 탭 1: 텍스트 질문 ────────────────────
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("성분에 대해 질문하세요! 예) 나이아신아마이드 효능이 뭐야?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                answer = ask(user_input, model, index, chunks)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── 탭 2: 이미지 OCR ─────────────────────
with tab2:
    st.markdown("화장품 성분표 사진을 업로드하면 성분을 분석해드립니다!")

    uploaded = st.file_uploader("화장품 이미지 업로드", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        st.image(uploaded, caption="업로드된 이미지", use_column_width=True)

        if st.button("이미지 분석하기", type="primary"):
            # PIL 변환 없이 numpy로 바로 변환
            import numpy as np
            from PIL import Image
            import io

            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            import cv2

            image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # BGR → RGB

            with st.spinner("OCR 텍스트 추출 중..."):
                ocr_results = ocr_reader.readtext(
                    image_np,
                    detail=0,
                )
                full_text = " ".join(ocr_results)

            st.subheader("📝 OCR로 추출된 텍스트")
            st.text_area("", value=full_text, height=120, disabled=True)

            if full_text.strip():
                with st.spinner("성분 분석 중..."):
                    question = f"다음 화장품 성분들에 대해 설명해줘: {full_text}"
                    answer = ask(question, model, index, chunks)
                st.subheader("🤖 성분 분석 답변")
                st.markdown(answer)
            else:
                st.warning("텍스트를 인식하지 못했습니다.")