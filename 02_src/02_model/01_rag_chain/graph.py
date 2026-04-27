"""
================================================================================
03_graph.py - LangGraph 질문 분류 및 라우팅
================================================================================

[목적]
사용자 질문을 3가지 유형으로 자동 분류하고, 각 유형에 최적의 처리 방식 선택.
- classify_node: 질문 유형 판단 → preset_id + search_type 결정
- ingredient_node: 성분 안전성 질문 → RAG 체인 실행
- recommend_node: 피부 고민 추천 질문 → GPT 추천 답변
- general_node: 일반 대화 → GPT 답변 (화장품 무관 거절)

[평가 기반 자동 선택]
질문 유형별 최적 조합:
- ingredient: Dense (1.444) + Preset2 (EWG 0.50)
- recommend: BM25 (1.822) + Preset4 (균형 0.45/0.45)
- general: BM25 (0.926) + Preset1 (균등 0.33)

[수업 자료 참고]
08_langgraph/01_langgraph_overview.ipynb

================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from chain import get_answer

load_dotenv()


# ============================================================================
# [1단계] GraphState 정의
# ============================================================================
class GraphState(TypedDict):
    """
    [목적]
    LangGraph의 상태 저장소. 각 노드를 거치면서 업데이트됨.

    [필드 설명]
    - query: 사용자 질문
    - search_type: 검색 방식 (dense/bm25/rrf/hyde) - classify_node에서 결정
    - history: 대화 히스토리 (최근 4개) - 맥락 유지용
    - question_type: 질문 유형 (ingredient/recommend/general) - classify_node 출력
    - preset_id: FAISS 프리셋 (1~4) - classify_node 출력
    - answer: 최종 답변
    - sources: 참고 문서 목록
    """
    query: str
    search_type: str
    history: list
    question_type: str
    preset_id: int
    answer: str
    sources: list


# ============================================================================
# [2단계] 프리셋 매핑 (FAISS 가중치 설정)
# ============================================================================
PRESET_MAP = {
    """
    [프리셋 설명]
    각 질문 유형에 맞는 청크 가중치 설정.
    
    - ingredient (성분 안전성): Preset2
      EWG 0.50 (최우선), 기본정보 0.35, 전문가 0.15
      → EWG 등급을 가장 중요하게 봄
    
    - recommend (피부 고민): Preset4
      EWG 0.45, 기본정보 0.45, 전문가 0.10
      → 안전성과 기본정보 균형
    
    - general (일반 정보): Preset1
      0.33 균등 분배
      → 모든 정보 동등 취급
    """
    "ingredient": 2,
    "recommend":  4,
    "general":    1,
}


# ============================================================================
# [3단계] 검색 방식 매핑 (평가 결과 기반)
# ============================================================================
SEARCH_MAP = {
    """
    [평가 결과 기반 최적 검색 방식]
    
    - ingredient: Dense (NDCG 1.444 최고)
      → 의미 기반 검색이 "성분 안전성" 추론에 강함
      → 유사 표현 (나이아신아마이드 vs 니코틴아마이드) 인식 가능
    
    - recommend: BM25 (NDCG 1.822 최고)
      → "지성 피부", "보습", "진정" 같은 직접 키워드 검색에 강함
      → 키워드 매칭으로 정확도 높음
    
    - general: BM25 (NDCG 0.926 최고)
      → "화장품 보관", "성분표" 같은 절차 정보는 키워드 중심
      → 속도와 효율성도 중요
    """
    "ingredient": "dense",
    "recommend":  "bm25",
    "general":    "bm25",
}


# ============================================================================
# [4단계] classify_node - 질문 분류
# ============================================================================
def classify_node(state: GraphState) -> GraphState:
    """
    [함수 설명]
    사용자 질문을 3가지 유형으로 자동 분류.
    히스토리 기반 맥락 고려.

    [분류 기준]
    - ingredient: "나이아신아마이드 EWG?" "레티놀 안전해?" 같은 성분 직접 질문
    - recommend: "지성 피부에 뭐?" "주름 개선 성분?" 같은 피부 고민 질문
    - general: "화장품 보관?" "성분표 읽는 법?" 같은 일반 정보 질문

    [히스토리 활용]
    "살리실산 추천" → "국내 제품은?"
    → 히스토리 없으면 "국내 제품은?"을 general로 오분류
    → 히스토리 있으면 recommend로 정확 분류

    [출력]
    - question_type: 분류 결과
    - preset_id: PRESET_MAP에서 선택
    - search_type: SEARCH_MAP에서 선택
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # [1단계] 히스토리 기반 컨텍스트 생성
    history_text = ""
    if state.get("history"):
        recent = state["history"][-4:]
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        )

    # [2단계] 분류 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
사용자 질문을 아래 3가지 중 하나로 분류하세요.
반드시 단어 하나만 출력하세요.
이전 대화 맥락이 있으면 참고하세요.

- ingredient : 특정 성분 안전성, EWG 등급, 성분 효능, 성분 위험성 질문
- recommend  : 피부 고민, 제품 추천, 어떤 성분 써야 하는지 질문
- general    : 인사, 잡담, 위 두 가지에 해당하지 않는 질문

이전 대화:
{history}
"""),
        ("human", "{query}")
    ])

    # [3단계] LLM으로 분류
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": state["query"],
        "history": history_text
    }).strip().lower()

    # [4단계] 결과 검증 (예외 처리)
    if result not in ["ingredient", "recommend", "general"]:
        result = "ingredient"

    # [5단계] preset_id와 search_type 결정
    preset_id   = PRESET_MAP.get(result, 1)
    search_type = SEARCH_MAP.get(result, "dense")

    print(f"[DEBUG] question_type: {result}, preset_id: {preset_id}, search_type: {search_type}")

    return {**state, "question_type": result, "preset_id": preset_id, "search_type": search_type}


# ============================================================================
# [5단계] ingredient_node - 성분 안전성 분석
# ============================================================================
def ingredient_node(state: GraphState) -> GraphState:
    """
    [함수 설명]
    성분 안전성 질문 처리.
    chain.py의 get_answer() 호출 → RAG 체인 실행

    [처리 흐름]
    1. Dense 검색 (1.444 NDCG)
    2. Preset2 로드 (EWG 중심 가중치)
    3. 후보 20개 → 성분명 필터링 → Rerank → 최종 5개
    4. CoT + Structured Output 으로 답변 생성

    [출력]
    - answer: "**EWG 등급**: 1 (안전)..." 형태
    - sources: [{"ingredient": "나이아신아마이드", ...}, ...]
    """
    result = get_answer(
        query=state["query"],
        search_type=state.get("search_type", "dense"),
        history=state.get("history", []),
        preset_id=state.get("preset_id", 2)
    )
    return {**state, "answer": result["answer"], "sources": result["sources"]}


# ============================================================================
# [6단계] recommend_node - 피부 고민 추천
# ============================================================================
def recommend_node(state: GraphState) -> GraphState:
    """
    [함수 설명]
    피부 고민/추천 질문 처리.
    RAG 없이 순수 GPT 추천 답변.

    [특징]
    - 검색 결과 불필요 (sources=[])
    - EWG 안전 등급 우선 추천
    - 히스토리 기반 맥락 유지 (최근 4개)

    [예시]
    "지성 피부에 좋은 성분?"
    → "지성 피부는 과잉 피지 분비가 문제이므로,
       세라마이드(EWG 1등급)와 나이아신아마이드(EWG 1등급)를 추천합니다."
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in state.get("history", [])[-4:]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 화장품 성분 전문가입니다.
사용자의 피부 고민에 맞는 성분이나 제품을 추천해주세요.
EWG 안전 등급이 낮은(안전한) 성분을 우선 추천하세요.
이전 대화 맥락이 있으면 반드시 참고하세요.

이전 대화:
{history}
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state["query"],
        "history": history_text
    })
    return {**state, "answer": answer, "sources": []}


# ============================================================================
# [7단계] general_node - 일반 대화
# ============================================================================
def general_node(state: GraphState) -> GraphState:
    """
    [함수 설명]
    일반 정보 질문 처리.
    화장품 관련이면 답변, 무관하면 거절.

    [특징]
    - 화장품 무관 질문 거절: "배고프다", "날씨?" 등
    - 화장품 관련 일반 정보는 친절하게 답변
    - 히스토리 기반 맥락 유지

    [예시]
    "화장품 보관 방법?" → 친절하게 답변
    "배고프다" → 정중히 거절 + 화장품 질문 유도
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in state.get("history", [])[-4:]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 화장품 성분 전문 AI입니다.
화장품, 피부, 성분과 관련 없는 질문에는 정중하게 답변을 거절하고
화장품 성분 관련 질문을 유도하세요.
화장품/피부/성분 관련 일반 대화는 친절하게 답변하세요.

이전 대화:
{history}
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state["query"],
        "history": history_text
    })
    return {**state, "answer": answer, "sources": []}


# ============================================================================
# [8단계] router - 조건부 엣지
# ============================================================================
def router(state: GraphState) -> Literal["ingredient", "recommend", "general"]:
    """
    [함수 설명]
    question_type에 따라 다음 노드 결정.

    [라우팅 로직]
    ingredient → ingredient_node → END
    recommend → recommend_node → END
    general → general_node → END
    """
    return state["question_type"]


# ============================================================================
# [9단계] build_graph - LangGraph 구성
# ============================================================================
def build_graph():
    """
    [함수 설명]
    LangGraph 워크플로우 구성.

    [구조]
    START
      ↓
    classify_node (질문 분류 + 자동 최적화)
      ↓ (router)
    ├→ ingredient_node → END
    ├→ recommend_node → END
    └→ general_node → END
    """
    graph = StateGraph(GraphState)

    # 노드 추가
    graph.add_node("classify",   classify_node)
    graph.add_node("ingredient", ingredient_node)
    graph.add_node("recommend",  recommend_node)
    graph.add_node("general",    general_node)

    # 엔트리 포인트
    graph.set_entry_point("classify")

    # 조건부 엣지 (라우팅)
    graph.add_conditional_edges(
        "classify",
        router,
        {
            "ingredient": "ingredient",
            "recommend":  "recommend",
            "general":    "general",
        }
    )

    # 종료 엣지
    graph.add_edge("ingredient", END)
    graph.add_edge("recommend",  END)
    graph.add_edge("general",    END)

    return graph.compile()


# ============================================================================
# [10단계] rag_graph 생성 및 run_graph 실행 함수
# ============================================================================
rag_graph = build_graph()


def run_graph(
    query: str,
    history: list = None
) -> dict:
    """
    [함수 설명]
    LangGraph 실행.

    [처리 흐름]
    1. classify_node: 질문 분류 → preset_id + search_type 결정
    2. 조건부 라우팅: 분류 결과에 따라 노드 선택
    3. 각 노드 실행: 최적화된 방식으로 답변 생성
    4. 결과 반환

    [매개변수]
    - query: 사용자 질문
    - history: 대화 히스토리 (없으면 [] 기본값)

    [반환값]
    {
        "answer": "최종 답변",
        "sources": [참고 문서],
        "question_type": "ingredient/recommend/general",
        "preset_id": 1~4,
        "search_type": "dense/bm25/rrf/hyde"
    }
    """
    result = rag_graph.invoke({
        "query": query,
        "search_type": "dense",  # 초기값 (classify_node에서 덮어씀)
        "history": history or [],
        "question_type": "",
        "preset_id": 1,
        "answer": "",
        "sources": []
    })

    return {
        "answer":        result["answer"],
        "sources":       result["sources"],
        "question_type": result["question_type"],
        "preset_id":     result["preset_id"],
        "search_type":   result["search_type"]
    }