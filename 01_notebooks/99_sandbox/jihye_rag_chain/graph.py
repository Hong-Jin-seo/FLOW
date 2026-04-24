"""
jihye_rag_chain/graph.py
수업 자료 참고: 08_langgraph/01_langgraph_overview.ipynb
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


class GraphState(TypedDict):
    query: str
    search_type: str
    history: list
    question_type: str
    preset_id: int
    answer: str
    sources: list


PRESET_MAP = {
    "ingredient": 2,
    "recommend":  4,
    "general":    1,
}

SEARCH_MAP = {
    "ingredient": "hyde",
    "recommend":  "dense",
    "general":    "dense",
}


def classify_node(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    history_text = ""
    if state.get("history"):
        recent = state["history"][-4:]
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        )

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

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": state["query"],
        "history": history_text
    }).strip().lower()

    if result not in ["ingredient", "recommend", "general"]:
        result = "ingredient"

    preset_id   = PRESET_MAP.get(result, 1)
    search_type = SEARCH_MAP.get(result, "dense")

    print(f"[DEBUG] question_type: {result}, preset_id: {preset_id}, search_type: {search_type}")

    return {**state, "question_type": result, "preset_id": preset_id, "search_type": search_type}


def ingredient_node(state: GraphState) -> GraphState:
    result = get_answer(
        query=state["query"],
        search_type=state.get("search_type", "hyde"),
        history=state.get("history", []),
        preset_id=state.get("preset_id", 2)
    )
    return {**state, "answer": result["answer"], "sources": result["sources"]}


def recommend_node(state: GraphState) -> GraphState:
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


def general_node(state: GraphState) -> GraphState:
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


def router(state: GraphState) -> Literal["ingredient", "recommend", "general"]:
    return state["question_type"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify",   classify_node)
    graph.add_node("ingredient", ingredient_node)
    graph.add_node("recommend",  recommend_node)
    graph.add_node("general",    general_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        router,
        {
            "ingredient": "ingredient",
            "recommend":  "recommend",
            "general":    "general",
        }
    )
    graph.add_edge("ingredient", END)
    graph.add_edge("recommend",  END)
    graph.add_edge("general",    END)
    return graph.compile()


rag_graph = build_graph()


def run_graph(
    query: str,
    history: list = None
) -> dict:
    result = rag_graph.invoke({
        "query": query,
        "search_type": "dense",
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