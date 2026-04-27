"""
================================================================================
04_evaluation.py - 검색 방식 평가 및 최적화 근거
================================================================================

[목적]
4가지 검색 방식(BM25, Dense, RRF, HyDE)을 3가지 질문 유형별로 평가.
NDCG@3, Precision@3, Recall@3, MRR 4가지 지표로 측정.

[평가 결과 요약]
- ingredient: Dense 1.444 (최고) → Dense 선택
- recommend: BM25 1.822 (최고) → BM25 선택
- general: BM25 0.926 (최고) → BM25 선택

[평가 방식]
각 질문 유형별 5~10개 테스트 질문
4가지 검색 방식 모두 실행
상위 3개 결과에서 예상 키워드 포함 여부 측정

================================================================================
"""

import pandas as pd
import numpy as np
from chain import get_answer

# ============================================================================
# [1단계] 테스트 데이터셋
# ============================================================================
TEST_DATA = {
    """
    [테스트 데이터 구성]

    ingredient (성분 안전성):
    - 직접적인 성분명 포함
    - EWG 등급 관련
    - 안전성 직접 질문

    recommend (피부 고민):
    - 피부 타입 언급 (지성, 민감성)
    - 목표 기능 언급 (보습, 진정, 주름개선)

    general (일반 정보):
    - 화장품 사용법
    - 성분표 읽기
    - 보관 방법
    """
    "ingredient": [
        ("나이아신아마이드 EWG 등급 알려줘", ["나이아신아마이드", "EWG"]),
        ("레티놀이 안전한가요?", ["레티놀", "EWG"]),
        ("살리실산 부작용 있어?", ["살리실산"]),
        ("비타민씨 성분 정보", ["비타민C"]),
        ("세라마이드 안전성", ["세라마이드"]),
    ],
    "recommend": [
        ("지성 피부에 좋은 성분 추천해줘", ["지성", "보습"]),
        ("민감성 피부용 진정 성분?", ["민감성", "진정"]),
        ("여름에 쓸 가벼운 성분", ["가벼운", "수분"]),
        ("주름 개선에 효과 좋은 거", ["주름", "개선"]),
        ("피부톤 개선 성분", ["톤", "밝음"]),
    ],
    "general": [
        ("스킨케어 순서 알려줄래?", ["순서", "스킨케어"]),
        ("화장품 유통기한 어떻게 봐?", ["유통기한", "보관"]),
        ("성분표 읽는 법", ["성분", "표"]),
    ],
}


# ============================================================================
# [2단계] 평가 지표 계산
# ============================================================================
def calculate_metrics(docs, keywords, k=3):
    """
    [함수 설명]
    4가지 평가 지표 계산: Precision@3, Recall@3, MRR, NDCG@3

    [지표 설명]

    1. Precision@3: 상위 3개 중 관련 문서 비율
       - 범위: 0~1
       - 높을수록: 검색 결과가 정확함
       - 예: 3개 중 3개 관련 = 1.0 / 3개 중 1개만 관련 = 0.333

    2. Recall@3: 관련 문서를 상위 3개에서 찾은 비율
       - 범위: 0~3 (키워드 개수 = 최대 관련도)
       - 높을수록: 모든 관련 정보를 찾음
       - 예: 키워드 2개 중 2개 찾음 = 1.0

    3. MRR (Mean Reciprocal Rank): 첫 관련 문서 순위
       - 범위: 0~1
       - 1위에서 찾으면 = 1.0 / 2위에서 찾으면 = 0.5
       - 높을수록: 빨리 찾음

    4. NDCG@3: 순위 반영 적합성
       - 범위: 0~2.0
       - 높을수록: 정확한 순서로 관련 문서 나열
       - 1위 관련 × 1 + 2위 관련 × 0.5 + 3위 관련 × 0.33...
    """
    if not docs:
        return {'precision': 0, 'recall': 0, 'mrr': 0, 'ndcg': 0}

    # [1] Precision@3: 상위 3개 중 관련 문서 비율
    relevant_in_top_k = sum(1 for doc in docs[:k]
                            if any(kw in doc for kw in keywords))
    precision = relevant_in_top_k / min(k, len(docs))

    # [2] Recall@3: 관련 문서 중 상위 3개에서 찾은 비율
    total_relevant = len(keywords)
    recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0

    # [3] MRR: 첫 관련 문서 순위
    mrr = 0
    for i, doc in enumerate(docs[:k]):
        if any(kw in doc for kw in keywords):
            mrr = 1 / (i + 1)
            break

    # [4] NDCG@3: 순위 반영 적합성
    dcg = 0
    for i, doc in enumerate(docs[:k]):
        relevance = sum(1 for kw in keywords if kw in doc)
        if relevance > 0:
            dcg += relevance / (i + 1)

    idcg = sum(1 / (i + 1) for i in range(min(k, len(keywords))))
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'mrr': mrr,
        'ndcg': ndcg
    }


# ============================================================================
# [3단계] 평가 실행
# ============================================================================
def evaluate(question, method, keywords, preset_id):
    """
    [함수 설명]
    특정 검색 방식으로 질문을 평가.

    [처리 흐름]
    1. get_answer() 호출 (search_type 지정)
    2. sources에서 문서 컨텐트 추출
    3. calculate_metrics() 로 4가지 지표 계산
    4. 예외 처리 (실패 시 0점)
    """
    try:
        result = get_answer(question, search_type=method, preset_id=preset_id, history=[])
        docs = [src.get("content", "") for src in result.get("sources", [])]
        return calculate_metrics(docs, keywords, k=3)
    except Exception as e:
        print(f"    Error: {e}")
        return {'precision': 0, 'recall': 0, 'mrr': 0, 'ndcg': 0}


# ============================================================================
# [4단계] 메인 평가 루프
# ============================================================================
results = {q_type: {} for q_type in TEST_DATA}

for q_type in TEST_DATA:
    # preset_id 결정
    preset_id = 2 if q_type == "ingredient" else (4 if q_type == "recommend" else 1)

    print(f"\n{'=' * 60}")
    print(f"{q_type.upper()} 평가")
    print(f"{'=' * 60}")

    for question, keywords in TEST_DATA[q_type]:
        print(f"\n질문: {question}")

        # 4가지 검색 방식 모두 실행
        for method in ["bm25", "dense", "hyde", "rrf"]:
            metrics = evaluate(question, method, keywords, preset_id)

            if method not in results[q_type]:
                results[q_type][method] = {'precision': [], 'recall': [], 'mrr': [], 'ndcg': []}

            results[q_type][method]['precision'].append(metrics['precision'])
            results[q_type][method]['recall'].append(metrics['recall'])
            results[q_type][method]['mrr'].append(metrics['mrr'])
            results[q_type][method]['ndcg'].append(metrics['ndcg'])

            print(
                f"  {method:6s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} MRR={metrics['mrr']:.3f} NDCG={metrics['ndcg']:.3f}")

# ============================================================================
# [5단계] 최종 결과 요약
# ============================================================================
print(f"\n{'=' * 60}")
print("최종 결과 (평균 점수)")
print(f"{'=' * 60}")

for q_type in TEST_DATA:
    print(f"\n{q_type.upper()}:")
    for method in ["bm25", "dense", "hyde", "rrf"]:
        p_avg = np.mean(results[q_type][method]['precision'])
        r_avg = np.mean(results[q_type][method]['recall'])
        m_avg = np.mean(results[q_type][method]['mrr'])
        n_avg = np.mean(results[q_type][method]['ndcg'])
        print(f"  {method:6s}: P={p_avg:.3f} R={r_avg:.3f} MRR={m_avg:.3f} NDCG={n_avg:.3f}")