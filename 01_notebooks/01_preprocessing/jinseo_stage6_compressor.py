"""
6단계: Contextual Compression + GPT 전달 인터페이스

- 상위 5개 청크를 GPT로 압축 → 질문 관련 핵심 문장만 추출
- 압축 결과 중 상위 3개 선별
- GPT 최종 답변용 프롬프트 생성 (시스템 메시지 + 검색 결과 + 사용자 질문)
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────────

COMPRESSION_MODEL = os.getenv("COMPRESSION_MODEL",  "gpt-4o-mini")
FINAL_ANSWER_MODEL = os.getenv("FINAL_ANSWER_MODEL",  "gpt-4o")
MIN_COMPRESSED_LEN = int(os.getenv("MIN_COMPRESSED_LEN", "30"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))


# ──────────────────────────────────────────────
# 2. 데이터 모델
# ──────────────────────────────────────────────

@dataclass
class CompressedChunk:
    """압축 완료된 청크."""
    original_content: str
    compressed_content: str
    metadata: dict[str, Any]
    final_score: float
    compression_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        orig_len = max(len(self.original_content), 1)
        self.compression_ratio = len(self.compressed_content) / orig_len

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_content": self.original_content,
            "compressed_content": self.compressed_content,
            "metadata": self.metadata,
            "final_score": self.final_score,
            "compression_ratio": self.compression_ratio,
        }


# ──────────────────────────────────────────────
# 3. 단일 청크 압축
# ──────────────────────────────────────────────

_COMPRESS_SYSTEM = textwrap.dedent("""\
    당신은 화장품 성분 정보 추출 전문가입니다.
    사용자 질문과 직접 관련된 핵심 문장만 한국어로 추출하세요.

    규칙:
    - 질문과 무관한 내용은 완전히 제거하세요.
    - 관련 내용은 원문 표현을 최대한 유지하되 간결하게 압축하세요.
    - 성분명, 안전 등급(COOS/EWG/Paula's Choice), 효능, 주의사항은 반드시 포함하세요.
    - 핵심 문장이 없으면 정확히 "IRRELEVANT" 라고만 출력하세요.
    - 부연 설명, 인사, 추가 코멘트를 절대 붙이지 마세요.
""")


def compress_single_chunk(
    client: OpenAI,
    query: str,
    chunk_content: str,
    model: str = COMPRESSION_MODEL,
) -> str:
    """
    단일 청크를 질문 기준으로 GPT를 통해 압축.
    관련 없으면 "IRRELEVANT" 반환.
    """
    user_msg = f"[질문]\n{query}\n\n[청크 내용]\n{chunk_content}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        result = response.choices[0].message.content.strip()
        logger.debug("압축 결과: %r", result[:80])
        return result
    except Exception as e:
        logger.error("압축 API 호출 실패: %s", e)
        return chunk_content  # 실패 시 원문 반환


# ──────────────────────────────────────────────
# 4. Contextual Compression 메인 함수
# ──────────────────────────────────────────────

def contextual_compress(
    client: OpenAI,
    query: str,
    ranked_chunks: list[Any],        # RankedChunk 리스트
    top_k_compress: int = 5,
    top_k_final: int = TOP_K_FINAL,
    model: str = COMPRESSION_MODEL,
    min_len: int = MIN_COMPRESSED_LEN,
) -> list[CompressedChunk]:
    """
    상위 ranked_chunks를 압축하고 최종 top_k_final 개 반환.

    Parameters
    ----------
    client : OpenAI 클라이언트
    query : 사용자 질문
    ranked_chunks : stage6_reranker.rerank() 결과 (RankedChunk 리스트)
    top_k_compress : 압축 대상 수 (기본 5)
    top_k_final : 최종 선별 수 (기본 3)
    model : 압축 GPT 모델
    min_len : 압축 후 최소 문자 수
    """
    candidates = ranked_chunks[:top_k_compress]
    logger.info("압축 대상: %d개 청크", len(candidates))

    compressed: list[CompressedChunk] = []

    for i, rc in enumerate(candidates):
        name = rc.metadata.get("ingredient_ko", "?")
        logger.info("[%d/%d] %s 청크 압축 중…", i + 1, len(candidates), name)

        result = compress_single_chunk(client, query, rc.content, model=model)

        if result == "IRRELEVANT" or len(result) < min_len:
            logger.info("  → 관련 없음, 제외 (%s)", name)
            continue

        compressed.append(
            CompressedChunk(
                original_content=rc.content,
                compressed_content=result,
                metadata=rc.metadata,
                final_score=rc.final_score,
            )
        )

    compressed.sort(key=lambda c: c.final_score, reverse=True)
    selected = compressed[:top_k_final]

    logger.info("압축 완료 → %d개 선별 (시도 %d개)", len(selected), len(candidates))
    return selected


# ──────────────────────────────────────────────
# 5. GPT 전달 인터페이스
# ──────────────────────────────────────────────

_SYSTEM_TEMPLATE = textwrap.dedent("""\
    당신은 화장품 성분 안전성 전문 AI 어시스턴트입니다.
    아래 [검색 결과]만을 근거로 사용자 질문에 답변하세요.

    답변 원칙:
    1. 검색 결과에 없는 내용은 추측하거나 생성하지 마세요.
    2. COOS 등급, EWG 등급, Paula's Choice 평가가 있으면 반드시 언급하세요.
    3. 안전성 관련 주의사항이 있으면 명확히 전달하세요.
    4. 답변은 명확하고 간결하게 한국어로 작성하세요.
    5. 검색 결과만으로 답변이 불가능하면 "제공된 정보만으로는 답변하기 어렵습니다."라고 말하세요.
""")


def build_prompt(
    query: str,
    compressed_chunks: list[CompressedChunk],
    system_template: str = _SYSTEM_TEMPLATE,
) -> list[dict[str, str]]:
    """
    압축된 청크를 GPT 메시지 형식으로 변환.

    Returns
    -------
    OpenAI Chat API messages 리스트
    """
    context_blocks: list[str] = []
    for idx, cc in enumerate(compressed_chunks, 1):
        name = cc.metadata.get("ingredient_ko", "?")
        chunk_type = cc.metadata.get("chunk_type", "?")
        coos = cc.metadata.get("coos_score", "-")
        ewg = cc.metadata.get("hw_ewg", "-")
        pc = cc.metadata.get("pc_rating", "-")
        score_str = f"{cc.final_score:.4f}"

        block = (
            f"[검색 결과 {idx}]\n"
            f"성분: {name} | 유형: {chunk_type} | "
            f"COOS: {coos} | EWG: {ewg} | PC: {pc} | 점수: {score_str}\n"
            f"{cc.compressed_content}"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)
    user_content = (
        f"[검색 결과]\n"
        f"{'─' * 40}\n"
        f"{context_str}\n"
        f"{'─' * 40}\n\n"
        f"[사용자 질문]\n{query}"
    )

    return [
        {"role": "system", "content": system_template},
        {"role": "user",   "content": user_content},
    ]


def call_final_gpt(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str = FINAL_ANSWER_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """최종 답변 GPT 호출."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("최종 GPT 답변 생성 완료 (%d자)", len(answer))
        return answer
    except Exception as e:
        logger.error("최종 GPT 호출 실패: %s", e)
        raise


# ──────────────────────────────────────────────
# 디버그 출력
# ──────────────────────────────────────────────

def print_compressed_table(chunks: list[CompressedChunk]) -> None:
    print(f"\n{'='*60}")
    print(f"  Contextual Compression 결과  ({len(chunks)}개)")
    print(f"{'='*60}")
    for i, c in enumerate(chunks, 1):
        name = c.metadata.get("ingredient_ko", "?")
        ratio_pct = c.compression_ratio * 100
        print(
            f"\n[{i}] {name} | final_score={c.final_score:.4f} | "
            f"압축률={ratio_pct:.1f}% | "
            f"COOS={c.metadata.get('coos_score','-')} | "
            f"EWG={c.metadata.get('hw_ewg','-')} | "
            f"PC={c.metadata.get('pc_rating','-')}"
        )
        print(f"  원문: {c.original_content[:80]}…")
        print(f"  압축: {c.compressed_content[:80]}…")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 동작 확인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    from jinseo_stage6_rerank import RankedChunk

    query = "나이아신아마이드가 피부에 안전한가요?"

    # 실제 minha_retriever.py 메타데이터 키 그대로 사용
    dummy_ranked = [
        RankedChunk(
            content="나이아신아마이드(Niacinamide)는 COOS 안전 등급 성분으로 EWG 1등급에 해당합니다. 미백·모공 축소 효과가 검증되어 있으며 화해에서 Good 등급입니다.",
            metadata={
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type": "summary",
                "coos_score": "안전",
                "hw_ewg": "1",
                "pc_rating": "훌륭함",
            },
            original_score=0.91, chunk_weight=1.5, domain_weight=1.5,
        ),
        RankedChunk(
            content="나이아신아마이드는 고농도(10% 이상) 사용 시 일부 민감성 피부에 홍조·따가움이 나타날 수 있습니다. 저농도(2~4%)부터 시작하는 것을 권장합니다.",
            metadata={
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type": "paragraph",
                "coos_score": "안전",
                "hw_ewg": "2",
                "pc_rating": "좋음",
            },
            original_score=0.84, chunk_weight=1.0, domain_weight=1.3,
        ),
        RankedChunk(
            content="성분: 나이아신아마이드 | EWG: 1 | COOS: 안전 | PC: 훌륭함 | 효능: 미백, 모공 축소, 피지 조절, 항산화 | 주의: 고농도 시 자극 가능",
            metadata={
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type": "table",
                "coos_score": "안전",
                "hw_ewg": "1",
                "pc_rating": "훌륭함",
            },
            original_score=0.78, chunk_weight=1.2, domain_weight=1.5,
        ),
        RankedChunk(
            content="Q: 나이아신아마이드와 비타민C 함께 사용해도 되나요? A: 일반 화장품 농도에서는 함께 사용해도 안전합니다.",
            metadata={
                "ingredient_ko": "나이아신아마이드",
                "ingredient_en": "Niacinamide",
                "chunk_type": "qa_pair",
                "coos_score": "안전",
                "hw_ewg": "1",
                "pc_rating": "좋음",
            },
            original_score=0.73, chunk_weight=1.3, domain_weight=1.4,
        ),
        RankedChunk(
            content="파라벤 계열 방부제는 EWG 4~6 등급으로 호르몬 교란 가능성이 일부 연구에서 제기됩니다.",
            metadata={
                "ingredient_ko": "파라벤",
                "ingredient_en": "Paraben",
                "chunk_type": "paragraph",
                "coos_score": "주의",
                "hw_ewg": "4",
                "pc_rating": "보통",
            },
            original_score=0.61, chunk_weight=1.0, domain_weight=0.7,
        ),
    ]

    client = OpenAI()
    compressed = contextual_compress(client, query, dummy_ranked)
    print_compressed_table(compressed)

    messages = build_prompt(query, compressed)
    print("[최종 프롬프트 미리보기]")
    for m in messages:
        print(f"\n--- {m['role'].upper()} ---")
        print(m["content"][:400] + ("…" if len(m["content"]) > 400 else ""))

    answer = call_final_gpt(client, messages)
    print("\n[최종 GPT 답변]")
    print(answer)