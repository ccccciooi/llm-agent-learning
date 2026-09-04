from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from memory_rag.domain.models import Memory

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def tokenize(text: str) -> list[str]:
    """保留英文词，对中文片段生成字符和双字词，无需外部分词器。"""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalize_text(text)):
        value = match.group(0)
        if value.isascii():
            tokens.append(value)
            continue
        characters = list(value)
        tokens.extend(characters)
        tokens.extend(
            "".join(characters[index:index + 2])
            for index in range(len(characters) - 1)
        )
    return tokens


def searchable_text(memory: Memory) -> str:
    values = [
        memory.title,
        memory.content,
        " ".join(memory.tags),
        memory.fact_key or "",
    ]
    return "\n".join(value for value in values if value)


def normalized_bm25_scores(query: str, memories: list[Memory]) -> dict[str, float]:
    """在 scope 内的有界候选集上计算 BM25，并归一化到 0..1。"""
    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not memories:
        return {memory.id: 0.0 for memory in memories}

    documents = {memory.id: tokenize(searchable_text(memory)) for memory in memories}
    average_length = sum(len(tokens) for tokens in documents.values()) / len(documents)
    if average_length == 0:
        return {memory.id: 0.0 for memory in memories}

    document_frequency = {
        term: sum(term in set(tokens) for tokens in documents.values())
        for term in query_terms
    }
    raw_scores: dict[str, float] = {}
    document_count = len(documents)
    k1 = 1.5
    b = 0.75
    for memory_id, tokens in documents.items():
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            matched_documents = document_frequency[term]
            inverse_frequency = math.log(
                1 + (document_count - matched_documents + 0.5) / (matched_documents + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        raw_scores[memory_id] = score

    maximum = max(raw_scores.values(), default=0.0)
    if maximum <= 0:
        return {memory_id: 0.0 for memory_id in raw_scores}
    return {memory_id: score / maximum for memory_id, score in raw_scores.items()}


class HeuristicMemoryReranker:
    """可替换的无外部模型 reranker，强调标题、标签和完整短语命中。"""

    def rerank(self, query: str, candidates: list[Memory]) -> dict[str, float]:
        query_text = normalize_text(query)
        query_terms = set(tokenize(query))
        scores: dict[str, float] = {}
        for memory in candidates:
            title_terms = set(tokenize(memory.title))
            tag_terms = set(tokenize(" ".join(memory.tags)))
            content_terms = set(tokenize(memory.content))
            fact_terms = set(tokenize(memory.fact_key or ""))
            denominator = max(len(query_terms), 1)
            title_coverage = len(query_terms & title_terms) / denominator
            tag_coverage = len(query_terms & (tag_terms | fact_terms)) / denominator
            content_coverage = len(query_terms & content_terms) / denominator
            phrase_match = float(
                bool(query_text) and query_text in normalize_text(searchable_text(memory))
            )
            scores[memory.id] = min(
                1.0,
                0.40 * title_coverage
                + 0.25 * tag_coverage
                + 0.25 * content_coverage
                + 0.10 * phrase_match,
            )
        return scores
