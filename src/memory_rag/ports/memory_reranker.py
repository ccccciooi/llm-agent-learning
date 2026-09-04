from typing import Protocol

from memory_rag.domain.models import Memory


class MemoryReranker(Protocol):
    """对首轮召回的小候选集进行二次排序。"""

    def rerank(self, query: str, candidates: list[Memory]) -> dict[str, float]: ...
