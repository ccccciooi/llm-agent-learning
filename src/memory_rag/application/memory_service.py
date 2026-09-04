from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from threading import RLock
from uuid import UUID, uuid5

from memory_rag.application.ranking import (
    HeuristicMemoryReranker,
    normalized_bm25_scores,
)
from memory_rag.domain.errors import (
    MemoryConflictError,
    MemoryIdempotencyConflictError,
    MemoryRevisionConflictError,
    MemoryStateError,
)
from memory_rag.domain.models import (
    CONFLICT_POLICIES,
    Memory,
    MemoryMutation,
    MemorySearchQuery,
    MemorySearchResult,
    MemoryVersion,
    utc_now,
)
from memory_rag.ports.embedding_provider import EmbeddingProvider
from memory_rag.ports.memory_repository import MemoryRepository
from memory_rag.ports.memory_reranker import MemoryReranker

FACT_MEMORY_NAMESPACE = UUID("24079509-239e-49ad-886a-83ca719d301a")


@dataclass
class MemoryService:
    embedding_provider: EmbeddingProvider
    memory_repository: MemoryRepository
    reranker: MemoryReranker = field(default_factory=HeuristicMemoryReranker)
    now_provider: Callable[[], datetime] = utc_now
    _mutation_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def add_memory(self, memory: Memory) -> Memory:
        """兼容旧接口；fact_key 存在时仍执行幂等和冲突检查。"""
        return self.upsert_memory(memory).memory

    def upsert_memory(
        self,
        memory: Memory,
        *,
        conflict_policy: str = "reject",
        expected_revision: int | None = None,
    ) -> MemoryMutation:
        """创建稳定事实，或在显式授权和版本匹配时替代旧版本。"""
        if conflict_policy not in CONFLICT_POLICIES:
            raise ValueError("conflict_policy 必须是 reject 或 supersede")
        if expected_revision is not None and expected_revision <= 0:
            raise ValueError("expected_revision 必须大于 0")

        with self._mutation_lock:
            normalized = self._normalize_candidate(memory)
            prepared = self._prepare_content(normalized)
            if normalized.idempotency_key is not None:
                processed = self.memory_repository.find_by_idempotency_key(
                    normalized.owner_id,
                    normalized.environment_id,
                    normalized.idempotency_key,
                )
                if processed is not None:
                    comparable_processed = (
                        processed if processed.content_hash else self._prepare_content(processed)
                    )
                    if prepared.content_hash != comparable_processed.content_hash:
                        raise MemoryIdempotencyConflictError(
                            "同一 idempotency_key 已用于不同的记忆内容",
                            comparable_processed,
                        )
                    if not processed.content_hash:
                        self.memory_repository.update_payload(comparable_processed)
                    return MemoryMutation(
                        action="unchanged",
                        memory=comparable_processed,
                    )
            existing = self._find_existing(prepared)
            if existing is not None:
                prepared = replace(prepared, id=existing.id)

            if existing is None:
                if expected_revision is not None:
                    raise MemoryRevisionConflictError(expected_revision, 0)
                now = self._now()
                created = replace(
                    prepared,
                    status="active",
                    revision=1,
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                    history=[],
                    schema_version=2,
                )
                vector = self.embedding_provider.embed(created.embedding_text)
                self.memory_repository.save(created, vector)
                return MemoryMutation(action="created", memory=created)

            existing_was_legacy = not existing.content_hash
            comparable_existing = (
                existing if existing.content_hash else self._prepare_content(existing)
            )
            existing = comparable_existing
            if existing.status == "deleted":
                raise MemoryStateError("该 fact_key 已删除；请先恢复，再替代内容")
            if existing.status != "active":
                raise MemoryStateError(f"状态 {existing.status} 不允许替代")
            if prepared.content_hash == comparable_existing.content_hash:
                if existing_was_legacy:
                    self.memory_repository.update_payload(comparable_existing)
                return MemoryMutation(action="unchanged", memory=comparable_existing)
            if conflict_policy == "reject":
                raise MemoryConflictError(
                    "同一 fact_key 已存在不同的活跃记忆",
                    existing,
                )
            self._require_revision(existing, expected_revision, require=True)

            now = self._now()
            replacement = replace(
                prepared,
                id=existing.id,
                status="active",
                revision=existing.revision + 1,
                created_at=existing.created_at,
                updated_at=now,
                deleted_at=None,
                history=[
                    *existing.history,
                    self._snapshot(existing, transition="superseded", at=now),
                ],
                schema_version=2,
            )
            vector = self.embedding_provider.embed(replacement.embedding_text)
            self.memory_repository.save(replacement, vector)
            return MemoryMutation(
                action="superseded",
                memory=replacement,
                previous_revision=existing.revision,
            )

    def recall(self, query: MemorySearchQuery) -> list[MemorySearchResult]:
        """组合 dense、BM25、启发式 rerank、重要性和时间衰减。"""
        candidate_limit = min(
            query.ranking.lexical_scan_limit,
            max(query.limit, query.limit * query.ranking.candidate_multiplier, 20),
        )
        dense_results: list[MemorySearchResult] = []
        if query.ranking.semantic_weight > 0:
            vector = self.embedding_provider.embed(query.text)
            dense_query = replace(
                query,
                limit=candidate_limit,
                score_threshold=None,
            )
            dense_results = self.memory_repository.search(dense_query, vector)

        lexical_memories = self.memory_repository.list_active(
            query,
            limit=query.ranking.lexical_scan_limit,
        )
        candidates = {memory.id: memory for memory in lexical_memories}
        candidates.update({result.memory.id: result.memory for result in dense_results})
        if not candidates:
            return []

        semantic_scores = {
            result.memory.id: max(0.0, min(1.0, float(result.score)))
            for result in dense_results
        }
        keyword_scores = normalized_bm25_scores(query.text, lexical_memories)
        recency_scores = {
            memory.id: self._recency_score(memory, query.ranking.recency_half_life_days)
            for memory in candidates.values()
        }
        importance_scores = {
            memory.id: memory.importance / 5.0 for memory in candidates.values()
        }

        preliminary = sorted(
            candidates.values(),
            key=lambda memory: (
                query.ranking.semantic_weight * semantic_scores.get(memory.id, 0.0)
                + query.ranking.keyword_weight * keyword_scores.get(memory.id, 0.0)
                + query.ranking.importance_weight * importance_scores[memory.id]
                + query.ranking.recency_weight * recency_scores[memory.id],
                memory.updated_at.timestamp(),
            ),
            reverse=True,
        )[:candidate_limit]
        rerank_scores = self.reranker.rerank(query.text, preliminary)

        results: list[MemorySearchResult] = []
        for memory in preliminary:
            semantic = semantic_scores.get(memory.id, 0.0)
            keyword = keyword_scores.get(memory.id, 0.0)
            rerank = max(0.0, min(1.0, rerank_scores.get(memory.id, 0.0)))
            importance = importance_scores[memory.id]
            recency = recency_scores[memory.id]
            score = (
                query.ranking.semantic_weight * semantic
                + query.ranking.keyword_weight * keyword
                + query.ranking.rerank_weight * rerank
                + query.ranking.importance_weight * importance
                + query.ranking.recency_weight * recency
            ) / query.ranking.total_weight
            if query.score_threshold is not None and score < query.score_threshold:
                continue
            results.append(MemorySearchResult(
                memory=memory,
                score=score,
                semantic_score=semantic,
                keyword_score=keyword,
                rerank_score=rerank,
                importance_score=importance,
                recency_score=recency,
            ))

        results.sort(
            key=lambda result: (
                result.score,
                result.memory.updated_at.timestamp(),
            ),
            reverse=True,
        )
        return results[:query.limit]

    def get_memory(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        """读取当前用户在指定环境中的一条活跃记忆。"""
        return self.memory_repository.get(memory_id, owner_id, environment_id)

    def get_memory_any_status(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        return self.memory_repository.get_any(memory_id, owner_id, environment_id)

    def get_memory_by_fact_key(
        self,
        fact_key: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        return self.memory_repository.find_by_fact_key(
            owner_id,
            environment_id,
            self.normalize_fact_key(fact_key),
        )

    def soft_delete_memory(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
        *,
        expected_revision: int | None = None,
    ) -> MemoryMutation | None:
        with self._mutation_lock:
            memory = self.memory_repository.get_any(memory_id, owner_id, environment_id)
            if memory is None:
                return None
            if not memory.content_hash:
                memory = self._prepare_content(memory)
            self._require_revision(memory, expected_revision)
            if memory.status == "deleted":
                return MemoryMutation(action="unchanged", memory=memory)
            if memory.status != "active":
                raise MemoryStateError(f"状态 {memory.status} 不允许删除")
            now = self._now()
            deleted = replace(
                memory,
                status="deleted",
                revision=memory.revision + 1,
                updated_at=now,
                deleted_at=now,
                history=[
                    *memory.history,
                    self._snapshot(memory, transition="deleted", at=now),
                ],
                schema_version=2,
            )
            self.memory_repository.update_payload(deleted)
            return MemoryMutation(
                action="deleted",
                memory=deleted,
                previous_revision=memory.revision,
            )

    def restore_memory(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
        *,
        expected_revision: int | None = None,
    ) -> MemoryMutation | None:
        with self._mutation_lock:
            memory = self.memory_repository.get_any(memory_id, owner_id, environment_id)
            if memory is None:
                return None
            if not memory.content_hash:
                memory = self._prepare_content(memory)
            self._require_revision(memory, expected_revision)
            if memory.status == "active":
                return MemoryMutation(action="unchanged", memory=memory)
            if memory.status != "deleted":
                raise MemoryStateError(f"状态 {memory.status} 不允许恢复")
            now = self._now()
            restored = replace(
                memory,
                status="active",
                revision=memory.revision + 1,
                updated_at=now,
                deleted_at=None,
                history=[
                    *memory.history,
                    self._snapshot(memory, transition="restored", at=now),
                ],
                schema_version=2,
            )
            self.memory_repository.update_payload(restored)
            return MemoryMutation(
                action="restored",
                memory=restored,
                previous_revision=memory.revision,
            )

    def forget_memory(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> bool:
        """永久删除记忆；常规生命周期应优先使用 soft_delete_memory。"""
        with self._mutation_lock:
            return self.memory_repository.delete(memory_id, owner_id, environment_id)

    @staticmethod
    def build_embedding_text(memory: Memory) -> str:
        """用固定模板生成实际参与向量化的文本。"""
        lines = [
            f"类型：{memory.kind}",
            f"主题：{memory.title}",
            f"记忆：{memory.content}",
        ]
        if memory.tags:
            lines.append(f"标签：{'、'.join(memory.tags)}")
        return "\n".join(lines)

    @staticmethod
    def normalize_fact_key(fact_key: str) -> str:
        normalized = fact_key.strip().lower()
        if not normalized:
            raise ValueError("fact_key 不能为空")
        return normalized

    def _find_existing(self, memory: Memory) -> Memory | None:
        if memory.fact_key is None:
            return None
        deterministic_id = self._fact_memory_id(memory)
        existing = self.memory_repository.get_any(
            deterministic_id,
            memory.owner_id,
            memory.environment_id,
        )
        if existing is not None:
            return existing
        return self.memory_repository.find_by_fact_key(
            memory.owner_id,
            memory.environment_id,
            memory.fact_key,
        )

    def _normalize_candidate(self, memory: Memory) -> Memory:
        fact_key = (
            self.normalize_fact_key(memory.fact_key)
            if memory.fact_key is not None
            else None
        )
        idempotency_key = (
            memory.idempotency_key.strip()
            if memory.idempotency_key is not None
            else None
        )
        normalized = replace(
            memory,
            owner_id=memory.owner_id.strip(),
            kind=memory.kind.strip(),
            title=memory.title.strip(),
            content=memory.content.strip(),
            tags=list(dict.fromkeys(tag.strip() for tag in memory.tags)),
            fact_key=fact_key,
            idempotency_key=idempotency_key,
        )
        if normalized.fact_key is not None:
            normalized = replace(normalized, id=self._fact_memory_id(normalized))
        return normalized

    def _prepare_content(self, memory: Memory) -> Memory:
        embedded = replace(memory, embedding_text=self.build_embedding_text(memory))
        return replace(embedded, content_hash=self._content_hash(embedded))

    @staticmethod
    def _content_hash(memory: Memory) -> str:
        source = (
            {"type": memory.source.type, "ref": memory.source.ref}
            if memory.source is not None
            else None
        )
        canonical = {
            "kind": memory.kind,
            "title": memory.title,
            "content": memory.content,
            "tags": sorted(memory.tags),
            "occurred_at": (
                memory.occurred_at.isoformat() if memory.occurred_at is not None else None
            ),
            "importance": memory.importance,
            "confidence": float(memory.confidence),
            "source": source,
            "related_memory_ids": sorted(memory.related_memory_ids),
            "evidence_memory_ids": sorted(memory.evidence_memory_ids),
            "fact_key": memory.fact_key,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _fact_memory_id(memory: Memory) -> str:
        if memory.fact_key is None:
            return memory.id
        identity = (
            f"{memory.owner_id}\x1f{memory.environment_id}\x1f{memory.fact_key}"
        )
        return str(uuid5(FACT_MEMORY_NAMESPACE, identity))

    @staticmethod
    def _snapshot(memory: Memory, *, transition: str, at: datetime) -> MemoryVersion:
        return MemoryVersion(
            revision=memory.revision,
            status=memory.status,
            kind=memory.kind,
            title=memory.title,
            content=memory.content,
            embedding_text=memory.embedding_text,
            tags=list(memory.tags),
            occurred_at=memory.occurred_at,
            updated_at=memory.updated_at,
            importance=memory.importance,
            confidence=float(memory.confidence),
            source=memory.source,
            related_memory_ids=list(memory.related_memory_ids),
            evidence_memory_ids=list(memory.evidence_memory_ids),
            fact_key=memory.fact_key,
            content_hash=memory.content_hash,
            idempotency_key=memory.idempotency_key,
            transition=transition,
            transitioned_at=at,
        )

    @staticmethod
    def _require_revision(
        memory: Memory,
        expected_revision: int | None,
        *,
        require: bool = False,
    ) -> None:
        if require and expected_revision is None:
            raise MemoryRevisionConflictError(None, memory.revision)
        if expected_revision is not None and expected_revision != memory.revision:
            raise MemoryRevisionConflictError(expected_revision, memory.revision)

    def _recency_score(self, memory: Memory, half_life_days: float) -> float:
        reference = memory.occurred_at or memory.updated_at
        age_seconds = max(0.0, (self._now() - reference).total_seconds())
        return math.pow(0.5, age_seconds / 86400.0 / half_life_days)

    def _now(self) -> datetime:
        value = self.now_provider()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("now_provider 必须返回带时区的 datetime")
        return value
