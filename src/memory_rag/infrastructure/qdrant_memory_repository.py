import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from qdrant_client import QdrantClient, models

from memory_rag.config import QdrantSettings
from memory_rag.domain.models import (
    Memory,
    MemorySearchQuery,
    MemorySearchResult,
    MemorySource,
    MemoryVersion,
)
from memory_rag.ports.memory_repository import MemoryRepository


@dataclass(frozen=True)
class QdrantMemoryPayload:
    """Qdrant 中记忆向量对应的完整 payload。"""

    schema_version: int
    owner_id: str
    environment_id: int
    kind: str
    status: str
    title: str
    content: str
    embedding_text: str
    tags: list[str]
    occurred_at: datetime | None
    created_at: datetime
    updated_at: datetime
    importance: int
    confidence: float
    source: MemorySource | None
    related_memory_ids: list[str]
    evidence_memory_ids: list[str]
    fact_key: str | None
    revision: int
    content_hash: str
    idempotency_key: str | None
    deleted_at: datetime | None
    history: list[MemoryVersion]

    @classmethod
    def from_memory(cls, memory: Memory) -> "QdrantMemoryPayload":
        return cls(
            schema_version=memory.schema_version,
            owner_id=memory.owner_id,
            environment_id=memory.environment_id,
            kind=memory.kind,
            status=memory.status,
            title=memory.title,
            content=memory.content,
            embedding_text=memory.embedding_text,
            tags=list(memory.tags),
            occurred_at=memory.occurred_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            importance=memory.importance,
            confidence=float(memory.confidence),
            source=memory.source,
            related_memory_ids=list(memory.related_memory_ids),
            evidence_memory_ids=list(memory.evidence_memory_ids),
            fact_key=memory.fact_key,
            revision=memory.revision,
            content_hash=memory.content_hash,
            idempotency_key=memory.idempotency_key,
            deleted_at=memory.deleted_at,
            history=list(memory.history),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QdrantMemoryPayload":
        source_data = payload.get("source")
        if source_data is not None and not isinstance(source_data, Mapping):
            raise ValueError("Qdrant payload 中的 source 必须是对象")
        source = (
            MemorySource(
                type=_required_string(source_data, "type"),
                ref=_required_string(source_data, "ref"),
            )
            if source_data is not None
            else None
        )

        return cls(
            schema_version=_required_int(payload, "schema_version"),
            owner_id=_required_string(payload, "owner_id"),
            environment_id=_required_int(payload, "environment_id"),
            kind=_required_string(payload, "kind"),
            status=_required_string(payload, "status"),
            title=_required_string(payload, "title"),
            content=_required_string(payload, "content"),
            embedding_text=_required_string(payload, "embedding_text"),
            tags=_string_list(payload, "tags"),
            occurred_at=_payload_datetime(payload.get("occurred_at"), "occurred_at"),
            created_at=_required_datetime(payload, "created_at"),
            updated_at=_required_datetime(payload, "updated_at"),
            importance=_required_int(payload, "importance"),
            confidence=_required_float(payload, "confidence"),
            source=source,
            related_memory_ids=_string_list(payload, "related_memory_ids"),
            evidence_memory_ids=_string_list(payload, "evidence_memory_ids"),
            fact_key=_optional_string(payload, "fact_key"),
            revision=_optional_positive_int(payload, "revision", default=1),
            content_hash=_optional_string(payload, "content_hash") or "",
            idempotency_key=_optional_string(payload, "idempotency_key"),
            deleted_at=_payload_datetime(payload.get("deleted_at"), "deleted_at"),
            history=_history_list(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "environment_id": self.environment_id,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "importance": self.importance,
            "confidence": self.confidence,
            "related_memory_ids": list(self.related_memory_ids),
            "evidence_memory_ids": list(self.evidence_memory_ids),
            "revision": self.revision,
            "history": [_version_to_dict(version) for version in self.history],
        }
        if self.occurred_at is not None:
            payload["occurred_at"] = self.occurred_at.isoformat()
        if self.source is not None:
            payload["source"] = {
                "type": self.source.type,
                "ref": self.source.ref,
            }
        if self.fact_key is not None:
            payload["fact_key"] = self.fact_key
        if self.content_hash:
            payload["content_hash"] = self.content_hash
        if self.idempotency_key is not None:
            payload["idempotency_key"] = self.idempotency_key
        if self.deleted_at is not None:
            payload["deleted_at"] = self.deleted_at.isoformat()
        return payload

    def to_memory(self, memory_id: str) -> Memory:
        return Memory(
            id=memory_id,
            owner_id=self.owner_id,
            environment_id=self.environment_id,
            kind=self.kind,
            status=self.status,
            title=self.title,
            content=self.content,
            embedding_text=self.embedding_text,
            tags=list(self.tags),
            occurred_at=self.occurred_at,
            created_at=self.created_at,
            updated_at=self.updated_at,
            importance=self.importance,
            confidence=self.confidence,
            source=self.source,
            related_memory_ids=list(self.related_memory_ids),
            evidence_memory_ids=list(self.evidence_memory_ids),
            fact_key=self.fact_key,
            revision=self.revision,
            content_hash=self.content_hash,
            idempotency_key=self.idempotency_key,
            deleted_at=self.deleted_at,
            history=list(self.history),
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class QdrantMemoryPoint:
    """Qdrant 中保存的记忆向量数据点。"""

    id: str
    vector: list[float]
    payload: QdrantMemoryPayload


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Qdrant payload 缺少有效字段: {key}")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Qdrant payload 字段 {key} 必须是非空字符串或 null")
    return value


def _optional_positive_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Qdrant payload 字段 {key} 必须是正整数")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Qdrant payload 字段 {key} 必须是整数")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Qdrant payload 字段 {key} 必须是数字")
    return float(value)


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"Qdrant payload 字段 {key} 必须是字符串数组")
    return list(value)


def _required_datetime(payload: Mapping[str, Any], key: str) -> datetime:
    value = _payload_datetime(payload.get(key), key)
    if value is None:
        raise ValueError(f"Qdrant payload 缺少有效字段: {key}")
    return value


def _payload_datetime(value: Any, key: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Qdrant payload 字段 {key} 不是合法时间") from exc
    else:
        raise ValueError(f"Qdrant payload 字段 {key} 不是合法时间")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Qdrant payload 字段 {key} 必须包含时区")
    return parsed


def _source_from_payload(
    value: Any,
    *,
    key: str = "source",
) -> MemorySource | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Qdrant payload 中的 {key} 必须是对象")
    return MemorySource(
        type=_required_string(value, "type"),
        ref=_required_string(value, "ref"),
    )


def _history_list(payload: Mapping[str, Any]) -> list[MemoryVersion]:
    value = payload.get("history", [])
    if not isinstance(value, list):
        raise ValueError("Qdrant payload 字段 history 必须是数组")
    history: list[MemoryVersion] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Qdrant history[{index}] 必须是对象")
        history.append(MemoryVersion(
            revision=_optional_positive_int(item, "revision", default=1),
            status=_required_string(item, "status"),
            kind=_required_string(item, "kind"),
            title=_required_string(item, "title"),
            content=_required_string(item, "content"),
            embedding_text=_required_string(item, "embedding_text"),
            tags=_string_list(item, "tags"),
            occurred_at=_payload_datetime(
                item.get("occurred_at"),
                f"history[{index}].occurred_at",
            ),
            updated_at=_required_datetime(item, "updated_at"),
            importance=_required_int(item, "importance"),
            confidence=_required_float(item, "confidence"),
            source=_source_from_payload(
                item.get("source"),
                key=f"history[{index}].source",
            ),
            related_memory_ids=_string_list(item, "related_memory_ids"),
            evidence_memory_ids=_string_list(item, "evidence_memory_ids"),
            fact_key=_optional_string(item, "fact_key"),
            content_hash=_required_string(item, "content_hash"),
            idempotency_key=_optional_string(item, "idempotency_key"),
            transition=_required_string(item, "transition"),
            transitioned_at=_required_datetime(item, "transitioned_at"),
        ))
    return history


def _version_to_dict(version: MemoryVersion) -> dict[str, Any]:
    value: dict[str, Any] = {
        "revision": version.revision,
        "status": version.status,
        "kind": version.kind,
        "title": version.title,
        "content": version.content,
        "embedding_text": version.embedding_text,
        "tags": list(version.tags),
        "updated_at": version.updated_at.isoformat(),
        "importance": version.importance,
        "confidence": version.confidence,
        "related_memory_ids": list(version.related_memory_ids),
        "evidence_memory_ids": list(version.evidence_memory_ids),
        "content_hash": version.content_hash,
        "transition": version.transition,
        "transitioned_at": version.transitioned_at.isoformat(),
    }
    if version.occurred_at is not None:
        value["occurred_at"] = version.occurred_at.isoformat()
    if version.source is not None:
        value["source"] = {
            "type": version.source.type,
            "ref": version.source.ref,
        }
    if version.fact_key is not None:
        value["fact_key"] = version.fact_key
    if version.idempotency_key is not None:
        value["idempotency_key"] = version.idempotency_key
    return value


class QdrantMemoryRepository(MemoryRepository):
    """基于 Qdrant 的记忆存储实现。"""

    PAYLOAD_INDEXES: ClassVar[dict[str, models.PayloadSchemaType]] = {
        "owner_id": models.PayloadSchemaType.KEYWORD,
        "environment_id": models.PayloadSchemaType.INTEGER,
        "status": models.PayloadSchemaType.KEYWORD,
        "kind": models.PayloadSchemaType.KEYWORD,
        "fact_key": models.PayloadSchemaType.KEYWORD,
        "idempotency_key": models.PayloadSchemaType.KEYWORD,
        "tags": models.PayloadSchemaType.KEYWORD,
        "occurred_at": models.PayloadSchemaType.DATETIME,
        "importance": models.PayloadSchemaType.INTEGER,
    }

    def __init__(
        self,
        settings: QdrantSettings | None = None,
        *,
        client: QdrantClient | None = None,
        ensure_payload_indexes: bool = True,
    ) -> None:
        self._settings = settings or QdrantSettings.from_yaml()
        self._owns_client = client is None
        self._client = client or QdrantClient(
            url=self._settings.url,
            api_key=self._settings.api_key,
            timeout=self._settings.timeout,
        )
        try:
            self.ensure_collection(ensure_payload_indexes=ensure_payload_indexes)
        except Exception:
            if self._owns_client:
                self._client.close()
            raise

    def ensure_collection(self, *, ensure_payload_indexes: bool = True) -> None:
        """创建 collection，或验证已有 collection 与配置一致。"""
        if not self._client.collection_exists(self._settings.collection_name):
            self._client.create_collection(
                collection_name=self._settings.collection_name,
                vectors_config={
                    self._settings.vector_name: models.VectorParams(
                        size=self._settings.vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )
        else:
            self._validate_collection()

        if ensure_payload_indexes:
            self._ensure_payload_indexes()

    def _validate_collection(self) -> None:
        collection = self._client.get_collection(self._settings.collection_name)
        vectors = collection.config.params.vectors
        if not isinstance(vectors, Mapping):
            raise RuntimeError(
                f"collection {self._settings.collection_name} 未使用命名向量"
            )
        vector_params = vectors.get(self._settings.vector_name)
        if vector_params is None:
            raise RuntimeError(
                f"collection {self._settings.collection_name} 缺少命名向量 "
                f"{self._settings.vector_name}"
            )
        if vector_params.size != self._settings.vector_size:
            raise RuntimeError(
                f"collection 向量维度为 {vector_params.size}，"
                f"配置要求 {self._settings.vector_size}"
            )
        if vector_params.distance != models.Distance.COSINE:
            raise RuntimeError("collection 距离类型必须是 Cosine")

    def _ensure_payload_indexes(self) -> None:
        collection = self._client.get_collection(self._settings.collection_name)
        existing_indexes = collection.payload_schema or {}
        for field_name, field_schema in self.PAYLOAD_INDEXES.items():
            existing = existing_indexes.get(field_name)
            if existing is not None:
                if existing.data_type != field_schema:
                    raise RuntimeError(
                        f"payload 索引 {field_name} 类型为 {existing.data_type}，"
                        f"配置要求 {field_schema}"
                    )
                continue
            self._client.create_payload_index(
                collection_name=self._settings.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def save(self, memory: Memory, vector: list[float]) -> None:
        """以 UUID point ID 保存或更新一张记忆卡。"""
        memory_id = self._validate_memory_id(memory.id)
        checked_vector = self._validate_vector(vector)
        if not memory.embedding_text.strip():
            raise ValueError("embedding_text 不能为空，请通过 MemoryService 保存记忆")
        self._assert_point_scope(
            memory_id,
            memory.owner_id,
            memory.environment_id,
        )

        point = QdrantMemoryPoint(
            id=memory_id,
            vector=checked_vector,
            payload=QdrantMemoryPayload.from_memory(memory),
        )
        self._client.upsert(
            collection_name=self._settings.collection_name,
            points=[
                models.PointStruct(
                    id=point.id,
                    vector={self._settings.vector_name: point.vector},
                    payload=point.payload.to_dict(),
                )
            ],
            wait=True,
        )

    def search(
        self,
        query: MemorySearchQuery,
        vector: list[float],
    ) -> list[MemorySearchResult]:
        """只召回当前 owner 的 active 记忆，并应用可选过滤条件。"""
        checked_vector = self._validate_vector(vector)
        response = self._client.query_points(
            collection_name=self._settings.collection_name,
            query=checked_vector,
            using=self._settings.vector_name,
            query_filter=self._active_filter(query),
            limit=query.limit,
            score_threshold=query.score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        results: list[MemorySearchResult] = []
        for point in response.points:
            if point.payload is None:
                raise ValueError(f"Qdrant point {point.id} 缺少 payload")
            payload = QdrantMemoryPayload.from_dict(point.payload)
            results.append(
                MemorySearchResult(
                    memory=payload.to_memory(str(point.id)),
                    score=float(point.score),
                )
            )
        return results

    def list_active(
        self,
        query: MemorySearchQuery,
        *,
        limit: int,
    ) -> list[Memory]:
        """有界扫描 scope 内活跃记忆，用于本地 BM25 候选召回。"""
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        memories: list[Memory] = []
        offset: int | str | UUID | None = None
        while len(memories) < limit:
            records, offset = self._client.scroll(
                collection_name=self._settings.collection_name,
                scroll_filter=self._active_filter(query),
                limit=min(256, limit - len(memories)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload is None:
                    raise ValueError(f"Qdrant point {record.id} 缺少 payload")
                memories.append(
                    QdrantMemoryPayload.from_dict(record.payload).to_memory(
                        str(record.id)
                    )
                )
            if offset is None or not records:
                break
        return memories

    def get(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        """按 point ID 读取记忆，并强制校验用户、环境及 active 状态。"""
        return self._get_for_scope(
            memory_id,
            owner_id,
            environment_id,
            active_only=True,
        )

    def get_any(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        return self._get_for_scope(
            memory_id,
            owner_id,
            environment_id,
            active_only=False,
        )

    def find_by_fact_key(
        self,
        owner_id: str,
        environment_id: int,
        fact_key: str,
    ) -> Memory | None:
        return self._find_one_by_field(
            owner_id,
            environment_id,
            field_name="fact_key",
            value=fact_key,
        )

    def find_by_idempotency_key(
        self,
        owner_id: str,
        environment_id: int,
        idempotency_key: str,
    ) -> Memory | None:
        return self._find_one_by_field(
            owner_id,
            environment_id,
            field_name="idempotency_key",
            value=idempotency_key,
        )

    def update_payload(self, memory: Memory) -> None:
        """只更新完整 payload，用于不需要重新向量化的状态转换。"""
        memory_id = self._validate_memory_id(memory.id)
        self._assert_point_scope(
            memory_id,
            memory.owner_id,
            memory.environment_id,
        )
        if self.get_any(memory_id, memory.owner_id, memory.environment_id) is None:
            raise ValueError(f"记忆不存在: {memory_id}")
        self._client.overwrite_payload(
            collection_name=self._settings.collection_name,
            payload=QdrantMemoryPayload.from_memory(memory).to_dict(),
            points=[memory_id],
            wait=True,
        )

    def delete(self, memory_id: str, owner_id: str, environment_id: int) -> bool:
        """按 point ID、用户和环境条件永久删除记忆索引。"""
        normalized_id = self._validate_memory_id(memory_id)
        if (
            self._get_for_scope(
                normalized_id,
                owner_id,
                environment_id,
                active_only=False,
            )
            is None
        ):
            return False

        self._client.delete(
            collection_name=self._settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.HasIdCondition(has_id=[normalized_id]),
                        models.FieldCondition(
                            key="owner_id",
                            match=models.MatchValue(value=owner_id),
                        ),
                        models.FieldCondition(
                            key="environment_id",
                            match=models.MatchValue(value=environment_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )
        return True

    @staticmethod
    def _active_filter(query: MemorySearchQuery) -> models.Filter:
        must: list[Any] = [
            models.FieldCondition(
                key="owner_id",
                match=models.MatchValue(value=query.owner_id),
            ),
            models.FieldCondition(
                key="environment_id",
                match=models.MatchValue(value=query.environment_id),
            ),
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="active"),
            ),
        ]
        if query.kinds:
            must.append(
                models.FieldCondition(
                    key="kind",
                    match=models.MatchAny(any=query.kinds),
                )
            )
        if query.tags:
            must.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=query.tags),
                )
            )
        if query.occurred_from is not None or query.occurred_to is not None:
            must.append(
                models.FieldCondition(
                    key="occurred_at",
                    range=models.DatetimeRange(
                        gte=query.occurred_from,
                        lte=query.occurred_to,
                    ),
                )
            )
        if query.min_importance is not None:
            must.append(
                models.FieldCondition(
                    key="importance",
                    range=models.Range(gte=query.min_importance),
                )
            )
        return models.Filter(must=must)

    def _find_one_by_field(
        self,
        owner_id: str,
        environment_id: int,
        *,
        field_name: str,
        value: str,
    ) -> Memory | None:
        self._validate_scope(owner_id, environment_id)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")
        records, _ = self._client.scroll(
            collection_name=self._settings.collection_name,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=owner_id),
                ),
                models.FieldCondition(
                    key="environment_id",
                    match=models.MatchValue(value=environment_id),
                ),
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchValue(value=value),
                ),
            ]),
            limit=2,
            with_payload=True,
            with_vectors=False,
        )
        if len(records) > 1:
            raise RuntimeError(
                f"scope 内存在多条相同 {field_name}={value} 的记忆"
            )
        if not records:
            return None
        record = records[0]
        if record.payload is None:
            raise ValueError(f"Qdrant point {record.id} 缺少 payload")
        return QdrantMemoryPayload.from_dict(record.payload).to_memory(str(record.id))

    def _get_for_scope(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
        *,
        active_only: bool,
    ) -> Memory | None:
        self._validate_scope(owner_id, environment_id)
        normalized_id = self._validate_memory_id(memory_id)
        records = self._client.retrieve(
            collection_name=self._settings.collection_name,
            ids=[normalized_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        record = records[0]
        if record.payload is None:
            raise ValueError(f"Qdrant point {record.id} 缺少 payload")
        payload = QdrantMemoryPayload.from_dict(record.payload)
        if payload.owner_id != owner_id:
            return None
        if payload.environment_id != environment_id:
            return None
        if active_only and payload.status != "active":
            return None
        return payload.to_memory(str(record.id))

    def _assert_point_scope(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> None:
        """阻止其他用户或环境通过复用 point UUID 覆盖已有记忆。"""
        records = self._client.retrieve(
            collection_name=self._settings.collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return
        payload = records[0].payload
        if payload is None:
            raise ValueError(f"Qdrant point {records[0].id} 缺少 payload")
        stored_owner_id = _required_string(payload, "owner_id")
        stored_environment_id = _required_int(payload, "environment_id")
        if (
            stored_owner_id != owner_id
            or stored_environment_id != environment_id
        ):
            raise PermissionError("不能覆盖其他用户或环境的记忆")

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self._settings.vector_size:
            raise ValueError(
                f"向量维度为 {len(vector)}，配置要求 {self._settings.vector_size}"
            )
        checked: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("向量只能包含有限数字")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("向量只能包含有限数字")
            checked.append(number)
        return checked

    @staticmethod
    def _validate_memory_id(memory_id: str) -> str:
        try:
            return str(UUID(memory_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("memory_id 必须是合法 UUID") from exc

    @staticmethod
    def _validate_scope(owner_id: str, environment_id: int) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id 不能为空")
        if (
            isinstance(environment_id, bool)
            or not isinstance(environment_id, int)
            or environment_id <= 0
        ):
            raise ValueError("environment_id 必须是大于 0 的整数")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QdrantMemoryRepository":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
