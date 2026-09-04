from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

MEMORY_STATUSES = frozenset({"active", "superseded", "deleted"})
MEMORY_TRANSITIONS = frozenset({"superseded", "deleted", "restored"})
MEMORY_MUTATION_ACTIONS = frozenset({"created", "unchanged", "superseded", "deleted", "restored"})
CONFLICT_POLICIES = frozenset({"reject", "supersede"})


def utc_now() -> datetime:
    """返回带时区的 UTC 时间，便于稳定写入 Qdrant datetime payload。"""
    return datetime.now(UTC)


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")


def _require_optional_non_empty(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_non_empty(value, field_name)


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是大于 0 的整数")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} 必须包含时区")


def _require_string_list(values: list[str], field_name: str) -> None:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{field_name} 只能包含非空字符串")


@dataclass(frozen=True)
class MemorySource:
    """记忆对应的原始证据位置。"""

    type: str
    ref: str

    def __post_init__(self) -> None:
        _require_non_empty(self.type, "source.type")
        _require_non_empty(self.ref, "source.ref")


@dataclass(frozen=True)
class MemoryVersion:
    """一次状态转换前的不可变记忆快照。"""

    revision: int
    status: str
    kind: str
    title: str
    content: str
    embedding_text: str
    tags: list[str]
    occurred_at: datetime | None
    updated_at: datetime
    importance: int
    confidence: float
    source: MemorySource | None
    related_memory_ids: list[str]
    evidence_memory_ids: list[str]
    fact_key: str | None
    content_hash: str
    idempotency_key: str | None
    transition: str
    transitioned_at: datetime

    def __post_init__(self) -> None:
        _require_positive_int(self.revision, "history.revision")
        if self.status not in MEMORY_STATUSES:
            raise ValueError("历史版本 status 不合法")
        if self.transition not in MEMORY_TRANSITIONS:
            raise ValueError("历史版本 transition 不合法")
        for value, name in (
            (self.kind, "history.kind"),
            (self.title, "history.title"),
            (self.content, "history.content"),
            (self.embedding_text, "history.embedding_text"),
            (self.content_hash, "history.content_hash"),
        ):
            _require_non_empty(value, name)
        _require_optional_non_empty(self.fact_key, "history.fact_key")
        _require_optional_non_empty(self.idempotency_key, "history.idempotency_key")
        _require_string_list(self.tags, "history.tags")
        _require_string_list(self.related_memory_ids, "history.related_memory_ids")
        _require_string_list(self.evidence_memory_ids, "history.evidence_memory_ids")
        if self.occurred_at is not None:
            _require_aware_datetime(self.occurred_at, "history.occurred_at")
        _require_aware_datetime(self.updated_at, "history.updated_at")
        _require_aware_datetime(self.transitioned_at, "history.transitioned_at")


@dataclass
class Memory:
    """一张可独立检索、可追溯且可版本化的记忆卡。"""

    owner_id: str
    environment_id: int
    kind: str
    title: str
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "active"
    embedding_text: str = ""
    tags: list[str] = field(default_factory=list)
    occurred_at: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    importance: int = 3
    confidence: float = 1.0
    source: MemorySource | None = None
    related_memory_ids: list[str] = field(default_factory=list)
    evidence_memory_ids: list[str] = field(default_factory=list)
    fact_key: str | None = None
    revision: int = 1
    content_hash: str = ""
    idempotency_key: str | None = None
    deleted_at: datetime | None = None
    history: list[MemoryVersion] = field(default_factory=list)
    schema_version: int = 2

    def __post_init__(self) -> None:
        _require_non_empty(self.owner_id, "owner_id")
        _require_positive_int(self.environment_id, "environment_id")
        _require_non_empty(self.kind, "kind")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.content, "content")
        _require_non_empty(self.status, "status")
        if not isinstance(self.embedding_text, str):
            raise ValueError("embedding_text 必须是字符串")

        try:
            UUID(self.id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("id 必须是合法 UUID") from exc

        if self.status not in MEMORY_STATUSES:
            allowed = ", ".join(sorted(MEMORY_STATUSES))
            raise ValueError(f"status 必须是以下值之一: {allowed}")
        if (
            isinstance(self.importance, bool)
            or not isinstance(self.importance, int)
            or not 1 <= self.importance <= 5
        ):
            raise ValueError("importance 必须是 1 到 5 的整数")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence 必须在 0 到 1 之间")
        _require_positive_int(self.revision, "revision")
        _require_positive_int(self.schema_version, "schema_version")

        _require_string_list(self.tags, "tags")
        _require_string_list(self.related_memory_ids, "related_memory_ids")
        _require_string_list(self.evidence_memory_ids, "evidence_memory_ids")
        if self.source is not None and not isinstance(self.source, MemorySource):
            raise ValueError("source 必须是 MemorySource 或 None")
        _require_optional_non_empty(self.fact_key, "fact_key")
        _require_optional_non_empty(self.idempotency_key, "idempotency_key")
        if self.content_hash:
            _require_non_empty(self.content_hash, "content_hash")
        if not isinstance(self.history, list) or any(
            not isinstance(item, MemoryVersion) for item in self.history
        ):
            raise ValueError("history 必须是 MemoryVersion 数组")

        if self.occurred_at is not None:
            _require_aware_datetime(self.occurred_at, "occurred_at")
        _require_aware_datetime(self.created_at, "created_at")
        _require_aware_datetime(self.updated_at, "updated_at")
        if self.deleted_at is not None:
            _require_aware_datetime(self.deleted_at, "deleted_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不能早于 created_at")


@dataclass(frozen=True)
class MemoryRanking:
    """混合检索和二次排序的可解释权重。"""

    semantic_weight: float = 0.45
    keyword_weight: float = 0.20
    rerank_weight: float = 0.15
    importance_weight: float = 0.10
    recency_weight: float = 0.10
    recency_half_life_days: float = 180.0
    candidate_multiplier: int = 4
    lexical_scan_limit: int = 1000

    def __post_init__(self) -> None:
        weights = (
            self.semantic_weight,
            self.keyword_weight,
            self.rerank_weight,
            self.importance_weight,
            self.recency_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in weights
        ):
            raise ValueError("排序权重必须是非负数")
        if sum(weights) <= 0:
            raise ValueError("至少一个排序权重必须大于 0")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days 必须大于 0")
        if not 1 <= self.candidate_multiplier <= 20:
            raise ValueError("candidate_multiplier 必须在 1 到 20 之间")
        if not 1 <= self.lexical_scan_limit <= 10000:
            raise ValueError("lexical_scan_limit 必须在 1 到 10000 之间")

    @property
    def total_weight(self) -> float:
        return (
            self.semantic_weight
            + self.keyword_weight
            + self.rerank_weight
            + self.importance_weight
            + self.recency_weight
        )


@dataclass
class MemorySearchQuery:
    """带强制用户隔离、payload 条件和排序策略的语义检索。"""

    text: str
    owner_id: str
    environment_id: int
    kinds: list[str] = field(default_factory=list)
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None
    tags: list[str] = field(default_factory=list)
    min_importance: int | None = None
    limit: int = 5
    score_threshold: float | None = None
    ranking: MemoryRanking = field(default_factory=MemoryRanking)

    def __post_init__(self) -> None:
        _require_non_empty(self.text, "text")
        _require_non_empty(self.owner_id, "owner_id")
        _require_positive_int(self.environment_id, "environment_id")
        _require_string_list(self.kinds, "kinds")
        _require_string_list(self.tags, "tags")

        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= 100
        ):
            raise ValueError("limit 必须在 1 到 100 之间")
        if (
            self.score_threshold is not None
            and (
                isinstance(self.score_threshold, bool)
                or not isinstance(self.score_threshold, (int, float))
                or not 0 <= self.score_threshold <= 1
            )
        ):
            raise ValueError("混合 score_threshold 必须在 0 到 1 之间")
        if (
            self.min_importance is not None
            and (
                isinstance(self.min_importance, bool)
                or not isinstance(self.min_importance, int)
                or not 1 <= self.min_importance <= 5
            )
        ):
            raise ValueError("min_importance 必须在 1 到 5 之间")
        if not isinstance(self.ranking, MemoryRanking):
            raise ValueError("ranking 必须是 MemoryRanking")

        if self.occurred_from is not None:
            _require_aware_datetime(self.occurred_from, "occurred_from")
        if self.occurred_to is not None:
            _require_aware_datetime(self.occurred_to, "occurred_to")
        if (
            self.occurred_from is not None
            and self.occurred_to is not None
            and self.occurred_from > self.occurred_to
        ):
            raise ValueError("occurred_from 不能晚于 occurred_to")


@dataclass
class MemorySearchResult:
    """记忆搜索结果及各排序分量。"""

    memory: Memory
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0


@dataclass(frozen=True)
class MemoryMutation:
    """幂等写入或状态转换的结果。"""

    action: str
    memory: Memory
    previous_revision: int | None = None

    def __post_init__(self) -> None:
        if self.action not in MEMORY_MUTATION_ACTIONS:
            raise ValueError("记忆变更 action 不合法")
