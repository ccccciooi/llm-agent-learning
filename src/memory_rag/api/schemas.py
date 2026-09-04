from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from memory_rag.domain.models import (
    Memory,
    MemoryMutation,
    MemoryRanking,
    MemorySearchQuery,
    MemorySearchResult,
    MemorySource,
    MemoryVersion,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiSchema(BaseModel):
    """HTTP JSON schema 的共同约束。"""

    model_config = ConfigDict(extra="forbid")


class MemorySourceSchema(ApiSchema):
    type: NonEmptyString = Field(description="来源类型，例如 diary、timeline、chat")
    ref: NonEmptyString = Field(description="可追溯的来源引用")

    def to_domain(self) -> MemorySource:
        return MemorySource(type=self.type, ref=self.ref)


class CreateMemoryRequest(ApiSchema):
    kind: NonEmptyString = Field(description="记忆类型")
    title: NonEmptyString = Field(description="记忆主题")
    content: NonEmptyString = Field(description="给模型阅读的原子记忆内容")
    tags: list[NonEmptyString] = Field(default_factory=list)
    occurred_at: AwareDatetime | None = None
    importance: int = Field(default=3, ge=1, le=5, strict=True)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: MemorySourceSchema | None = None
    related_memory_ids: list[NonEmptyString] = Field(default_factory=list)
    evidence_memory_ids: list[NonEmptyString] = Field(default_factory=list)
    fact_key: NonEmptyString | None = None
    idempotency_key: NonEmptyString | None = Field(
        default=None,
        description="来源事件的稳定唯一键，重试时不会重复写入",
    )

    def to_domain(self, *, user_id: str, environment_id: int) -> Memory:
        return Memory(
            owner_id=user_id,
            environment_id=environment_id,
            kind=self.kind,
            title=self.title,
            content=self.content,
            tags=list(self.tags),
            occurred_at=self.occurred_at,
            importance=self.importance,
            confidence=self.confidence,
            source=self.source.to_domain() if self.source is not None else None,
            related_memory_ids=list(self.related_memory_ids),
            evidence_memory_ids=list(self.evidence_memory_ids),
            fact_key=self.fact_key,
            idempotency_key=self.idempotency_key,
        )


class UpsertMemoryRequest(CreateMemoryRequest):
    fact_key: NonEmptyString = Field(description="要幂等维护的稳定事实键")
    conflict_policy: Literal["reject", "supersede"] = "reject"
    expected_revision: int | None = Field(default=None, ge=1, strict=True)


class MemoryRankingSchema(ApiSchema):
    semantic_weight: float = Field(default=0.45, ge=0)
    keyword_weight: float = Field(default=0.20, ge=0)
    rerank_weight: float = Field(default=0.15, ge=0)
    importance_weight: float = Field(default=0.10, ge=0)
    recency_weight: float = Field(default=0.10, ge=0)
    recency_half_life_days: float = Field(default=180.0, gt=0)
    candidate_multiplier: int = Field(default=4, ge=1, le=20, strict=True)
    lexical_scan_limit: int = Field(default=1000, ge=1, le=10000, strict=True)

    def to_domain(self) -> MemoryRanking:
        return MemoryRanking(**self.model_dump())

    @model_validator(mode="after")
    def validate_at_least_one_weight(self) -> Self:
        self.to_domain()
        return self


class SearchMemoryRequest(ApiSchema):
    query: NonEmptyString = Field(description="需要向量化和关键词分析的检索问题")
    kinds: list[NonEmptyString] = Field(default_factory=list)
    occurred_from: AwareDatetime | None = None
    occurred_to: AwareDatetime | None = None
    tags: list[NonEmptyString] = Field(default_factory=list)
    min_importance: int | None = Field(default=None, ge=1, le=5, strict=True)
    limit: int = Field(default=5, ge=1, le=100, strict=True)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    ranking: MemoryRankingSchema = Field(default_factory=MemoryRankingSchema)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if (
            self.occurred_from is not None
            and self.occurred_to is not None
            and self.occurred_from > self.occurred_to
        ):
            raise ValueError("occurred_from 不能晚于 occurred_to")
        return self

    def to_domain(self, *, user_id: str, environment_id: int) -> MemorySearchQuery:
        return MemorySearchQuery(
            text=self.query,
            owner_id=user_id,
            environment_id=environment_id,
            kinds=list(self.kinds),
            occurred_from=self.occurred_from,
            occurred_to=self.occurred_to,
            tags=list(self.tags),
            min_importance=self.min_importance,
            limit=self.limit,
            score_threshold=self.score_threshold,
            ranking=self.ranking.to_domain(),
        )


class LifecycleRequest(ApiSchema):
    expected_revision: int | None = Field(default=None, ge=1, strict=True)


class MemoryResponse(ApiSchema):
    id: str
    user_id: str
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
    source: MemorySourceSchema | None
    related_memory_ids: list[str]
    evidence_memory_ids: list[str]
    fact_key: str | None
    revision: int
    content_hash: str
    idempotency_key: str | None
    deleted_at: datetime | None
    history_count: int
    schema_version: int

    @classmethod
    def from_domain(cls, memory: Memory) -> "MemoryResponse":
        return cls(
            id=memory.id,
            user_id=memory.owner_id,
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
            confidence=memory.confidence,
            source=(
                MemorySourceSchema(type=memory.source.type, ref=memory.source.ref)
                if memory.source is not None
                else None
            ),
            related_memory_ids=list(memory.related_memory_ids),
            evidence_memory_ids=list(memory.evidence_memory_ids),
            fact_key=memory.fact_key,
            revision=memory.revision,
            content_hash=memory.content_hash,
            idempotency_key=memory.idempotency_key,
            deleted_at=memory.deleted_at,
            history_count=len(memory.history),
            schema_version=memory.schema_version,
        )


class MemoryVersionResponse(ApiSchema):
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
    source: MemorySourceSchema | None
    related_memory_ids: list[str]
    evidence_memory_ids: list[str]
    fact_key: str | None
    content_hash: str
    idempotency_key: str | None
    transition: str
    transitioned_at: datetime

    @classmethod
    def from_domain(cls, version: MemoryVersion) -> "MemoryVersionResponse":
        return cls(
            revision=version.revision,
            status=version.status,
            kind=version.kind,
            title=version.title,
            content=version.content,
            embedding_text=version.embedding_text,
            tags=list(version.tags),
            occurred_at=version.occurred_at,
            updated_at=version.updated_at,
            importance=version.importance,
            confidence=version.confidence,
            source=(
                MemorySourceSchema(type=version.source.type, ref=version.source.ref)
                if version.source is not None
                else None
            ),
            related_memory_ids=list(version.related_memory_ids),
            evidence_memory_ids=list(version.evidence_memory_ids),
            fact_key=version.fact_key,
            content_hash=version.content_hash,
            idempotency_key=version.idempotency_key,
            transition=version.transition,
            transitioned_at=version.transitioned_at,
        )


class MemoryHistoryResponse(ApiSchema):
    memory_id: str
    fact_key: str | None
    current_revision: int
    versions: list[MemoryVersionResponse]

    @classmethod
    def from_domain(cls, memory: Memory) -> "MemoryHistoryResponse":
        return cls(
            memory_id=memory.id,
            fact_key=memory.fact_key,
            current_revision=memory.revision,
            versions=[MemoryVersionResponse.from_domain(item) for item in memory.history],
        )


class MemoryMutationResponse(ApiSchema):
    action: str
    previous_revision: int | None
    memory: MemoryResponse

    @classmethod
    def from_domain(cls, mutation: MemoryMutation) -> "MemoryMutationResponse":
        return cls(
            action=mutation.action,
            previous_revision=mutation.previous_revision,
            memory=MemoryResponse.from_domain(mutation.memory),
        )


class MemorySearchResultResponse(ApiSchema):
    memory: MemoryResponse
    score: float
    semantic_score: float
    keyword_score: float
    rerank_score: float
    importance_score: float
    recency_score: float

    @classmethod
    def from_domain(
        cls,
        result: MemorySearchResult,
    ) -> "MemorySearchResultResponse":
        return cls(
            memory=MemoryResponse.from_domain(result.memory),
            score=result.score,
            semantic_score=result.semantic_score,
            keyword_score=result.keyword_score,
            rerank_score=result.rerank_score,
            importance_score=result.importance_score,
            recency_score=result.recency_score,
        )


class DeleteMemoryResponse(ApiSchema):
    deleted: bool
