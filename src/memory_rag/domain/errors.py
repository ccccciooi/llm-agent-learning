from dataclasses import dataclass

from memory_rag.domain.models import Memory


@dataclass
class MemoryConflictError(RuntimeError):
    """同一 fact_key 有不同内容，且调用方没有授权替代。"""

    message: str
    current: Memory

    def __str__(self) -> str:
        return self.message


@dataclass
class MemoryRevisionConflictError(RuntimeError):
    """乐观锁版本不匹配。"""

    expected_revision: int | None
    actual_revision: int

    def __str__(self) -> str:
        return (
            f"记忆版本冲突: expected={self.expected_revision}, "
            f"actual={self.actual_revision}"
        )


@dataclass
class MemoryIdempotencyConflictError(RuntimeError):
    """同一个幂等键被用于不同的请求内容。"""

    message: str
    current: Memory

    def __str__(self) -> str:
        return self.message


class MemoryStateError(RuntimeError):
    """当前状态不允许请求的生命周期转换。"""
