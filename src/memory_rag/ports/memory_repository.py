from typing import Protocol

from memory_rag.domain.models import Memory, MemorySearchQuery, MemorySearchResult


class MemoryRepository(Protocol):
    """记忆存储需要提供的能力"""

    # vector 是记忆内容的向量
    def save(self, memory: Memory, vector: list[float]) -> None:
        """保存记忆"""
        ...

    # vector 是查询文本的向量
    def search(
        self,
        query: MemorySearchQuery,
        vector: list[float],
    ) -> list[MemorySearchResult]:
        """搜索记忆"""
        ...

    def list_active(
        self,
        query: MemorySearchQuery,
        *,
        limit: int,
    ) -> list[Memory]:
        """按同一 scope/filter 扫描活跃记忆，供关键词候选召回。"""
        ...

    def get(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        """读取当前用户在指定环境中的一条有效记忆。"""
        ...

    def get_any(
        self,
        memory_id: str,
        owner_id: str,
        environment_id: int,
    ) -> Memory | None:
        """读取任意生命周期状态的记忆。"""
        ...

    def find_by_fact_key(
        self,
        owner_id: str,
        environment_id: int,
        fact_key: str,
    ) -> Memory | None:
        """查找 scope 内的稳定事实 head。"""
        ...

    def find_by_idempotency_key(
        self,
        owner_id: str,
        environment_id: int,
        idempotency_key: str,
    ) -> Memory | None:
        """查找已处理的来源事件。"""
        ...

    def update_payload(self, memory: Memory) -> None:
        """原地更新状态和历史，保留已有向量。"""
        ...

    def delete(self, memory_id: str, owner_id: str, environment_id: int) -> bool:
        """按用户和环境双重隔离条件永久删除记忆。"""
        ...
