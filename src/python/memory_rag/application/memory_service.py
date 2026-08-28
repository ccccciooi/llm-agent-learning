from dataclasses import dataclass

from python.memory_rag.domain.models import Memory, MemorySearchQuery, MemorySearchResult
from python.memory_rag.ports.embedding_provider import EmbeddingProvider
from python.memory_rag.ports.memory_repository import MemoryRepository

@dataclass
class MemoryService:
    embedding_provider: EmbeddingProvider
    memory_repository: MemoryRepository

    def add_memory(self,memory:Memory) -> None:
        """添加记忆"""
        vector=self.embedding_provider.embed(memory.content)
        return self.memory_repository.save(memory, vector)

    def recall(
            self,
            query: MemorySearchQuery,
    ) -> list[MemorySearchResult]:
        """ 召回 """
        vector = self.embedding_provider.embed(query.text)
        return self.memory_repository.search(query, vector)