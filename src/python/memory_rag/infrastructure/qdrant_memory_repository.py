from asyncio.windows_events import NULL
from dataclasses import dataclass

from python.memory_rag.domain.models import Memory, MemorySearchQuery, MemorySearchResult
from python.memory_rag.ports.memory_repository import MemoryRepository


@dataclass
class QdrantMemoryPayload:
    """Qdrant 中记忆向量对应的 payload。"""

    content: str
    user_id: str
    project_id: str | None
    memory_type: str


@dataclass
class QdrantMemoryPoint:
    """Qdrant 中保存的记忆向量数据点。"""

    id: str
    vector: list[float]
    payload: QdrantMemoryPayload


class QdrantMemoryRepository(MemoryRepository):
    """实现类"""

    def save(self,memory:Memory,vector:list[float]) ->None:
        return None

    def search(self,query:MemorySearchQuery,vector:list[float]) -> list[MemorySearchResult]:
        return NULL

    def delete(self,memory_id:str) ->None:
        return None

