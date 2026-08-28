from typing import Protocol

from memory_rag.domain.models import Memory, MemorySearchQuery, MemorySearchResult


class MemoryRepository(Protocol):
    """记忆存储需要提供的能力"""

    # vector 是记忆内容的向量
    def save(self,memory:Memory,vector:list[float])->None:
        """保存记忆"""
        pass

    # vector 是查询文本的向量
    def search(self,query:MemorySearchQuery,vector:list[float])->list[MemorySearchResult]:
        """搜索记忆"""
        pass

    # memory_id：记忆ID
    def delete(self,memory_id:str)->None:
        """删除某个记忆"""
        pass
    