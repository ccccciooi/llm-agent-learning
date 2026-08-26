from typing import Protocol
from symtable import Class

from memory_rag.domain.models import Memory


class EmbeddingProvider(Protocol):
    """文本向量化接口"""
    def embed(self,text:str)->list[float]:
        pass

