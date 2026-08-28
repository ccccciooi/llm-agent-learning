from typing import Protocol


class EmbeddingProvider(Protocol):
    """文本向量化接口"""
    def embed(self,text:str)->list[float]:
        pass


    def close(self) -> None:
        pass
