from openai import OpenAI

from python.memory_rag.ports.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    MAX_BATCH_SIZE = 10

    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            dimensions: int = 1024
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        # 调用 OpenAI Embeddings API
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start:start + self.MAX_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
                encoding_format="float"
            )

            batch_vectors = [
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            ]

            if len(batch_vectors) != len(batch):
                raise RuntimeError("Embedding API 返回的向量数量不正确")

            if any(len(vector) != self._dimensions for vector in batch_vectors):
                raise RuntimeError("Embedding 向量维度与配置不一致")

            vectors.extend(batch_vectors)

        return vectors

    def close(self) -> None:
        self._client.close()
        return None

