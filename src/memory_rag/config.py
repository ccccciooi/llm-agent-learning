from dataclasses import dataclass

from memory_rag.util.yaml_config import get_yaml_config


def _positive_int(value: str, path: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"配置项 {path} 必须是整数") from exc
    if parsed <= 0:
        raise RuntimeError(f"配置项 {path} 必须大于 0")
    return parsed


@dataclass(frozen=True)
class QdrantSettings:
    """memory-rag 使用的 Qdrant 连接和 collection 配置。"""

    url: str = "http://127.0.0.1:6333"
    api_key: str | None = None
    collection_name: str = "user_memories"
    vector_name: str = "semantic"
    vector_size: int = 1024
    timeout: int = 10

    def __post_init__(self) -> None:
        for field_name in ("url", "collection_name", "vector_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if (
            isinstance(self.vector_size, bool)
            or not isinstance(self.vector_size, int)
            or self.vector_size <= 0
        ):
            raise ValueError("vector_size 必须大于 0")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, int)
            or self.timeout <= 0
        ):
            raise ValueError("timeout 必须大于 0")

    @classmethod
    def from_yaml(cls) -> "QdrantSettings":
        api_key = get_yaml_config("memory-rag.qdrant-api-key").strip() or None
        return cls(
            url=get_yaml_config("memory-rag.qdrant-url"),
            api_key=api_key,
            collection_name=get_yaml_config("memory-rag.collection-name"),
            vector_name=get_yaml_config("memory-rag.vector-name"),
            vector_size=_positive_int(
                get_yaml_config("embedding.dimensions"),
                "embedding.dimensions",
            ),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """OpenAI 兼容 Embedding 接口配置。"""

    base_url: str
    api_key: str
    model: str
    dimensions: int = 1024

    def __post_init__(self) -> None:
        for field_name in ("base_url", "api_key", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 不能为空")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions <= 0
        ):
            raise ValueError("dimensions 必须大于 0")

    @classmethod
    def from_yaml(cls) -> "EmbeddingSettings":
        return cls(
            base_url=get_yaml_config("embedding.base-url"),
            api_key=get_yaml_config("embedding.api-key"),
            model=get_yaml_config("embedding.model"),
            dimensions=_positive_int(
                get_yaml_config("embedding.dimensions"),
                "embedding.dimensions",
            ),
        )
