from memory_rag.infrastructure.openai_embedding_provider import OpenAIEmbeddingProvider
from memory_rag.util.yaml_config import get_yaml_config
from memory_rag.ports.embedding_provider import EmbeddingProvider

if __name__ == '__main__':

    provider: EmbeddingProvider = OpenAIEmbeddingProvider(
        base_url=get_yaml_config("embedding.base-url"),
        api_key=get_yaml_config("embedding.api-key"),
        model=get_yaml_config("embedding.model"),
        dimensions=int(get_yaml_config("embedding.dimensions")),
    )

    try:
        vector = provider.embed("你好，我想查询合同续签流程")
        print(f"向量维度：{len(vector)}")
        print(f"前五个值：{vector[:5]}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError()
