from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memory_rag.api.memory_routes import router as memory_router
from memory_rag.application.memory_service import MemoryService
from memory_rag.config import EmbeddingSettings, QdrantSettings
from memory_rag.infrastructure.openai_embedding_provider import OpenAIEmbeddingProvider
from memory_rag.infrastructure.qdrant_memory_repository import QdrantMemoryRepository


@asynccontextmanager
async def _configured_lifespan(app: FastAPI) -> AsyncIterator[None]:
    qdrant_settings = QdrantSettings.from_yaml()
    embedding_settings = EmbeddingSettings.from_yaml()
    if qdrant_settings.vector_size != embedding_settings.dimensions:
        raise RuntimeError("Qdrant 向量维度与 Embedding 输出维度不一致")

    repository = QdrantMemoryRepository(qdrant_settings)
    try:
        embedding_provider = OpenAIEmbeddingProvider(
            base_url=embedding_settings.base_url,
            api_key=embedding_settings.api_key,
            model=embedding_settings.model,
            dimensions=embedding_settings.dimensions,
        )
    except Exception:
        repository.close()
        raise

    app.state.memory_service = MemoryService(
        embedding_provider=embedding_provider,
        memory_repository=repository,
    )
    try:
        yield
    finally:
        embedding_provider.close()
        repository.close()


def create_app(memory_service: MemoryService | None = None) -> FastAPI:
    """创建 HTTP 应用；测试时可注入不访问外部服务的 MemoryService。"""
    if memory_service is None:
        lifespan = _configured_lifespan
    else:

        @asynccontextmanager
        async def injected_lifespan(app: FastAPI) -> AsyncIterator[None]:
            app.state.memory_service = memory_service
            yield

        lifespan = injected_lifespan

    application = FastAPI(
        title="Memory RAG API",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.include_router(memory_router)
    return application


app = create_app()
