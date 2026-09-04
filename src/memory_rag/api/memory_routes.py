from collections.abc import Callable
from typing import Annotated, TypeVar, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from memory_rag.api.schemas import (
    CreateMemoryRequest,
    DeleteMemoryResponse,
    LifecycleRequest,
    MemoryHistoryResponse,
    MemoryMutationResponse,
    MemoryResponse,
    MemorySearchResultResponse,
    SearchMemoryRequest,
    UpsertMemoryRequest,
)
from memory_rag.application.memory_service import MemoryService
from memory_rag.domain.errors import (
    MemoryConflictError,
    MemoryIdempotencyConflictError,
    MemoryRevisionConflictError,
    MemoryStateError,
)

router = APIRouter(tags=["memory-rag"])

UserIdPath = Annotated[
    str,
    Path(min_length=1, description="全局唯一的用户 ID 字符串"),
]
EnvironmentIdPath = Annotated[
    int,
    Path(gt=0, description="该用户下唯一的环境编号"),
]
MemoryIdPath = Annotated[UUID, Path(description="记忆 UUID")]
FactKeyQuery = Annotated[str, Query(min_length=1, description="稳定事实键")]
T = TypeVar("T")


def get_memory_service(request: Request) -> MemoryService:
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MemoryService 尚未初始化",
        )
    return cast(MemoryService, service)


MemoryServiceDependency = Annotated[MemoryService, Depends(get_memory_service)]


def _normalized_user_id(user_id: str) -> str:
    normalized = user_id.strip()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id 不能为空",
        )
    return normalized


def _run_mutation(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except MemoryConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "type": "fact_key_conflict",
                "message": str(error),
                "current": MemoryResponse.from_domain(error.current).model_dump(
                    mode="json"
                ),
            },
        ) from error
    except MemoryRevisionConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "type": "revision_conflict",
                "message": str(error),
                "expected_revision": error.expected_revision,
                "actual_revision": error.actual_revision,
            },
        ) from error
    except MemoryIdempotencyConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "type": "idempotency_key_conflict",
                "message": str(error),
                "current": MemoryResponse.from_domain(error.current).model_dump(
                    mode="json"
                ),
            },
        ) from error
    except MemoryStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"type": "state_conflict", "message": str(error)},
        ) from error


@router.post(
    "/users/{user_id}/environments/{environment_id}/memories",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="memory_save",
)
def memory_save(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    body: CreateMemoryRequest,
    service: MemoryServiceDependency,
) -> MemoryResponse:
    memory = body.to_domain(
        user_id=_normalized_user_id(user_id),
        environment_id=environment_id,
    )
    saved = _run_mutation(lambda: service.add_memory(memory))
    return MemoryResponse.from_domain(saved)


@router.post(
    "/users/{user_id}/environments/{environment_id}/memories/upsert",
    response_model=MemoryMutationResponse,
    operation_id="memory_upsert",
)
def memory_upsert(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    body: UpsertMemoryRequest,
    service: MemoryServiceDependency,
) -> MemoryMutationResponse:
    memory = body.to_domain(
        user_id=_normalized_user_id(user_id),
        environment_id=environment_id,
    )
    mutation = _run_mutation(lambda: service.upsert_memory(
        memory,
        conflict_policy=body.conflict_policy,
        expected_revision=body.expected_revision,
    ))
    return MemoryMutationResponse.from_domain(mutation)


@router.post(
    "/users/{user_id}/environments/{environment_id}/memories/search",
    response_model=list[MemorySearchResultResponse],
    operation_id="memory_search",
)
def memory_search(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    body: SearchMemoryRequest,
    service: MemoryServiceDependency,
) -> list[MemorySearchResultResponse]:
    query = body.to_domain(
        user_id=_normalized_user_id(user_id),
        environment_id=environment_id,
    )
    return [
        MemorySearchResultResponse.from_domain(result)
        for result in service.recall(query)
    ]


@router.get(
    "/users/{user_id}/environments/{environment_id}/memories/by-fact-key",
    response_model=MemoryResponse,
    operation_id="memory_get_by_fact_key",
)
def memory_get_by_fact_key(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    fact_key: FactKeyQuery,
    service: MemoryServiceDependency,
) -> MemoryResponse:
    memory = service.get_memory_by_fact_key(
        fact_key,
        _normalized_user_id(user_id),
        environment_id,
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return MemoryResponse.from_domain(memory)


@router.get(
    "/users/{user_id}/environments/{environment_id}/memories/{memory_id}/history",
    response_model=MemoryHistoryResponse,
    operation_id="memory_history",
)
def memory_history(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    memory_id: MemoryIdPath,
    service: MemoryServiceDependency,
) -> MemoryHistoryResponse:
    memory = service.get_memory_any_status(
        str(memory_id),
        _normalized_user_id(user_id),
        environment_id,
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return MemoryHistoryResponse.from_domain(memory)


@router.post(
    "/users/{user_id}/environments/{environment_id}/memories/{memory_id}/soft-delete",
    response_model=MemoryMutationResponse,
    operation_id="memory_soft_delete",
)
def memory_soft_delete(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    memory_id: MemoryIdPath,
    body: LifecycleRequest,
    service: MemoryServiceDependency,
) -> MemoryMutationResponse:
    mutation = _run_mutation(lambda: service.soft_delete_memory(
        str(memory_id),
        _normalized_user_id(user_id),
        environment_id,
        expected_revision=body.expected_revision,
    ))
    if mutation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return MemoryMutationResponse.from_domain(mutation)


@router.post(
    "/users/{user_id}/environments/{environment_id}/memories/{memory_id}/restore",
    response_model=MemoryMutationResponse,
    operation_id="memory_restore",
)
def memory_restore(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    memory_id: MemoryIdPath,
    body: LifecycleRequest,
    service: MemoryServiceDependency,
) -> MemoryMutationResponse:
    mutation = _run_mutation(lambda: service.restore_memory(
        str(memory_id),
        _normalized_user_id(user_id),
        environment_id,
        expected_revision=body.expected_revision,
    ))
    if mutation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return MemoryMutationResponse.from_domain(mutation)


@router.get(
    "/users/{user_id}/environments/{environment_id}/memories/{memory_id}",
    response_model=MemoryResponse,
    operation_id="memory_get",
)
def memory_get(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    memory_id: MemoryIdPath,
    service: MemoryServiceDependency,
) -> MemoryResponse:
    memory = service.get_memory(
        str(memory_id),
        _normalized_user_id(user_id),
        environment_id,
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return MemoryResponse.from_domain(memory)


@router.delete(
    "/users/{user_id}/environments/{environment_id}/memories/{memory_id}",
    response_model=DeleteMemoryResponse,
    operation_id="memory_forget",
)
def memory_forget(
    user_id: UserIdPath,
    environment_id: EnvironmentIdPath,
    memory_id: MemoryIdPath,
    service: MemoryServiceDependency,
) -> DeleteMemoryResponse:
    deleted = service.forget_memory(
        str(memory_id),
        _normalized_user_id(user_id),
        environment_id,
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return DeleteMemoryResponse(deleted=True)
