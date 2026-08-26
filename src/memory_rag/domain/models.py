from datetime import datetime
from dataclasses import dataclass




@dataclass
class Memory:
    """ 记忆存储对象 """
    id: str
    content: str
    user_id: str
    project_id: str | None
    memory_type: str
    importance: float
    created_time: datetime
    updated_time: datetime


@dataclass
class MemorySearchQuery:
    """记忆搜索对象"""

    text: str
    user_id: str
    project_id: str | None = None
    memory_type: str | None = None
    limit: int = 5
    score_threshold: float | None = None


@dataclass
class MemorySearchResult:
    """记忆搜索结果对象"""

    memory: Memory
    score: float
