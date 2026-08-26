from openai import BaseModel

#对外接口
#  - Schema 面向 HTTP JSON
#  - Domain Model 面向内部业务
# - API层负责两者转换

class CreateMemoryRequest(BaseModel):
    content: str
    user_id: str
    project_id: str | None = None

class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    limit: int=5
