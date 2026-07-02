from dataclasses import dataclass
from httpx import Client
from llm_agent_learning.config import LLMConfig
import json



@dataclass
class LLMClient:
    model: str
    config: LLMConfig
    _client: Client
    system_prompt: str

    @property
    def client(self) -> Client:
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def ask_once(self,prompt: str,) -> str:
        #构造请求体
        body = self._build_openai_post_body(prompt)
        #构造调用
        response = self._client.post(
            url=self.config.base_uri,
            json=body
        )
        return response.json()

    '''
    构造请求体
    '''
    def _build_openai_post_body(self,prompt: str) -> dict[str, object]:
        model=self.config.model

        messages=[]
        sys_msg={}
        sys_msg["role"]="system"
        sys_msg["content"]=self.system_prompt
        messages.append(sys_msg)

        user_msg={}
        user_msg["role"]="user"
        user_msg["content"]=prompt
        messages.append(user_msg)

        body={}
        body["model"]=model
        body["messages"]=messages
        return body


"""
获取一个配置好了大模型的参数的LLMClient
"""
def create_llm_client(config: LLMConfig,system_prompt:str) -> LLMClient:

    client=Client(
        base_url=config.base_url,
        headers={
            "Content-Type": "application/json",
            config.authorization_header: f"{config.authorization_prefix} {config.api_key}",
        },
        timeout=60.0
    )
    return LLMClient(model=config.model, config=config, _client=client,system_prompt=system_prompt)




