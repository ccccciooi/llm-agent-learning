"""OpenAI-compatible HTTP 客户端。

这个模块负责配置 HTTP 请求，不负责 Agent 循环。Agent 只调用
create_chat_completion()，因此网络层和循环层可以分别学习和测试。
"""

import json
from dataclasses import dataclass

from httpx import Client

from python.llm_agent_learning.config import LLMConfig


@dataclass
class LLMClient:
    """封装模型配置和可复用的 httpx.Client。"""

    # 每次请求时放入 body.model 的模型名称。
    model: str

    # 从 YAML 和环境变量加载的完整配置。
    config: LLMConfig

    # 底层 HTTP 客户端；复用它可以共享连接池。
    _client: Client

    # ask_once() 使用的系统提示词。
    system_prompt: str

    @property
    def client(self) -> Client:
        """暴露底层客户端，便于需要时访问 httpx 功能。"""

        return self._client

    def close(self) -> None:
        """关闭连接池；程序使用完客户端后必须调用。"""

        if self._client is not None:
            self._client.close()

    def ask_once(self, prompt: str) -> dict[str, object]:
        """执行一次不带工具的简单问答，保留原有学习示例接口。"""

        # 简单问答只需要 system 和 user 两条消息。
        body = self._build_openai_post_body(prompt)
        return self._post_chat_completion(body)

    def create_chat_completion(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """使用完整消息历史调用 Chat Completions API。

        Agent 每执行一轮都会调用这个方法。messages 中会逐渐加入
        assistant 的 tool_calls 和 Python 执行后的 tool 结果。
        """

        # OpenAI-compatible Chat Completions 的基础请求体。
        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }

        # 没有工具时不发送 tools 字段，兼容不支持工具调用的供应商。
        if tools:
            body["tools"] = tools
        return self._post_chat_completion(body)

    def _post_chat_completion(
        self,
        body: dict[str, object],
    ) -> dict[str, object]:
        """发送 HTTP POST 请求并返回已经解析的 JSON 对象。"""

        # 学习阶段打印完整输入，方便观察每一轮 messages 如何变化。
        # 请求头中的 API key 不在 body 中，所以不会被这里打印出来。
        print("=====模型调用-输入=====")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print("=====================")

        # base_url 在创建 Client 时设置，这里只传配置中的接口路径。
        response = self._client.post(
            url=self.config.base_uri,
            json=body,
        )

        # 遇到 4xx 或 5xx 时立即抛出 httpx.HTTPStatusError。
        response.raise_for_status()

        # response.json() 把 HTTP 返回的 JSON 文本转换成 Python 对象。
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("模型响应必须是 JSON 对象")
        return payload

    def _build_openai_post_body(self, prompt: str) -> dict[str, object]:
        """构造 ask_once() 使用的最小请求体。"""

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return {
            "model": self.model,
            "messages": messages,
        }

def create_llm_client(
    config: LLMConfig,
    system_prompt: str,
) -> LLMClient:
    """根据配置创建 LLMClient。

    httpx.Client 中的 base_url、headers 和 timeout 会被后续请求复用。
    """

    # authorization_header 和 authorization_prefix 都来自配置，
    # 因而也能适配不使用标准 Authorization: Bearer 的供应商。
    client = Client(
        base_url=config.base_url,
        headers={
            "Content-Type": "application/json",
            config.authorization_header: f"{config.authorization_prefix} {config.api_key}",
        },
        timeout=60.0,
    )

    # 把底层 HTTP Client 和业务配置包装成项目自己的 LLMClient。
    return LLMClient(
        model=config.model,
        config=config,
        _client=client,
        system_prompt=system_prompt,
    )
