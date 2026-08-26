"""最小 Agent 的核心实现。

这个模块只关注一条最重要的执行链：

1. 把用户消息和工具定义发送给 LLM。
2. 如果 LLM 要求调用工具，就在 Python 中执行对应函数。
3. 把工具结果放回消息列表，再次请求 LLM。
4. 直到 LLM 返回普通文本，或者达到最大调用轮数。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# JSON 能表达的所有基础类型。
# 这里使用递归类型，是因为列表和字典中还可以继续包含 JSON 值。
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

# 每个工具本质上都是一个 Python 函数：
# 输入是 LLM 生成的参数字典，输出必须是可以编码成 JSON 的值。
ToolExecutor = Callable[[dict[str, JsonValue]], JsonValue]


class ChatCompletionClient(Protocol):
    """Agent 所需要的最小客户端接口。

    Protocol 类似 Java 的 interface。只要一个对象实现了下面的方法，
    就可以传给 MinimalAgent，不要求它必须继承某个具体类。
    """

    def create_chat_completion(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class AgentTool:
    """描述一个可以提供给 LLM 调用的 Python 工具。

    dataclass 会自动生成 __init__ 等基础方法。
    frozen=True 表示对象创建后不能再修改字段，避免运行时意外改变工具定义。
    """

    # LLM 发起 tool call 时使用的唯一名称。
    name: str

    # 给 LLM 阅读的工具功能说明。
    description: str

    # 使用 JSON Schema 描述工具允许接收哪些参数。
    parameters: dict[str, JsonValue]

    # 真正执行工具工作的 Python 函数。
    execute: ToolExecutor

    def to_openai_schema(self) -> dict[str, object]:
        """转换成 OpenAI-compatible API 要求的 tools 数组元素。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MinimalAgent:
    """同步、非流式的最小 Agent。

    一个 MinimalAgent 保存客户端、系统提示词和工具注册表。
    run() 每次创建一段新的独立对话，不保留上一次 run() 的历史。
    """

    def __init__(
        self,
        client: ChatCompletionClient,
        system_prompt: str,
        tools: list[AgentTool],
        max_turns: int = 8,
    ) -> None:
        """初始化 Agent。

        Args:
            client: 用来请求 LLM 的客户端。
            system_prompt: 放在 messages 第一项的系统提示词。
            tools: 允许 LLM 调用的工具列表。
            max_turns: 一次 run() 最多调用 LLM 的次数。
        """

        # 至少要允许调用一次 LLM，否则 Agent 永远无法产生结果。
        if max_turns < 1:
            raise ValueError("max_turns 必须大于 0")

        self._client = client
        self._system_prompt = system_prompt

        # 把列表转换成 name -> AgentTool 的字典。
        # 后面收到工具名时，可以直接通过名称找到工具。
        self._tools = {tool.name: tool for tool in tools}

        # 字典中重复的 key 会被覆盖，所以通过长度变化检测重复工具名。
        if len(self._tools) != len(tools):
            raise ValueError("工具名称不能重复")

        # API 只需要工具的名称、描述和参数，不需要 Python execute 函数。
        self._tool_schemas = [tool.to_openai_schema() for tool in tools]
        self._max_turns = max_turns

    def run(self, prompt: str) -> str:
        """执行一次完整的 LLM -> tools -> LLM 循环。

        Args:
            prompt: 用户本次提出的问题。

        Returns:
            LLM 最终返回的文本内容。

        Raises:
            RuntimeError: LLM 响应格式不正确，或超过最大调用轮数。
        """

        # OpenAI-compatible API 使用 messages 数组表示整段对话。
        # 每次调用 LLM 时都要把完整历史重新发送过去。
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # 每循环一次，就代表调用一次 LLM。
        for _ in range(self._max_turns):
            response = self._client.create_chat_completion(
                messages=messages,
                tools=self._tool_schemas,
            )

            # 从完整 HTTP 响应中取出 choices[0].message。
            assistant_message = self._read_assistant_message(response)

            # 必须保留 assistant 消息，因为下一次请求要让 LLM 知道
            # 是它自己发起了这些 tool calls。
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls")

            # 没有 tool_calls，说明 LLM 已经给出最终答案，循环结束。
            if tool_calls is None:
                content = assistant_message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError("模型最终响应缺少文本 content")
                return content

            if not isinstance(tool_calls, list) or not tool_calls:
                raise RuntimeError("模型响应中的 tool_calls 必须是非空数组")

            # 同一条 assistant 消息可能请求多个工具。
            # 最小实现按 LLM 给出的顺序逐个执行。
            for tool_call in tool_calls:
                # 工具结果也属于对话历史，role 必须是 tool。
                messages.append(self._execute_tool_call(tool_call))

            # 执行完工具后不 return，继续下一轮循环。
            # 下一次 LLM 请求会同时看到 assistant tool_calls 和 tool 结果。

        # 防止模型不断调用工具而永远不返回最终文本。
        raise RuntimeError(f"Agent 超过最大调用轮数: {self._max_turns}")

    @staticmethod
    def _read_assistant_message(
        response: dict[str, object],
    ) -> dict[str, object]:
        """读取并检查 Chat Completions 响应中的 assistant 消息。

        OpenAI-compatible 响应的主要结构是：
        {"choices": [{"message": {"role": "assistant", ...}}]}
        """

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("模型响应缺少 choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("模型响应 choices[0] 不是对象")

        raw_message = first_choice.get("message")
        if not isinstance(raw_message, dict):
            raise RuntimeError("模型响应缺少 assistant message")

        role = raw_message.get("role")
        if role != "assistant":
            raise RuntimeError("模型响应 message.role 必须是 assistant")

        # 只保留下一次 API 请求需要的标准消息字段，
        # 不把 usage、finish_reason 等响应元数据塞进 messages。
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": raw_message.get("content"),
        }

        # 普通文本响应没有 tool_calls，工具调用响应才需要添加它。
        if "tool_calls" in raw_message:
            assistant_message["tool_calls"] = raw_message["tool_calls"]
        return assistant_message

    def _execute_tool_call(self, raw_tool_call: object) -> dict[str, object]:
        """解析并执行一个 tool call，返回标准 tool 消息。

        只要 tool call 带有合法 id，参数或工具执行错误就不会直接终止 Agent。
        错误会作为 tool 结果返回给 LLM，让 LLM 有机会修正。
        """

        # tool_call_id 用于把工具结果与原始工具请求关联起来。
        tool_call_id = self._read_tool_call_id(raw_tool_call)
        try:
            function = self._read_function(raw_tool_call)
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("tool call 缺少 function.name")

            tool = self._tools.get(name)
            if tool is None:
                raise ValueError(f"未知工具: {name}")

            # OpenAI 协议中的 arguments 不是字典，而是一段 JSON 字符串。
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, str):
                raise ValueError("function.arguments 必须是 JSON 字符串")

            # 把类似 '{"left": 1, "right": 2}' 的字符串解析成 Python 字典。
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            # 调用注册工具的 execute 函数。
            result = tool.execute(arguments)

            # role=tool 的 content 必须是字符串，所以把结果编码成 JSON。
            content = json.dumps(
                {
                    "ok": True,
                    "result": result,
                },
                ensure_ascii=False,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            # 参数解析、参数校验或结果 JSON 编码错误都作为可恢复错误返回。
            content = json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        except Exception as error:
            # 工具可能抛出其他业务异常，同样把错误交还给 LLM。
            content = json.dumps(
                {
                    "ok": False,
                    "error": f"工具执行失败: {error}",
                },
                ensure_ascii=False,
            )

        # 这是下一轮发送给 LLM 的标准工具结果消息。
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @staticmethod
    def _read_tool_call_id(raw_tool_call: object) -> str:
        """读取 tool call id；没有 id 时无法构造合法的工具结果。"""

        if not isinstance(raw_tool_call, dict):
            raise RuntimeError("tool call 必须是对象")
        tool_call_id = raw_tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise RuntimeError("tool call 缺少 id")
        return tool_call_id

    @staticmethod
    def _read_function(raw_tool_call: object) -> dict[str, object]:
        """读取 tool call 中嵌套的 function 对象。"""

        if not isinstance(raw_tool_call, dict):
            raise ValueError("tool call 必须是对象")
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool call 缺少 function")
        return function
