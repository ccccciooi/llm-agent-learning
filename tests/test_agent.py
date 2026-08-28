"""MinimalAgent 的真实 LLM 集成测试。

这些测试会读取 src/resources/config.yaml 和本地 .env，真实请求配置的
LLM，并产生相应的请求费用。测试不使用 FakeClient 或模拟模型响应。
"""

import unittest

from python.llm_agent_learning import AgentTool, JsonValue, MinimalAgent
from python.llm_agent_learning.client import LLMClient, create_llm_client
from python.llm_agent_learning.config import load_config
from python.llm_agent_learning.demo_tools import TEST_WEATHER_TOOL, get_test_weather


class RealLLMAgentIntegrationTest(unittest.TestCase):
    """使用真实 LLM 验证工具调用循环。"""

    def create_client(self, system_prompt: str) -> LLMClient:
        """创建真实客户端，并在测试结束时自动关闭连接。"""

        config = load_config()
        client = create_llm_client(config, system_prompt)
        self.addCleanup(client.close)
        return client

    def test_real_llm_calls_add_tool_and_returns_result(self) -> None:
        """真实 LLM 应调用加法工具，并根据工具结果回答。"""

        executed_arguments: list[dict[str, JsonValue]] = []

        def add_numbers(arguments: dict[str, JsonValue]) -> JsonValue:
            """记录真实 LLM 生成的参数，然后执行加法。"""

            executed_arguments.append(arguments.copy())
            left = arguments.get("left")
            right = arguments.get("right")
            if (
                not isinstance(left, (int, float))
                or isinstance(left, bool)
                or not isinstance(right, (int, float))
                or isinstance(right, bool)
            ):
                raise ValueError("left 和 right 必须是数字")
            return {"sum": left + right}

        add_tool = AgentTool(
            name="add_numbers",
            description="计算两个数字的和",
            parameters={
                "type": "object",
                "properties": {
                    "left": {
                        "type": "number",
                        "description": "第一个加数",
                    },
                    "right": {
                        "type": "number",
                        "description": "第二个加数",
                    },
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            execute=add_numbers,
        )
        system_prompt = (
            "你正在执行集成测试。遇到算术问题必须调用 add_numbers 工具，"
            "禁止自己计算。工具返回后，用一句中文回答最终结果。"
        )
        client = self.create_client(system_prompt)
        agent = MinimalAgent(
            client=client,
            system_prompt=system_prompt,
            tools=[add_tool],
        )

        answer = agent.run("请使用工具计算 123 加 456。")

        self.assertGreaterEqual(len(executed_arguments), 1)
        self.assertTrue(
            any(
                arguments.get("left") in (123, 456)
                and arguments.get("right") in (123, 456)
                and arguments.get("left") != arguments.get("right")
                for arguments in executed_arguments
            ),
            f"LLM 没有使用预期参数调用工具: {executed_arguments}",
        )
        self.assertIn("579", answer)

    def test_real_llm_calls_weather_tool_and_returns_result(self) -> None:
        """真实 LLM 应调用模拟天气工具并回答北京天气。"""

        executed_arguments: list[dict[str, JsonValue]] = []

        def tracked_weather(arguments: dict[str, JsonValue]) -> JsonValue:
            """记录参数后复用项目中的模拟天气实现。"""

            executed_arguments.append(arguments.copy())
            return get_test_weather(arguments)

        weather_tool = AgentTool(
            name=TEST_WEATHER_TOOL.name,
            description=TEST_WEATHER_TOOL.description,
            parameters=TEST_WEATHER_TOOL.parameters,
            execute=tracked_weather,
        )
        system_prompt = (
            "你正在执行集成测试。回答天气问题前必须调用 get_test_weather，"
            "禁止猜测天气。工具返回后，用一句中文回答。"
        )
        client = self.create_client(system_prompt)
        agent = MinimalAgent(
            client=client,
            system_prompt=system_prompt,
            tools=[weather_tool],
        )

        answer = agent.run("请使用工具查询北京的天气。")

        self.assertGreaterEqual(len(executed_arguments), 1)
        self.assertTrue(
            any(arguments.get("city") == "北京" for arguments in executed_arguments),
            f"LLM 没有使用北京作为参数调用工具: {executed_arguments}",
        )
        self.assertIn("北京", answer)
        self.assertTrue(
            "28" in answer or "二十八" in answer,
            f"最终回答没有包含工具返回的 28°C: {answer}",
        )


if __name__ == "__main__":
    unittest.main()
