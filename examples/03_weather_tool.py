"""模拟天气工具的完整 Agent 用例。

这个示例会真实调用配置的 LLM，但天气数据来自本地固定字典，
不会访问外部天气服务。

运行命令：
    uv run python examples/03_weather_tool.py
"""

from python.llm_agent_learning import MinimalAgent
from python.llm_agent_learning.client import create_llm_client
from python.llm_agent_learning.config import load_config
from python.llm_agent_learning.demo_tools import TEST_WEATHER_TOOL


def main() -> None:
    """创建 Agent，注册天气工具并提出一个需要工具的问题。"""

    # 加载模型地址、API key 和模型名。
    config = load_config()

    # create_llm_client() 创建底层 HTTP 客户端。
    client = create_llm_client(
        config,
        "你是天气助手。回答天气问题前必须调用天气工具。",
    )

    # 把 TEST_WEATHER_TOOL 放入 tools，LLM 才能看到并选择这个工具。
    agent = MinimalAgent(
        client=client,
        system_prompt=(
            "你是天气助手。回答天气问题前必须调用 get_test_weather，"
            "并且只能根据工具结果回答。"
        ),
        tools=[TEST_WEATHER_TOOL],
    )

    try:
        # 正常流程：
        # 1. LLM 返回 get_test_weather tool call。
        # 2. Agent 执行本地 Python 函数。
        # 3. Agent 把天气结果发回 LLM。
        # 4. LLM 组织最终中文答案。
        answer = agent.run("北京今天天气怎么样？")
        print(f"最终回答：{answer}")
    finally:
        # 无论运行成功还是异常，都关闭 HTTP 连接池。
        client.close()


if __name__ == "__main__":
    main()
