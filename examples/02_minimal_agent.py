"""运行最小 Agent 的完整示例。

运行命令：
    uv run python examples/02_minimal_agent.py

示例会真实请求配置的模型，因此运行前需要正确填写本地 .env。
"""

from python.llm_agent_learning import AgentTool, JsonValue, MinimalAgent
from python.llm_agent_learning.client import create_llm_client
from python.llm_agent_learning.config import load_config


def add_numbers(arguments: dict[str, JsonValue]) -> JsonValue:
    """把 LLM 提供的 left 和 right 相加。

    Agent 会把 function.arguments 的 JSON 字符串解析成 arguments 字典，
    然后调用这个普通 Python 函数。
    """

    # dict.get() 在字段不存在时返回 None，便于统一做参数校验。
    left = arguments.get("left")
    right = arguments.get("right")

    # Python 中 bool 是 int 的子类，因此需要额外排除 True 和 False。
    if (
        not isinstance(left, (int, float))
        or isinstance(left, bool)
        or not isinstance(right, (int, float))
        or isinstance(right, bool)
    ):
        raise ValueError("left 和 right 必须是数字")

    # 返回值必须能被 json.dumps() 编码。
    return {
        "sum": left + right,
    }


def main() -> None:
    """加载配置、注册工具并执行一次 Agent。"""

    # 从 config.yaml 和本地环境变量读取供应商地址、密钥和模型名。
    config = load_config()

    # 创建可复用的 HTTP 客户端。
    # 这里的 system_prompt 供旧的 ask_once() 使用；
    # Agent 会在自己的 messages 中设置系统提示词。
    client = create_llm_client(
        config,
        "你是一个严谨的助手。遇到算术问题必须调用提供的工具。",
    )

    # 把普通 Python 函数包装成 LLM 能理解的工具定义。
    add_tool = AgentTool(
        name="add_numbers",
        description="计算两个数字的和",

        # parameters 使用 JSON Schema。
        # LLM 根据这份定义生成 {"left": ..., "right": ...}。
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

    # 注册工具并设置 Agent 的系统提示词。
    agent = MinimalAgent(
        client=client,
        system_prompt="你是一个严谨的助手。遇到算术问题必须调用提供的工具。",
        tools=[add_tool],
    )

    try:
        # run() 内部会自动完成 LLM -> 工具 -> LLM 循环。
        answer = agent.run("请计算 123 加 456，并告诉我结果。")
        print(f"最终回答：{answer}")
    finally:
        # 即使请求或工具执行报错，也要关闭 HTTP 连接池。
        client.close()


# 只有直接运行这个文件时才执行 main()；
# 如果其他模块 import 本文件，则不会自动请求模型。
if __name__ == "__main__":
    main()
