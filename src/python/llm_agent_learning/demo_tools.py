"""供学习和测试使用的简单工具。

这里使用固定天气数据，不访问任何外部天气接口。这样可以把注意力集中在
工具的参数、执行函数和返回值上，而不需要处理网络请求。
"""

from python.llm_agent_learning.agent import AgentTool, JsonValue

# 模拟一个小型天气数据库。
# 字典第一层的 key 是城市名，value 是这个城市对应的天气信息。
TEST_WEATHER_DATA: dict[str, dict[str, JsonValue]] = {
    "北京": {
        "weather": "晴",
        "temperature_c": 28,
    },
    "上海": {
        "weather": "多云",
        "temperature_c": 31,
    },
    "深圳": {
        "weather": "阵雨",
        "temperature_c": 30,
    },
}


def get_test_weather(arguments: dict[str, JsonValue]) -> JsonValue:
    """根据城市名称返回固定的模拟天气。

    Args:
        arguments: Agent 从 LLM 的 function.arguments 中解析出的参数。
            正常结构为 {"city": "北京"}。

    Returns:
        可以被 json.dumps() 编码的天气字典。

    Raises:
        ValueError: city 缺失、类型错误或不在测试数据中。
    """

    # dict.get() 在 city 不存在时返回 None。
    city = arguments.get("city")

    # isinstance(city, str) 检查类型，not city.strip() 排除空字符串和全空格。
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city 必须是非空字符串")

    # 去除用户或 LLM 可能添加在城市名前后的空格。
    normalized_city = city.strip()
    weather = TEST_WEATHER_DATA.get(normalized_city)
    if weather is None:
        raise ValueError(f"没有城市 {normalized_city} 的测试天气数据")

    # 使用 **weather 把内层天气字典展开到返回结果中。
    return {
        "city": normalized_city,
        **weather,
    }


# AgentTool 把 Python 函数和提供给 LLM 阅读的 JSON Schema 绑定在一起。
TEST_WEATHER_TOOL = AgentTool(
    name="get_test_weather",
    description="查询指定中国城市的模拟天气，仅用于学习和测试",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "需要查询的城市名称，例如北京、上海或深圳",
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    execute=get_test_weather,
)
