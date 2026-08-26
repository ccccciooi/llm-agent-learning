# llm-agent-learning

这是一个练习项目，用来学习如何调用兼容 OpenAI 协议的 LLM 供应商。

## 1. 按 Java/Spring 的习惯理解配置

这个项目按接近 Spring Boot 的方式组织配置：

- `application.yaml`：提交到 git，声明配置项和环境变量占位符。
- `.env`：本地开发用，放真实值，不提交 git。
- `src/llm_agent_learning/config.py`：类似 Spring 的 `@ConfigurationProperties`，负责把配置绑定成对象。

`application.yaml`：

```yaml
llm:
  base-url: ${OPENAI_COMPAT_BASE_URL}
  api-key: ${OPENAI_COMPAT_API_KEY}
  model: ${OPENAI_COMPAT_MODEL}
```

复制 `.env.example` 为 `.env`，然后填入你的供应商信息：

```bash
OPENAI_COMPAT_BASE_URL=https://your-provider.example.com/v1
OPENAI_COMPAT_API_KEY=your-api-key
OPENAI_COMPAT_MODEL=your-model-name
```

说明：

- `OPENAI_COMPAT_BASE_URL` 决定请求发给哪个供应商。
- `OPENAI_COMPAT_API_KEY` 是供应商给你的 API key。
- `OPENAI_COMPAT_MODEL` 是供应商文档里的模型名。

`.env` 已经被 `.gitignore` 忽略，不要把真实 key 提交到 git。这个模式对应 Spring Boot 里的：

```yaml
llm:
  base-url: ${OPENAI_COMPAT_BASE_URL}
  api-key: ${OPENAI_COMPAT_API_KEY}
  model: ${OPENAI_COMPAT_MODEL}
```

真实值由环境变量、启动参数、IDE Run Configuration 或本地 `.env` 注入。

## 2. 跑第一个最小脚本

```bash
uv run python hello_llm.py
```

## 3. 项目结构

```text
llm-agent-learning/
├── .env.example
├── .gitignore
├── README.md
├── application.yaml
├── pyproject.toml
├── hello_llm.py
├── examples/
│   └── 01_hello_llm.py
└── src/
    └── llm_agent_learning/
        ├── __init__.py
        ├── client.py
        └── config.py
```

学习顺序：

1. 先读 `application.yaml`，理解配置项如何声明。
2. 再读 `src/llm_agent_learning/config.py`，理解配置如何被绑定成 `LLMConfig`。
3. 再读 `src/llm_agent_learning/client.py`，理解如何封装可复用的 LLM 调用函数。
4. 最后跑 `examples/01_hello_llm.py`，看正式项目结构下如何复用代码。

## 4. 常见错误

- `Missing required environment variable`：`.env` 没创建，或变量没填完整。
- `Missing config section: llm`：`application.yaml` 结构不对，缺少 `llm:`。
- `401` / `403`：API key 错误、过期，或账号没有权限。
- `404`：通常是 `base_url` 或 `model` 写错；先确认 `base_url` 是否应该以 `/v1` 结尾。
- `429`：触发供应商限流，稍后重试。

## 5. 最小 Agent 循环

`src/llm_agent_learning/agent.py` 实现了同步、非流式的最小 Agent：

```text
用户消息
  → LLM 返回 tool_calls
  → Python 执行工具
  → 把 tool 结果追加到 messages
  → 再次调用 LLM
  → LLM 返回最终文本
```

运行加法工具示例：

```bash
uv run python examples/02_minimal_agent.py
```

运行模拟天气工具示例：

```bash
uv run python examples/03_weather_tool.py
```

天气示例中的数据来自本地固定字典，用于观察完整工具调用流程，不会请求外部天气服务。

核心用法：

```python
from llm_agent_learning.agent import AgentTool, MinimalAgent

tool = AgentTool(
    name="add_numbers",
    description="计算两个数字的和",
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["left", "right"],
    },
    execute=add_numbers,
)

agent = MinimalAgent(
    client=client,
    system_prompt="遇到算术问题必须调用工具。",
    tools=[tool],
)
answer = agent.run("计算 123 加 456")
```

这个实现支持一轮中的多个工具调用，并设置了最大调用轮数，防止模型无限循环。第一版不包含流式输出、并行工具、跨调用会话记忆和 TUI。

运行真实 LLM 集成测试：

```bash
uv run python -m unittest discover -s tests -v
```

当前 `tests/` 使用 `config.yaml` 和本地 `.env` 中配置的真实 LLM，不使用模拟响应。运行测试会访问网络并产生模型请求费用，结果也可能受到供应商状态和模型行为影响。
