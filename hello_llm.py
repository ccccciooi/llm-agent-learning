import sys

from python.llm_agent_learning.client import create_client
from python.llm_agent_learning.config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    config = load_config()
    client = create_client(config)

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": "用一句话介绍你自己，并说明你当前由哪个模型提供服务。",
            }
        ],
    )

    print("回应：")
    print(response.choices[0].message.content)
    print("\nusage:")
    print(response.usage)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"配置错误：{error}", file=sys.stderr)
        raise SystemExit(1) from None
