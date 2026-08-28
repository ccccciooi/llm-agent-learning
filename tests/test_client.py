"""LLMClient 的真实 HTTP 集成测试。

这个测试使用配置文件中的供应商、API key 和模型，不使用 MockTransport。
运行测试会真实请求一次模型并产生相应费用。
"""

import unittest

from study.llm_agent_learning import create_llm_client
from study.llm_agent_learning import load_config


class RealLLMClientIntegrationTest(unittest.TestCase):
    """验证当前配置能够完成真实 Chat Completions 请求。"""

    def test_real_chat_completion_returns_assistant_message(self) -> None:
        """真实 LLM 应返回包含非空文本的 assistant 消息。"""

        config = load_config()
        client = create_llm_client(
            config,
            "你正在执行连通性测试。请严格按照用户要求回答。",
        )
        try:
            response = client.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "你正在执行连通性测试。请严格按照用户要求回答。",
                    },
                    {
                        "role": "user",
                        "content": "请用一句简短中文回复：连接成功。",
                    },
                ]
            )
        finally:
            client.close()

        choices = response.get("choices")
        self.assertIsInstance(choices, list)
        self.assertGreater(len(choices), 0)

        first_choice = choices[0]
        self.assertIsInstance(first_choice, dict)
        message = first_choice.get("message")
        self.assertIsInstance(message, dict)
        self.assertEqual(message.get("role"), "assistant")

        content = message.get("content")
        self.assertIsInstance(content, str)
        self.assertTrue(content.strip())


if __name__ == "__main__":
    unittest.main()
