from llm_agent_learning.client import create_llm_client
from llm_agent_learning.config import load_config

if __name__ == '__main__':
    llm_config = load_config()
    client= create_llm_client(llm_config, "你是一个AI助手")
    response = client.ask_once("你好呀")
    print(response)
