import json

from llm_agent_learning.client import create_llm_client
from llm_agent_learning.config import load_config

if __name__ == '__main__':
    llm_config = load_config()
    client= create_llm_client(llm_config, "你是嚴謹的合約律師。回答要精準、引用法條編號、避免任何主觀形容詞")
    response = client.ask_once("請幫我解釋什麼是租賃合約")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    client.close()

    client= create_llm_client(llm_config, "你是溫柔的幼兒園老師、要對 5 歲小孩說話。用比喻、口語、少於 80 字。")
    response2 = client.ask_once("請幫我解釋什麼是租賃合約")
    print(json.dumps(response2, ensure_ascii=False, indent=2))
    client.close()

    client= create_llm_client(llm_config, "你只回 JSON。schema: {\"answer\": string, \"confidence\": float}")
    response3 = client.ask_once("請幫我解釋什麼是租賃合約")
    print(json.dumps(response3, ensure_ascii=False, indent=2))
    client.close()
