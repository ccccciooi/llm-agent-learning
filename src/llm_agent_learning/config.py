import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "src" / "resources" / "config.yaml"
ROOT_DOTENV_FILE = PROJECT_ROOT / ".env"
RESOURCE_DOTENV_FILE = PROJECT_ROOT / "src" / "resources" / ".env"
ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z0-9_.-]+)(?::([^}]*))?\}$")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    base_uri: str
    api_key: str
    model: str
    authorization_header: str
    authorization_prefix: str


"""
配置文件内容解析
"""


def _resolve_value(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"缺少必要的配置项: {path}。"
            f"请检查配置文件 {CONFIG_FILE} 和环境变量。"
        )

    match = ENV_PLACEHOLDER.fullmatch(value)
    if not match:
        return value

    env_name, default = match.groups()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    if default is not None:
        return default

    raise RuntimeError(
        f"缺少必要的环境变量: {env_name}。"
        f"{path} 在配置文件 {CONFIG_FILE} 中配置为 {value}。"
    )


"""
读取配置文件
"""


def _read_yaml() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise RuntimeError(f"配置文件不存在: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise RuntimeError(f"文件类型不合法: {CONFIG_FILE}")
    return data


"""
加载配置
"""


def load_config() -> LLMConfig:
    logging.info("======开始加载配置文件=======")
    load_dotenv(ROOT_DOTENV_FILE)
    load_dotenv(RESOURCE_DOTENV_FILE)
    data = _read_yaml()
    llm = data.get("llm")
    if not isinstance(llm, dict):
        raise RuntimeError(f"配置文件缺少LLM配置: llm in {CONFIG_FILE}")

    llm_config = LLMConfig(base_url=_resolve_value(llm.get("base-url"), "llm.base-url"),
                           base_uri=_resolve_value(llm.get("base-uri"), "llm.base-uri"),
                           api_key=_resolve_value(llm.get("api-key"), "llm.api-key"),
                           model=_resolve_value(llm.get("model"), "llm.model"),
                           authorization_header=_resolve_value(llm.get("authorization-header"), "llm.authorization"),
                           authorization_prefix=_resolve_value(llm.get("authorization-prefix"), "llm.authorization"),
                           )
    logging.info("======配置加载完毕======")
    return llm_config
