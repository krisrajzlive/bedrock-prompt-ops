"""Bedrock model aliases verified for the sandbox (direct ids only; no cross-region profiles)."""
from __future__ import annotations

from typing import Any

MODEL_ALIASES = {
    "nova-micro": "amazon.nova-micro-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "mistral-large-3": "mistral.mistral-large-3-675b-instruct",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
}


def build_llm(alias: str, session: Any, *, temperature: float = 0.0, max_tokens: int = 800):
    from langchain_aws import ChatBedrockConverse

    model_id = MODEL_ALIASES.get(alias, alias)
    client = session.client("bedrock-runtime")
    return ChatBedrockConverse(
        client=client,
        model=model_id,
        region_name=session.region_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def message_text(message: Any) -> str:
    """Plain text of a model reply, dropping reasoning blocks some Bedrock models emit."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
