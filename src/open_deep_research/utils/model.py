from __future__ import annotations

import os
from functools import partial
from typing import Optional

from langchain_openai import ChatOpenAI


def init_chat_model(model: str, max_tokens: int, api_key: Optional[str] = None) -> ChatOpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Missing OpenAI API key. Pass `api_key` or set environment variable OPENAI_API_KEY."
        )
    return ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        api_key=key,
    )


default_model_config = partial(
    init_chat_model,
    model="gpt-4o-mini",
    max_tokens=2048,  
)