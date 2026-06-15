"""Thin factory around ChatOpenAI so model/temperature are configured once."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from .config import DEFAULT_MODEL, TEMPERATURE


def get_llm(model: Optional[str] = None, temperature: Optional[float] = None, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client. Reads OPENAI_API_KEY from the environment."""
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=TEMPERATURE if temperature is None else temperature,
        **kwargs,
    )
