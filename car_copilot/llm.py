"""Thin factory around ChatOpenAI so model/temperature are configured once."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from .config import DEFAULT_MODEL, TEMPERATURE


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> ChatOpenAI:
    """Build a ChatOpenAI client.

    If ``api_key`` is given it is used explicitly (the web app passes the
    visitor's key this way). If omitted, ChatOpenAI falls back to the
    OPENAI_API_KEY environment variable (used by the CLI and evals locally).
    """
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=TEMPERATURE if temperature is None else temperature,
        **kwargs,
    )
