from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import settings


def llm_enabled() -> bool:
    return bool(getattr(settings, "openai_api_key", None) and str(settings.openai_api_key).strip())


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


async def chat_completion(*, system: str, user: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not llm_enabled():
        return ""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
