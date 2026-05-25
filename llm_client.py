"""Gateway LLM client with automatic retry on 503/429 (provider cooldown / rate limit)."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import httpx


def _load_base() -> type:
    spec = importlib.util.spec_from_file_location(
        "gateway_client", Path(__file__).parent / "llm_gatewayV3" / "client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.LLM


_BaseLLM = _load_base()


class LLM(_BaseLLM):
    """Gateway LLM with retry-on-503/429 to survive single-provider cooldowns."""

    def chat(self, *args, **kwargs) -> dict:
        delay = 3.0
        for attempt in range(4):  # up to 3 retries; 3s clears Groq's 2s cooldown
            try:
                return super().chat(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 503) and attempt < 3:
                    print(f"[llm] {exc.response.status_code} — retry {attempt + 1}/3 in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 1.5  # 3s → 4.5s → 6.75s
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover
