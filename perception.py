"""Perception cognitive layer — parses user query into structured intent."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_client import LLM

from schemas import MemoryFact, PerceptionInput, PerceptionOutput

_SYSTEM = """\
You are the perception layer of a cognitive agent. Your job is to parse the user's
query and produce a structured JSON object.

Rules:
1. `intent`: one sentence describing what the user wants.
2. `needs_tools`: true unless `memory_context` already fully answers the query.
3. `memory_write_facts`: extract every explicit "remember X" / "store X" / "save X"
   directive into a list of {key, value, timestamp} objects.
   - key: snake_case identifier (e.g. "moms_birthday")
   - value: the exact fact to store (e.g. "May 15, 2026")
   - timestamp: current ISO datetime (provided below)
4. `has_memory_answer`: true only when `memory_context` contains a fact that directly
   and completely answers the question.
5. `memory_answer`: if `has_memory_answer` is true, write the natural-language answer
   here (e.g. "Your mom's birthday is May 15, 2026."). Otherwise null.
"""


class PerceptionLayer:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def perceive(self, inp: PerceptionInput) -> PerceptionOutput:
        now = datetime.now(timezone.utc).isoformat()
        memory_block = ""
        if inp.memory_context:
            lines = [f"  - {f.key}: {f.value}" for f in inp.memory_context]
            memory_block = "Memory context (facts retrieved for this query):\n" + "\n".join(lines) + "\n\n"

        prompt = (
            f"Current UTC time: {now}\n\n"
            f"{memory_block}"
            f"User query: {inp.query}"
        )

        schema = PerceptionOutput.model_json_schema()
        result = self._llm.chat(
            prompt=prompt,
            system=_SYSTEM,
            auto_route="perception",
            response_format={
                "type": "json_schema",
                "schema": schema,
                "name": "perception_output",
                "strict": True,
            },
            temperature=0.3,
            max_tokens=1024,
        )

        parsed = result.get("parsed") or {}
        # Ensure any memory_write_facts have a timestamp
        facts = []
        for f in parsed.get("memory_write_facts", []):
            if not f.get("timestamp"):
                f["timestamp"] = now
            facts.append(MemoryFact(**f))
        parsed["memory_write_facts"] = [f.model_dump() for f in facts]

        return PerceptionOutput.model_validate(parsed)
