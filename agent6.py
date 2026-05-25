"""
Cognitive agent for EAGV3 Session 6.

Usage:
    uv run agent6.py <query-text-or-label>

Labels:  A  B  C1  C2  D
  A   — Claude Shannon Wikipedia
  B   — Tokyo weekend activities + weather
  C1  — Query C run 1 (store mom's birthday)
  C2  — Query C run 2 (recall mom's birthday)
  D   — Python asyncio best practices synthesis

Clean state between attempts:  rm -rf state/
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLM

from action import ActionLayer
from decision import DecisionLayer
from memory import MemoryLayer
from perception import PerceptionLayer
from schemas import (
    ActionInput,
    AgentRunInput,
    AgentRunOutput,
    GatewayToolCall,
    PerceptionInput,
)

# ── constants ────────────────────────────────────────────────────────────────
STATE_DIR = Path(__file__).parent / "state"
MAX_ITER = 15

QUERY_LABELS: dict[str, str] = {
    "A": (
        "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
        "birth date, death date, and three key contributions to information theory."
    ),
    "B": (
        "Find 3 family-friendly things to do in Tokyo this weekend. "
        "Check Saturday's weather forecast there and tell me which one is most appropriate."
    ),
    "C1": (
        "My mom's birthday is 15 May 2026. Remember that and give me "
        "a calendar reminder for two weeks before and on the day."
    ),
    "C2": "When is mom's birthday?",
    "D": (
        "Search for 'Python asyncio best practices', read the top 3 results, "
        "and give me a short numbered list of the advice they agree on."
    ),
}


# ── message history helpers ───────────────────────────────────────────────────

def _assistant_turn(tool_calls: list[GatewayToolCall]) -> dict:
    """Build the assistant message containing tool call requests."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
                **({"provider_meta": tc.provider_meta} if tc.provider_meta else {}),
            }
            for tc in tool_calls
        ],
    }


_MAX_TOOL_CONTENT = 6000  # chars per tool result — prevents TPM blowout on large fetches


def _truncate(content: str, limit: int = _MAX_TOOL_CONTENT) -> str:
    if len(content) <= limit:
        return content
    kept = limit - 80
    return content[:kept] + f"\n\n[... truncated {len(content) - kept} chars ...]"


def _tool_turns(results) -> list[dict]:
    return [
        {
            "role": "tool",
            "tool_call_id": r.tool_call_id,
            "content": _truncate(r.content),
        }
        for r in results
    ]


# ── main agent loop ───────────────────────────────────────────────────────────

async def run_agent(inp: AgentRunInput) -> AgentRunOutput:
    load_dotenv(Path(__file__).parent / ".env")

    llm = LLM()
    memory = MemoryLayer(STATE_DIR)
    perception_layer = PerceptionLayer(llm)
    decision_layer = DecisionLayer(llm)

    # 1. Pull relevant memory context
    memory_context = memory.search(inp.query)

    # 2. Perception
    print(f"[perception] query: {inp.query[:80]}...")
    perception_out = perception_layer.perceive(
        PerceptionInput(query=inp.query, memory_context=memory_context)
    )
    print(f"[perception] intent: {perception_out.intent}")
    print(f"[perception] needs_tools: {perception_out.needs_tools}")
    if perception_out.memory_write_facts:
        print(f"[perception] will store {len(perception_out.memory_write_facts)} fact(s)")

    # 3. Short-circuit: memory already has the answer
    if perception_out.has_memory_answer and perception_out.memory_answer:
        memory.write(perception_out.memory_write_facts)
        print("[agent] answered from memory (0 tool iterations)")
        return AgentRunOutput(
            query=inp.query,
            answer=perception_out.memory_answer,
            iterations=0,
            run_id=inp.run_id,
        )

    # 4. Build system suffix with intent + memory context
    system_suffix_parts = [f"User intent: {perception_out.intent}"]
    if memory_context:
        facts_text = "\n".join(f"  - {f.key}: {f.value}" for f in memory_context)
        system_suffix_parts.append(f"Relevant memory facts:\n{facts_text}")
    system_suffix = "\n".join(system_suffix_parts)

    # 5. Decision-action loop
    messages: list[dict] = [{"role": "user", "content": inp.query}]
    iterations = 0
    final_answer = ""

    async with ActionLayer() as action_layer:
        for iteration in range(MAX_ITER):
            print(f"[decision] iteration {iteration + 1}")
            decision_out = decision_layer.decide(messages, system_suffix)

            if decision_out.kind == "final":
                final_answer = decision_out.answer or ""
                iterations = iteration + 1
                print(f"[decision] final answer reached after {iterations} iteration(s)")
                break

            # Execute tool calls
            assert decision_out.tool_calls
            tool_names = [tc.name for tc in decision_out.tool_calls]
            print(f"[action] calling: {tool_names}")
            action_out = await action_layer.execute(
                ActionInput(tool_calls=decision_out.tool_calls)
            )

            # Append to message history
            messages.append(_assistant_turn(decision_out.tool_calls))
            messages.extend(_tool_turns(action_out.results))

            for r in action_out.results:
                status = "ok" if r.success else "error"
                print(f"[action] {r.name}: {status} ({len(r.content)} chars)")
        else:
            # Loop exhausted without final answer — use last text if available
            final_answer = "[max iterations reached without a final answer]"
            iterations = MAX_ITER
            print(f"[agent] WARNING: hit MAX_ITER={MAX_ITER}")

    # 6. Persist any memory facts extracted by perception
    memory.write(perception_out.memory_write_facts)
    if perception_out.memory_write_facts:
        for f in perception_out.memory_write_facts:
            print(f"[memory] stored: {f.key} = {f.value}")

    return AgentRunOutput(
        query=inp.query,
        answer=final_answer,
        iterations=iterations,
        run_id=inp.run_id,
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run agent6.py <query-text-or-label>")
        print(f"Labels: {', '.join(QUERY_LABELS)}")
        sys.exit(1)

    raw = " ".join(sys.argv[1:]).strip()
    query = QUERY_LABELS.get(raw.upper(), raw)

    result = asyncio.run(run_agent(AgentRunInput(query=query, run_id=raw)))

    print("\n" + "=" * 60)
    print(f"QUERY:      {result.query}")
    print(f"ITERATIONS: {result.iterations}")
    print(f"\nANSWER:\n{result.answer}")
    print("=" * 60)


if __name__ == "__main__":
    main()
