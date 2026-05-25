"""Decision cognitive layer — decides next tool call or final answer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_client import LLM

from schemas import DecisionOutput, GatewayToolCall

# MCP tool definitions passed to every decision call
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web (Tavily primary, DuckDuckGo fallback). Hard-capped at 5 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch clean markdown from a URL via headless browser. Returns full page text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout": {"type": "integer", "default": 20},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_time",
        "description": "Get current time in a named IANA timezone.",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "default": "UTC"}},
            "required": [],
        },
    },
    {
        "name": "currency_convert",
        "description": "Convert money between ISO-3 currencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List a directory inside the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
            "required": [],
        },
    },
    {
        "name": "create_file",
        "description": "Create a new file in the sandbox; errors if it already exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "update_file",
        "description": "Overwrite an existing sandbox file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Find-and-replace inside a sandbox file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "find", "replace"],
        },
    },
]

_SYSTEM_BASE = """\
You are the decision layer of a research agent. Use the available tools to answer
the user's question accurately and completely.

Guidelines:
- When multiple tool calls are independent (e.g. fetching several URLs), issue them
  all in the same turn to minimise round-trips.
- After web_search returns URLs, immediately fetch the most relevant ones in the next turn.
- Stop calling tools once you have enough information to give a thorough answer.
- Your final answer must directly address the user's original question.
"""


class DecisionLayer:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def decide(
        self,
        messages: list[dict[str, Any]],
        system_suffix: str = "",
    ) -> DecisionOutput:
        system = _SYSTEM_BASE + ("\n" + system_suffix if system_suffix else "")

        result = self._llm.chat(
            messages=messages,
            system=system,
            tools=TOOL_DEFS,
            tool_choice="auto",
            auto_route="decision",
            max_tokens=2048,
            temperature=0.7,
        )

        if result.get("stop_reason") == "tool_use" and result.get("tool_calls"):
            calls = [
                GatewayToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    provider_meta=tc.get("provider_meta"),
                )
                for tc in result["tool_calls"]
            ]
            return DecisionOutput(kind="tool_call", tool_calls=calls)

        return DecisionOutput(kind="final", answer=result.get("text", ""))
