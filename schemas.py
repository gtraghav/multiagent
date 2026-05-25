"""Pydantic v2 contracts for all inter-module boundaries in agent6."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    key: str        # snake_case identifier e.g. "moms_birthday"
    value: str      # human-readable value
    timestamp: str  # ISO datetime string


class MemoryStore(BaseModel):
    facts: list[MemoryFact] = Field(default_factory=list)


class PerceptionInput(BaseModel):
    query: str
    memory_context: list[MemoryFact] = Field(default_factory=list)


class PerceptionOutput(BaseModel):
    intent: str
    needs_tools: bool
    memory_write_facts: list[MemoryFact] = Field(default_factory=list)
    has_memory_answer: bool = False
    memory_answer: str | None = None


class GatewayToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    provider_meta: dict[str, Any] | None = None  # echoed back for Gemini


class GatewayToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    success: bool = True


class DecisionOutput(BaseModel):
    kind: Literal["tool_call", "final"]
    tool_calls: list[GatewayToolCall] | None = None
    answer: str | None = None


class ActionInput(BaseModel):
    tool_calls: list[GatewayToolCall]


class ActionOutput(BaseModel):
    results: list[GatewayToolResult]


class AgentRunInput(BaseModel):
    query: str
    run_id: str = ""


class AgentRunOutput(BaseModel):
    query: str
    answer: str
    iterations: int
    run_id: str
