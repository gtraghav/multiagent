"""Action layer — executes tool calls against the MCP stdio server."""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from schemas import ActionInput, ActionOutput, GatewayToolResult

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"


class ActionLayer:
    """Async context manager that owns an MCP stdio session for its lifetime."""

    def __init__(self, server_path: Path = MCP_SERVER_PATH) -> None:
        self._server_path = server_path
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "ActionLayer":
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command="python",
            args=[str(self._server_path)],
            env={**os.environ},
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()

    async def execute(self, inp: ActionInput) -> ActionOutput:
        assert self._session is not None, "ActionLayer used outside async context"
        results: list[GatewayToolResult] = []
        for tc in inp.tool_calls:
            try:
                mcp_result = await self._session.call_tool(tc.name, tc.arguments)
                parts = []
                for item in mcp_result.content or []:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(json.dumps(item.model_dump() if hasattr(item, "model_dump") else str(item)))
                content = "\n".join(parts) if parts else "(empty result)"
                results.append(
                    GatewayToolResult(tool_call_id=tc.id, name=tc.name, content=content)
                )
            except Exception as exc:
                results.append(
                    GatewayToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=f"error: {exc}",
                        success=False,
                    )
                )
        return ActionOutput(results=results)
