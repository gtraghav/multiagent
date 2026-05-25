# Assignment 6 — Cognitive Agent

A four-layer cognitive agent built on top of **LLM Gateway V3** and an **MCP stdio server**. No third-party agentic frameworks — the architecture and the contracts are the assignment.

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐   PerceptionInput / PerceptionOutput
│  Perception │──────────────────────────────────────▶  structured intent
│  (LLM)      │                                         memory_write_facts
└─────────────┘                                         has_memory_answer
    │
    ▼  (if needs tools)
┌─────────────┐   messages []  ◀──────────────────────  conversation history
│  Decision   │──────────────────────────────────────▶  DecisionOutput
│  (LLM)      │    kind = "tool_call" | "final"
└─────────────┘
    │ tool_call
    ▼
┌─────────────┐   ActionInput / ActionOutput
│  Action     │──────────────────────────────────────▶  tool results via MCP
│  (MCP)      │
└─────────────┘
    │ loop until final
    ▼
┌─────────────┐
│  Memory     │──────────────────────────────────────▶  state/memory.json
│  (disk)     │    upsert memory_write_facts
└─────────────┘
```

**Every boundary between layers is a typed Pydantic v2 contract. No free-form dicts, no regex on LLM output.**

---

## Cognitive layers

| File | Role | LLM call |
|---|---|---|
| [perception.py](perception.py) | Parses user intent, detects memory directives, checks if memory already answers | `auto_route="perception"`, structured output via `response_format` |
| [decision.py](decision.py) | Decides next tool call or final answer; loops until done | `auto_route="decision"`, native tool use |
| [action.py](action.py) | Executes tool calls against the MCP server over stdio | No LLM — MCP `ClientSession` |
| [memory.py](memory.py) | Reads/writes `state/memory.json`; word-based search for context retrieval | No LLM — pure disk I/O |
| [schemas.py](schemas.py) | All Pydantic v2 contracts for every inter-module boundary | — |
| [llm_client.py](llm_client.py) | Thin wrapper around `llm_gatewayV3/client.py` with retry-on-503/429 | — |
| [agent6.py](agent6.py) | Main loop — wires the four layers together | — |

---

## Setup

**Prerequisites:** Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/)

```bash
# 1. Install dependencies
cd "assignment 6"
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY or ANTHROPIC_API_KEY
# (any single worker provider is enough to run)

# 3. Start LLM Gateway V3 (separate terminal)
cd llm_gatewayV3
./run.sh          # starts on http://localhost:8101
```

---

## Running the agent

```bash
# By label (recommended)
uv run agent6.py A
uv run agent6.py B
uv run agent6.py C1    # Query C — run 1 (stores mom's birthday)
uv run agent6.py C2    # Query C — run 2 (recalls from memory)
uv run agent6.py D

# Or pass the query text directly
uv run agent6.py "What is the capital of France?"

# Clean persistent memory between full re-runs
rm -rf state/
```

---

## Target queries

| Label | Query | Expected behaviour |
|---|---|---|
| **A** | Fetch `wikipedia.org/wiki/Claude_Shannon` — birth date, death date, 3 contributions | 1–2 tool calls (`fetch_url`) |
| **B** | Find 3 family-friendly things to do in Tokyo this weekend; check Saturday's weather; recommend one | 2–3 tool calls (`web_search`) |
| **C1** | "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder." | 0 tool calls — stores fact to `state/memory.json` |
| **C2** | "When is mom's birthday?" *(run after C1)* | 0 tool calls — answered directly from memory |
| **D** | Search "Python asyncio best practices", read top 3 results, list common advice | 4–5 tool calls (`web_search` + `fetch_url` ×3) |

Queries that exceed **2× the expected iteration count** are not considered passing.

---

## MCP tools available

| Tool | Description |
|---|---|
| `web_search` | Tavily (primary) / DuckDuckGo fallback — up to 5 results |
| `fetch_url` | Clean markdown via headless Chromium (crawl4ai) |
| `get_time` | Current time in any IANA timezone |
| `currency_convert` | Live rates via frankfurter.dev |
| `read_file` / `list_dir` | Sandbox file reads |
| `create_file` / `update_file` / `edit_file` | Sandbox file writes |

The MCP server runs as a subprocess managed by `action.py`; the agent never calls provider SDKs directly.

---

## Provider configuration

The gateway reads `assignment 6/.env` and routes automatically. Add keys for any providers you have — the gateway skips providers with missing keys and fails over across the rest.

| Provider | Env var | Shortcut |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `cl`, `claude`, `ant` |
| Google Gemini | `GEMINI_API_KEY` | `g`, `gem` |
| Groq | `GROQ_API_KEY` | `gr` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `n`, `nv` |
| Cerebras | `CEREBRAS_API_KEY` | `c`, `cer` |
| OpenRouter | `OPEN_ROUTER_API_KEY` | `or` |
| GitHub Models | `GITHUB_ACCESS_TOKEN` | `gh` |
| Ollama (local) | `OLLAMA_MODEL` | `o` |

Default model per provider and all rate limits live in `llm_gatewayV3/router.py`. Override any model via `{PROVIDER}_MODEL` in `.env` (e.g. `ANTHROPIC_MODEL=claude-sonnet-4-6`).

---

## Project structure

```
assignment 6/
├── agent6.py          main entry point
├── schemas.py         Pydantic v2 contracts
├── memory.py          persistent memory layer
├── perception.py      perception cognitive layer
├── decision.py        decision cognitive layer
├── action.py          action layer (MCP stdio client)
├── llm_client.py      gateway LLM client with retry
├── mcp_server.py      MCP tool server (stdio)
├── pyproject.toml     uv dependencies
├── .env.example       key/config template
├── .gitignore
├── state/             runtime state (gitignored)
│   └── memory.json    durable memory store
├── sandbox/           MCP file sandbox (gitignored)
└── llm_gatewayV3/     LLM router gateway
    ├── main.py
    ├── providers.py   includes AnthropicProvider
    ├── router.py
    ├── schemas.py
    ├── client.py
    └── run.sh
```
