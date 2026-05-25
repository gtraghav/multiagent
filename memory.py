"""Persistent memory layer — reads/writes state/memory.json."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from schemas import MemoryFact, MemoryStore


class MemoryLayer:
    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / "memory.json"
        state_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> MemoryStore:
        if not self._path.exists():
            return MemoryStore()
        try:
            return MemoryStore.model_validate_json(self._path.read_text(encoding="utf-8"))
        except Exception:
            return MemoryStore()

    def _save(self, store: MemoryStore) -> None:
        self._path.write_text(store.model_dump_json(indent=2), encoding="utf-8")

    def read_all(self) -> list[MemoryFact]:
        return self._load().facts

    def get(self, key: str) -> MemoryFact | None:
        for fact in self._load().facts:
            if fact.key == key:
                return fact
        return None

    _STOP = frozenset({
        "what", "when", "where", "which", "who", "how", "the", "and", "for",
        "that", "this", "with", "have", "are", "was", "his", "her", "its",
        "can", "you", "your", "give", "tell", "find", "get",
    })

    def search(self, query: str) -> list[MemoryFact]:
        # Normalise, split into words, drop stop-words and very short tokens.
        normalized = re.sub(r"[^\w\s]", " ", query.lower())
        words = [w for w in normalized.split() if len(w) > 2 and w not in self._STOP]
        if not words:
            return self._load().facts  # no signal → return everything
        facts = self._load().facts
        return [
            f for f in facts
            if any(
                w in re.sub(r"[^\w\s]", " ", f.key.lower()) or w in f.value.lower()
                for w in words
            )
        ]

    def write(self, facts: list[MemoryFact]) -> None:
        if not facts:
            return
        store = self._load()
        existing = {f.key: i for i, f in enumerate(store.facts)}
        now = datetime.now(timezone.utc).isoformat()
        for fact in facts:
            if not fact.timestamp:
                fact = fact.model_copy(update={"timestamp": now})
            if fact.key in existing:
                store.facts[existing[fact.key]] = fact
            else:
                store.facts.append(fact)
                existing[fact.key] = len(store.facts) - 1
        self._save(store)
