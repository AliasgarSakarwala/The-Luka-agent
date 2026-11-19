"""Conversation store: create, list, load, and save conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Conversation:
    """One conversation between a user and the agent."""

    conversation_id: str
    turns: list[dict] = field(default_factory=list)
    ended: bool = False


def _resolve_store_dir(store_dir: str = "data/conversations") -> Path:
    """Resolve store_dir relative to current working directory or repo root."""
    path = Path(store_dir)
    if path.is_absolute():
        return path
    if path.exists() or store_dir != "data/conversations":
        return path
    return REPO_ROOT / path


def _conversation_path(conversation_id: str, store_dir: str = "data/conversations") -> Path:
    safe_id = Path(conversation_id).name
    return _resolve_store_dir(store_dir) / f"{safe_id}.json"


def create_conversation(store_dir: str = "data/conversations") -> Conversation:
    """Start a new, empty conversation and save it."""
    timestamp_ms = int(time.time() * 1000)
    short_uid = uuid.uuid4().hex[:6]
    conversation_id = f"conv_{timestamp_ms}_{short_uid}"
    conversation = Conversation(
        conversation_id=conversation_id,
        turns=[],
        ended=False,
    )
    save_conversation(conversation, store_dir=store_dir)
    return conversation


def list_conversations(store_dir: str = "data/conversations") -> list[Conversation]:
    """Return every saved conversation, sorted most recent first."""
    directory = _resolve_store_dir(store_dir)
    if not directory.exists():
        return []

    records: list[tuple[float, str, Conversation]] = []
    for file_path in directory.glob("*.json"):
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            conv = Conversation(
                conversation_id=str(raw.get("conversation_id") or file_path.stem),
                turns=list(raw.get("turns") or []),
                ended=bool(raw.get("ended", False)),
            )
            updated_at = float(raw.get("updated_at") or file_path.stat().st_mtime)
            records.append((updated_at, conv.conversation_id, conv))
        except Exception:
            continue

    records.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in records]


def load_conversation(conversation_id: str, store_dir: str = "data/conversations") -> Conversation:
    """Load one saved conversation."""
    file_path = _conversation_path(conversation_id, store_dir=store_dir)
    if not file_path.exists():
        raise FileNotFoundError(f"Conversation '{conversation_id}' not found in {store_dir}.")

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    return Conversation(
        conversation_id=str(raw.get("conversation_id") or conversation_id),
        turns=list(raw.get("turns") or []),
        ended=bool(raw.get("ended", False)),
    )


def save_conversation(conversation: Conversation, store_dir: str = "data/conversations") -> None:
    """Write a conversation to disk."""
    directory = _resolve_store_dir(store_dir)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{Path(conversation.conversation_id).name}.json"

    payload = {
        "conversation_id": conversation.conversation_id,
        "turns": conversation.turns,
        "ended": bool(conversation.ended),
        "updated_at": time.time(),
    }
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
