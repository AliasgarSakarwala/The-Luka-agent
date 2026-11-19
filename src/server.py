"""FastAPI backend server for the Luka Dončić celebrity conversational agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import respond
from .conversations import (
    Conversation,
    create_conversation,
    list_conversations,
    load_conversation,
    save_conversation,
)
from .memory_store import MemoryStore, build_memory_store, load_memory_store

CONVERSATIONS_DIR = Path("data/conversations")
INDEX_DIR = Path("data/index")

_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Return the loaded MemoryStore singleton, building the index first if missing."""
    global _memory_store
    if _memory_store is not None:
        return _memory_store

    try:
        if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
            build_memory_store(str(INDEX_DIR))
        _memory_store = load_memory_store(str(INDEX_DIR))
    except FileNotFoundError:
        build_memory_store(str(INDEX_DIR))
        _memory_store = load_memory_store(str(INDEX_DIR))

    return _memory_store


def _truncate(text: str, max_len: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _derive_title(turns: list[dict[str, Any]], explicit_title: str | None = None) -> str:
    if explicit_title and explicit_title.strip():
        return _truncate(explicit_title.strip(), 56)
    for turn in turns:
        if turn.get("role") == "user" and str(turn.get("content", "")).strip():
            return _truncate(str(turn["content"]), 52)
    return "New Conversation"


def _derive_preview(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "No messages yet — start chatting with Luka!"
    last_turn = turns[-1]
    prefix = "You: " if last_turn.get("role") == "user" else "Luka: "
    return prefix + _truncate(str(last_turn.get("content", "")), 76)


def _derive_updated_at(conversation_id: str, raw_updated_at: Any = None) -> str | None:
    if isinstance(raw_updated_at, str) and raw_updated_at.strip():
        return raw_updated_at
    if isinstance(raw_updated_at, (int, float)):
        return datetime.fromtimestamp(raw_updated_at, tz=timezone.utc).isoformat()

    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if conv_file.exists():
        mtime = conv_file.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return None


def _serialize_conversation(conv: Conversation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(conv, dict):
        data = dict(conv)
    elif is_dataclass(conv):
        data = asdict(conv)
    else:
        data = {
            "conversation_id": getattr(conv, "conversation_id", ""),
            "turns": list(getattr(conv, "turns", [])),
            "ended": bool(getattr(conv, "ended", False)),
        }

    conversation_id = str(data.get("conversation_id") or data.get("id") or "")
    turns: list[dict[str, Any]] = list(data.get("turns") or [])
    ended = bool(data.get("ended", False))
    user_turn_count = sum(1 for t in turns if t.get("role") == "user")

    return {
        "conversation_id": conversation_id,
        "id": conversation_id,
        "turns": turns,
        "ended": ended,
        "turn_count": len(turns),
        "user_turn_count": user_turn_count,
        "title": _derive_title(turns, data.get("title")),
        "preview": _derive_preview(turns),
        "updated_at": _derive_updated_at(conversation_id, data.get("updated_at")),
    }


def _load_or_404(conversation_id: str) -> Conversation:
    try:
        conv = load_conversation(conversation_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        ) from exc

    if conv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return conv


class MessageRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_memory_store()
    except NotImplementedError:
        pass
    yield


app = FastAPI(
    title="Luka Dončić Celebrity Agent API",
    description="Conversational agent API embodying NBA superstar Luka Dončić (#77, Los Angeles Lakers).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return server health and celebrity persona metadata."""
    return {
        "status": "ok",
        "persona": "Luka Dončić",
        "number": 77,
        "team": "Los Angeles Lakers",
        "knowledge_cutoff": "March 2026",
    }


@app.get("/api/conversations")
def api_list_conversations() -> list[dict[str, Any]]:
    """List all saved conversations ordered from most recent to oldest."""
    conversations = list_conversations()
    serialized = [_serialize_conversation(conv) for conv in conversations]
    return serialized


@app.post("/api/conversations", status_code=201)
def api_create_conversation() -> dict[str, Any]:
    """Create and persist a new empty conversation."""
    conv = create_conversation()
    return _serialize_conversation(conv)


@app.get("/api/conversations/{conversation_id}")
def api_get_conversation(conversation_id: str) -> dict[str, Any]:
    """Retrieve a single conversation by ID."""
    conv = _load_or_404(conversation_id)
    return _serialize_conversation(conv)


@app.post("/api/conversations/{conversation_id}/messages")
def api_send_message(conversation_id: str, payload: MessageRequest) -> dict[str, Any]:
    """Send a user message in an active conversation and return Luka's response."""
    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    conv = _load_or_404(conversation_id)
    if conv.ended:
        raise HTTPException(
            status_code=400,
            detail="Conversation has already ended and is read-only.",
        )

    memory = get_memory_store()
    try:
        reply = respond(memory, conv, user_message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_conversation(conv)
    serialized = _serialize_conversation(conv)
    return {
        "reply": reply,
        "conversation": serialized,
        **serialized,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)
