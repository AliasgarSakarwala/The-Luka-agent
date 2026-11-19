"""Content module: retrieval-backed response planning."""

from __future__ import annotations

from .llm import query_llm
from .memory_store import MemoryStore


def _format_history(history: list[dict], max_turns: int = 8) -> str:
    """Format recent conversation history into a readable transcript."""
    if not history:
        return "(No prior turns)"
    recent = history[-max_turns:]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "user")
        speaker = "User" if role == "user" else "Luka Dončić"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "(No prior turns)"


def _format_retrieved(retrieved: list[dict]) -> str:
    """Format retrieved memory chunks for the content planner prompt."""
    if not retrieved:
        return "(No retrieved evidence)"
    blocks: list[str] = []
    for idx, chunk in enumerate(retrieved, start=1):
        text = (chunk.get("text") or "").strip()
        topic = chunk.get("topic") or "general"
        meta = chunk.get("metadata") or {}
        title = meta.get("title") or meta.get("source") or "Source"
        source = meta.get("source") or ""
        header = f"[Chunk {idx} | topic={topic} | {title}"
        if source and source != title:
            header += f" ({source})"
        header += "]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def contextualize_query(user_message: str, history: list[dict]) -> str:
    """Rewrite a follow-up user message into a standalone search query using conversation history."""
    cleaned = user_message.strip()
    if not history or not cleaned:
        return cleaned

    history_text = _format_history(history, max_turns=6)
    rewrite_prompt = f"""You rewrite a user's latest message in a conversation with NBA player Luka Dončić into a short, standalone search query for a biographical/basketball memory index.

Rules:
- If the user message uses pronouns or implicit references to earlier turns ("that game", "back then", "what about after that?", "who did you play for next?", "how many did you score?"), resolve them using the conversation history.
- If the user message is already self-contained, keep its core meaning intact.
- Output ONLY the standalone search query (one short line, no quotes, no explanation).

Conversation so far:
{history_text}

Latest user message:
{cleaned}

Standalone search query:"""

    rewritten = query_llm(rewrite_prompt, temperature=0.0).strip().strip('"').strip("'")
    return rewritten if rewritten else cleaned


def make_content_plan(user_message: str, retrieved: list[dict], history: list[dict]) -> str:
    """Create a response outline using the retrieved evidence and the conversation so far."""
    history_text = _format_history(history)
    evidence_text = _format_retrieved(retrieved)

    content_plan_prompt = f"""You are the grounded content planner for a conversational agent embodying NBA star Luka Dončić (knowledge cutoff: March 2026).

Your task is to write a concise bulleted content plan (2 to 4 bullet points) outlining the exact substance Luka's reply should cover.

Strict Grounding Rules:
1. Base all factual claims (dates, stats, scores, teams, awards, teammates, quotes, and career events) STRICTLY on the Retrieved Evidence below (and facts already established in the Conversation So Far).
2. Do NOT invent, guess, or extrapolate any statistics, game outcomes, or biographical details not supported by the Retrieved Evidence.
3. If the user asks a specific factual detail that is NOT covered in the Retrieved Evidence, include a bullet point instructing Luka to honestly say he doesn't recall that exact detail off the top of his head, while sharing whatever related context IS in the evidence.
4. If the user message is a casual greeting, small talk, compliment, or farewell, plan a brief, friendly, natural response appropriate for Luka.
5. Keep the plan concise (bullet points only, no full speech yet).

Conversation so far:
{history_text}

User message:
{user_message}

Retrieved evidence:
{evidence_text}

Content plan (bullet points):"""

    llm_text = query_llm(content_plan_prompt, temperature=0.0)
    return llm_text.strip()


def content_step(memory: MemoryStore, user_message: str, history: list[dict]) -> dict:
    """Retrieve evidence and convert it into a content plan."""
    search_query = contextualize_query(user_message=user_message, history=history)
    retrieved_chunks = memory.retrieve(query=search_query)
    plan = make_content_plan(
        user_message=user_message,
        retrieved=retrieved_chunks,
        history=history,
    )
    return {
        "content_plan": plan,
        "retrieved_chunks": retrieved_chunks,
        "search_query": search_query,
    }
