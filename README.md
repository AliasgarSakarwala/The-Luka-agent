# Luka Dončić (#77) Celebrity Conversational Agent

A full-stack conversational celebrity agent embodying NBA superstar **Luka Dončić** (Los Angeles Lakers #77, Slovenian National Team, former Dallas Mavericks & Real Madrid Baloncesto guard) with a knowledge cutoff of **March 2026**.

The system integrates:
- **Memory Store (`src/memory_store.py`)**: Ingests curated public sources (`data/sources.json`), splits them into overlapping passages, and indexes dense embeddings (`BAAI/bge-small-en-v1.5` via FastEmbed) alongside lexical TF-IDF scores into `data/index/`.
- **Content Module (`src/content_module.py`)**: Contextualizes multi-turn follow-up queries against conversation history, retrieves top-$k$ memory chunks, and produces a strictly grounded content plan (`make_content_plan`).
- **Style Module (`src/style_module.py`)**: Rewrites the grounded content plan into Luka Dončić's relaxed, slightly deadpan European-English athlete cadence (`stylize`) without introducing ungrounded facts.
- **Guardrails (`src/guardrails.py`)**: Detects questions beyond the **March 2026** knowledge cutoff or outside Luka's persona expertise (`out_of_scope`) and responds with an in-character refusal (`decline`).
- **Conversations & Sign-off (`src/conversations.py`, `src/agent.py`)**: Persists multi-turn conversations on disk (`data/conversations/`), detects natural farewells (`should_end`), appends a Luka sign-off (`add_signoff`), and locks ended conversations as read-only.
- **Web Application (`src/server.py`, `frontend/`)**: FastAPI REST backend and React + Vite web interface for starting, browsing, inspecting, and continuing conversations.

---

## Project Layout

```text
celebrity_agent/
  README.md
  requirements.txt
  data/
    sources.json          # Curated public sources on Luka Dončić (up to March 2026)
    index/                # Built memory store (chunks.json, embeddings.npy)
    conversations/        # Saved conversation JSON files
  src/
    llm.py                # Gemini client (gemini-3.5-flash-lite) with .env loading & retry
    memory_store.py       # Chunking, FastEmbed indexing, and hybrid retrieval
    content_module.py     # Follow-up contextualization & grounded content planning
    style_module.py       # Luka Dončić voice stylization
    guardrails.py         # March 2026 cutoff & domain guardrails + in-character decline
    conversations.py      # Create, list, load, and save conversations
    agent.py              # End-to-end turn pipeline (respond, should_end, add_signoff)
    server.py             # FastAPI backend server (/api/conversations/*)
  frontend/               # React + Vite frontend web application
```

---

## Setup & Installation

### 1. Install Python Backend Dependencies

From the `celebrity_agent/` root directory:

```bash
pip install -r requirements.txt
```

### 2. Configure Your Gemini API Key

Export `GEMINI_API_KEY` in your shell environment (or place `GEMINI_API_KEY="your_api_key_here"` in a `.env` file in `celebrity_agent/`):

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. Build the Memory Store Index

Build the vector and lexical index from `data/sources.json` into `data/index/`:

```bash
python -m src.memory_store
```

*(Note: `src/server.py` also automatically builds the memory store on startup if `data/index/` does not exist yet.)*

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Launching the Celebrity Agent Web App

Run the backend and frontend servers in two separate terminals from `celebrity_agent/`:

### Terminal 1 — Start the FastAPI Backend (`http://127.0.0.1:8000`)

```bash
uvicorn src.server:app --host 127.0.0.1 --port 8000 --reload
```

Or equivalently:

```bash
python -m src.server
```

### Terminal 2 — Start the React + Vite Frontend (`http://127.0.0.1:5173`)

```bash
cd frontend
npm run dev
```

Then open **http://127.0.0.1:5173** in your browser. All `/api/*` requests from the Vite dev server are automatically proxied to `http://127.0.0.1:8000`.

---

## REST API Endpoints (`src/server.py`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and persona metadata (`Luka Dončić`, cutoff `March 2026`) |
| `GET` | `/api/conversations` | List all saved conversations (with title, preview, turn count, `ended` status) |
| `POST` | `/api/conversations` | Create and persist a new empty conversation |
| `GET` | `/api/conversations/{conversation_id}` | Load a single conversation by ID |
| `POST` | `/api/conversations/{conversation_id}/messages` | Send `{"message": "..."}` and receive `{reply, conversation}` (returns HTTP 400 if conversation already ended) |
