"""Memory store: ingest Luka Dončić public sources -> chunk & embed -> retrieve. It turns our store of information into a searchable index and finds the most relevant passages"""

from __future__ import annotations

from collections import Counter #Counter is a dictionary that counts things 
from dataclasses import asdict, dataclass, field #Tools to help us create and manage our data.
import json
import math
from pathlib import Path
import re
from typing import Any #Type hint meaning
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul") #Suppressing warning related to matrix multiplication.

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
DEFAULT_INDEX_JSON = PROJECT_ROOT / "data" / "index.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 5
HYBRID_DENSE_WEIGHT = 0.80
HYBRID_KEYWORD_WEIGHT = 0.20

_embedder = None
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*") #This is a regular expression that matches any alphanumeric string with hyphens.


def _clean_text(text: str) -> str:
    """Clean bracketed Wikipedia-style citations and normalize whitespace."""
    text = re.sub(r"\[\d+\]", "", text) #This is a regular expression that removes any text in square brackets.
    text = re.sub(r"\s+", " ", text) #This is a regular expression that replaces any whitespace with a single space and double space with a single space.
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer preserving hyphenated terms."""
    return _TOKEN_RE.findall(text.lower()) #Lowercase it then find all the tokens in the text, findall returns a list of all the matches.



#After chunking, we create a matrix of all the embedded chunks, each row is a vector representation of a chunk.
def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings using FastEmbed BAAI/bge-small-en-v1.5."""
    global _embedder
    if not texts:
        return np.zeros((0, 0), dtype=np.float32) #The the text is empty, return a zero matrix.
    if _embedder is None:
        from fastembed import TextEmbedding #Import the TextEmbedding class from the fastembed library. (This is a lazy import)

        cache_dir = PROJECT_ROOT / "data" / ".fastembed_cache" #Create a cache directory for the embeddings.
        cache_dir.mkdir(parents=True, exist_ok=True) #Create the cache directory if it doesn't exist.
        _embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(cache_dir)) #Create an instance of the TextEmbedding class.
    return np.array(list(_embedder.embed(list(texts))), dtype=np.float32) #Embed the text using the FastEmbed BAAI/bge-small-en-v1.5 model.


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and every row of matrix."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10) #Dividing by the vectors length, adding a small epsilon to avoid division by zero.
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) #Calculating the length of each row in the matrix.
    matrix_normed = matrix / np.clip(row_norms, 1e-10, None) #Clipping the row norms to avoid division by zero.
    return (matrix_normed @ query_norm).astype(np.float32)


def _keyword_tfidf_scores(query: str, chunks: list[MemoryChunk]) -> np.ndarray:
    """Compute TF-IDF cosine similarity scores across all chunks ."""
    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    doc_tokens = [
        _tokenize(f"{c.topic} {c.metadata.get('title', '')} {c.text}") #Doc tokens are the tokens in the topic, title, and text of the chunk.
        for c in chunks
    ]
    n_docs = len(doc_tokens) #Length of the document tokens.

    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens)) #Updating the frequency dictionary with the tokens.

    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    def tfidf_vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {term: count * idf.get(term, 0.0) for term, count in counts.items()}

    def sparse_cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        numerator = sum(vec_a[t] * vec_b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return numerator / (norm_a * norm_b)

    query_vec = tfidf_vector(_tokenize(query))
    scores = [sparse_cosine(query_vec, tfidf_vector(tokens)) for tokens in doc_tokens]
    return np.array(scores, dtype=np.float32)


@dataclass
class MemoryChunk:
    """Single memory chunk stored in the index."""

    text: str
    topic: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStore:
    """Container for memory chunks and retrieval interface."""

    chunks: list[MemoryChunk] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32),
        repr=False,
    )
        #Retrieve first embeds the question using the same embedder, then runs a cosine similarity score against all the embedded chunks. Returning the top k chunks.
        #Then we use TF-IDF to score each word in the question against the chunks and include it in our final score of each chunk
    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
        """Return the top-k most relevant chunks for a given user query."""
        if not self.chunks or not query or not query.strip():
            return []

        if self.embeddings.size == 0 or self.embeddings.shape[0] != len(self.chunks):
            self.embeddings = embed([c.text for c in self.chunks])

        query_vec = embed([query.strip()])[0]
        dense_scores = _cosine_scores(query_vec, self.embeddings)
        kw_scores = _keyword_tfidf_scores(query.strip(), self.chunks)

        combined_scores = (
            HYBRID_DENSE_WEIGHT * dense_scores + HYBRID_KEYWORD_WEIGHT * kw_scores
        )

        k = min(k, len(self.chunks))
        top_indices = np.argsort(-combined_scores)[:k]

        results: list[dict] = []
        for idx in top_indices:
            chunk_obj = self.chunks[int(idx)]
            results.append(
                {
                    "text": chunk_obj.text,
                    "topic": chunk_obj.topic,
                    "metadata": dict(chunk_obj.metadata),
                    "score": round(float(combined_scores[int(idx)]), 4),
                }
            )
        return results


def _chunk_source(
    source_record: dict[str, Any],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[MemoryChunk]:
    """Split a single source record into overlapping word-window MemoryChunks."""
    raw_text = source_record.get("text", "")
    cleaned = _clean_text(raw_text)
    words = cleaned.split() #Split the setence into words and store it as a list.
    if not words:
        return []

    topic = source_record.get("topic") or "biography"
    source_name = source_record.get("source", "Public Source")
    title = source_record.get("title", "Luka Dončić")
    url = source_record.get("url", "")

    #This is where we slide the window across the sentence and create the chunks.
    step = max(1, size - overlap) 
    chunks: list[MemoryChunk] = []
    i = 0
    chunk_id = 0
    while True:
        window = words[i : i + size]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append(
            MemoryChunk(
                text=chunk_text,
                topic=topic,
                metadata={
                    "source": source_name,
                    "title": title,
                    "url": url,
                    "chunk_id": chunk_id,
                },
            )
        )
        chunk_id += 1
        if i + size >= len(words):
            break
        i += step
    return chunks


def _resolve_path(path_str: str | Path) -> Path:
    """Resolve relative paths cleanly whether called from repo root or elsewhere."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def build_memory_store(out_dir: str = "data/index") -> None:
    """Build the memory store from data/sources.json and save to out_dir."""
    sources_path = DEFAULT_SOURCES_PATH
    if not sources_path.exists():
        raise FileNotFoundError(f"Sources file not found at {sources_path}")

    raw_sources = json.loads(sources_path.read_text(encoding="utf-8"))
    all_chunks: list[MemoryChunk] = []
    for source_record in raw_sources:
        all_chunks.extend(_chunk_source(source_record))

    texts = [c.text for c in all_chunks]
    matrix = embed(texts)

    target_dir = _resolve_path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    serialized_chunks = [asdict(c) for c in all_chunks]
    chunks_path = target_dir / "chunks.json"
    embeddings_path = target_dir / "embeddings.npy"

    chunks_path.write_text(
        json.dumps(serialized_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(embeddings_path, matrix)

    # Also write data/index.json for easy human inspection (matching sample_index.json format)
    DEFAULT_INDEX_JSON.write_text(
        json.dumps({"chunks": serialized_chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Built memory store: {len(raw_sources)} sources -> {len(all_chunks)} chunks "
        f"(embeddings shape: {matrix.shape}) saved to {target_dir}"
    )

#The load memory store loads both the list of chunks and the matrix of embedded chunks.
def load_memory_store(index_dir: str = "data/index") -> MemoryStore:
    """Load a memory store from disk, building it automatically if missing."""
    target_dir = _resolve_path(index_dir)
    chunks_path = target_dir / "chunks.json"
    embeddings_path = target_dir / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        build_memory_store(out_dir=str(target_dir))

    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [
        MemoryChunk(
            text=item["text"],
            topic=item.get("topic", "general"),
            metadata=item.get("metadata", {}),
        )
        for item in raw_chunks
    ]
    matrix = np.load(embeddings_path).astype(np.float32)
    return MemoryStore(chunks=chunks, embeddings=matrix)


def main() -> None:
    build_memory_store()


if __name__ == "__main__":
    main()
