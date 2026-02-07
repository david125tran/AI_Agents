# ./employee_handbook_kb.py

# ---------------------------------- Imports ----------------------------------
import boto3
from colorama import Fore, Style, init
init(autoreset=True)
from datetime import datetime, timezone
from docx import Document as DocxDocument
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List


# ------------------------------------ Functions ----------------------------------
def print_banner(text: str, *, color: str = Fore.LIGHTWHITE_EX, width: int = 65) -> None:
    """
    Print a banner with the given text for visual separation in console output.
    color should be a colorama Fore.* value (e.g., Fore.CYAN).
    """
    line = f"{color}* {Style.RESET_ALL}" * width
    print("\n")
    print(line)
    print(f"{color}*{Style.RESET_ALL} {color}{text}{Style.RESET_ALL}")
    print(line)
    print("\n")


def stable_chunk_id(text: str) -> str:
    """Stable ID for a chunk so duplicates collapse reliably."""
    normalized = " ".join(text.split())  # collapse whitespace
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embed_unique_chunks_bedrock(
    chunk_strings: Dict[str, str],
    *,
    region: str,
    embedding_model_id: str,
    kb_name: str,
) -> List[Dict[str, Any]]:
    """
    Embed the given chunk strings using AWS Bedrock embeddings, 
    ensuring that duplicate chunks (by stable ID) are only embedded once.
    """

    client = boto3.client("bedrock-runtime", region_name=region)

    now = datetime.now(timezone.utc).isoformat()

    seen_ids = set()
    records: List[Dict[str, Any]] = []

    for bin_name, chunk_text in chunk_strings.items():
        if not chunk_text or not chunk_text.strip():
            continue

        chunk_id = stable_chunk_id(chunk_text)
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        body = {"inputText": chunk_text}

        resp = client.invoke_model(
            modelId=embedding_model_id,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )

        payload = json.loads(resp["body"].read())

        embedding = payload.get("embedding")
        if embedding is None:
            raise RuntimeError(f"Unexpected embedding response keys: {list(payload.keys())}")

        metadata = {
            "kb": kb_name,
            "bin": bin_name,
            "format": "csv",
            "created_at": now
        }

        records.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "embedding": embedding,
                "metadata": metadata,
            }
        )

    return records


def chunk_employee_handbook(EMPLOYEE_HANDBOOK: Path) -> List[Dict[str, Any]]:
    """
    Split the handbook into chunks based on section headers.
    Each chunk = one policy section (e.g., 'Workplace Attire').
    """
    doc = DocxDocument(str(EMPLOYEE_HANDBOOK))

    chunks = []
    current_title = "General"
    current_text = []

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue

        # Detect section headers (common handbook pattern)
        if re.match(r"^\d+(\.\d+)*\s+", text) or len(text) < 60 and text.isupper():
            # Save previous section
            if current_text:
                chunks.append({
                    "chunk_id": f"section_{len(chunks)}_{current_title[:20]}",
                    "title": current_title,
                    "text": "\n".join(current_text)
                })
                current_text = []

            current_title = text
        else:
            current_text.append(text)

    # Final section
    if current_text:
        chunks.append({
            "chunk_id": f"section_{len(chunks)}_{current_title[:20]}",
            "title": current_title,
            "text": "\n".join(current_text)
        })

    return chunks


def embed_handbook_chunks(AWS_REGION, EMBEDDING_MODEL_ID, SCRIPT_DIR, chunks):
    """
    Build (or load) a persisted embedding index for the employee handbook.

    - First run: embeds all chunks via Bedrock + saves to disk (JSON)
    - Later runs: loads the saved embeddings so we don't re-embed every time

    Returns: records list of dicts:
      [{"id","text","embedding","metadata"}, ...]
    """
    index_path = SCRIPT_DIR / "employee_handbook_index.json"

    # If we already built the index, load it.
    if index_path.exists():
        try:
            records = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(records, list) and records:
                return records
        except Exception:
            # If file is corrupted or unreadable, fall through and rebuild
            pass

    # Build from scratch
    chunk_strings = {c["chunk_id"]: c["text"] for c in chunks}
    records = embed_unique_chunks_bedrock(
        chunk_strings,
        region=AWS_REGION,
        embedding_model_id=EMBEDDING_MODEL_ID,
        kb_name="employee handbook",
    )

    # Save for next run
    try:
        index_path.write_text(json.dumps(records), encoding="utf-8")
        print(f"[Handbook] Saved embedding index to: {index_path}")
    except Exception as e:
        print(f"[Handbook] Warning: failed to persist index: {e}")

    return records



def extract_section_window(text: str, *, section: str = "8.1", title: str = "Dress Code", max_lines: int = 12) -> str:
    """
    Try to extract the subsection like:
      8.1 Dress Code
      ...
    until the next subsection header (e.g., 8.2 ...) or max_lines.
    """
    if not text:
        return ""

    # Normalize line breaks
    lines = [ln.rstrip() for ln in text.splitlines()]

    # Pattern that matches: "8.1 Dress Code" (flexible spacing)
    start_re = re.compile(rf"^\s*{re.escape(section)}\s+{re.escape(title)}\s*$", re.IGNORECASE)

    # Next header like: "8.2 Something" OR "9.0 Something" etc.
    next_header_re = re.compile(r"^\s*\d+(\.\d+)+\s+\S+", re.IGNORECASE)

    start_idx = None
    for i, ln in enumerate(lines):
        if start_re.match(ln.strip()):
            start_idx = i
            break

    if start_idx is None:
        return ""

    snippet = [lines[start_idx].strip()]
    for j in range(start_idx + 1, len(lines)):
        ln = lines[j].strip()

        # stop at next subsection header (but allow the very next line if it's empty)
        if ln and next_header_re.match(ln) and not ln.lower().startswith(f"{section}."):
            break

        snippet.append(lines[j])
        if len(snippet) >= max_lines:
            break

    # clean extra blank lines at end
    while snippet and not snippet[-1].strip():
        snippet.pop()

    return "\n".join(snippet).strip()


def search_handbook_chunks(AWS_REGION, EMBEDDING_MODEL_ID, query, records, top_k=3):
    """
    Semantic retrieval + visualization:
    - Embeds the query with Bedrock 
    - Computes cosine similarity vs stored chunk embeddings
    - Prints a user friendly "handbook search" visualization showing the exact chunks retrieved
    - Returns top_k record dicts (same structure as before)

    records must contain: "embedding" vectors from Bedrock and "metadata" with the original chunk key.
    """

    # --- helper: embed the query using the same Bedrock embedding model ---
    def embed_query_bedrock(q: str) -> List[float]:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = {"inputText": q}
        resp = client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        embedding = payload.get("embedding")
        if embedding is None:
            raise RuntimeError(f"Unexpected query embedding response keys: {list(payload.keys())}")
        return embedding

    # Cosine similarity
    def cosine(a: List[float], b: List[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        denom = (na ** 0.5) * (nb ** 0.5)
        return dot / denom if denom else 0.0

    # Try to recover the original "handbook_chunk_###" label
    def chunk_label(rec: Dict[str, Any]) -> str:
        # The chunk_id is passed as the dict key to embed_unique_chunks_bedrock, which stores it in metadata["bin"]
        md = rec.get("metadata") or {}
        label = md.get("bin") or md.get("chunk_id") or rec.get("id") or "unknown_chunk"
        return str(label)

    # Safe preview so console output stays readable
    def preview_chunk(text: str, max_chars: int = 300) -> str:
        text = text[:max_chars].rstrip() + "..." 

        return text

    print_banner("🧑‍💼 Employee Handbook RAG Search Visualization")
    print("🧠 Model thought: “Let me look that up in the handbook…”")
    print("📖 Flipping pages…  📄📄📄")
    print(f"🔎 Query: {Fore.YELLOW}{query}{Style.RESET_ALL}\n")

    # Embed query
    print("🧬 Embedding the query with AWS Bedrock…")
    q_emb = embed_query_bedrock(query)

    # Score all records
    print(f"🧲 Comparing against {len(records)} handbook chunks…")
    scored: List[Dict[str, Any]] = []
    for rec in records:
        emb = rec.get("embedding")
        if not emb:
            continue
        score = cosine(q_emb, emb)
        scored.append({"score": score, "rec": rec})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: max(1, int(top_k))]

    # Print what the retriever chose
    print("\n✅ Retrieved chunks (what the model is 'seeing'):\n")
    for rank, item in enumerate(top, start=1):
        rec = item["rec"]
        score = item["score"]
        label = chunk_label(rec)

        # Pretty rank medal
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📌"
        target = extract_section_window(rec.get("text",""), section="8.1", title="Dress Code", max_lines=10)

        print(f"{medal}  Rank #{rank}  |  Chunk: {Fore.CYAN}{label}{Style.RESET_ALL}  |  Similarity: {score:.4f}")

        if target:
            print(f"    🎯 Dress Code Snippet:\n{Fore.LIGHTWHITE_EX}{target}{Style.RESET_ALL}\n")
        else:
            print(f"    👀 Preview: {Fore.LIGHTWHITE_EX}{preview_chunk(rec.get('text',''))}{Style.RESET_ALL}\n")

    print("📌 End of retrieved context. (Only these chunks will be fed into the LLM for RAG.)\n")

    return [x["rec"] for x in top]


