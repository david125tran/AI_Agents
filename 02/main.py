"""
Stock Due Diligence Pipeline (LangGraph + Bedrock + Local RAG)

This script implements an end-to-end research pipeline that takes a natural-language
question about a single stock and produces a sourced, PDF due-diligence report.

High-level flow:

1. Orchestrator
   - Uses an LLM to extract exactly one ticker from the user question.
   - If ambiguous, stops and asks a single clarification question.
   - Otherwise, generates a fixed set of retrieval-oriented queries.

2. Deterministic data layer (data_analyst)
   - Pulls structured data from:
       • Alpha Vantage (quote, price history, overview, financials)
       • SEC EDGAR (recent filings via CIK mapping)
   - Each endpoint is cached to disk with per-endpoint TTLs.
   - On transient failures, falls back to any available cached copy.
   - Produces a consolidated “deterministic” payload plus coverage/warnings metadata.

3. News layer (news_fetcher)
   - Fetches the last ~12 months of company news from Finnhub.
   - Uses a disk cache with TTL and stale fallback.
   - Normalizes, time-filters, sorts, and optionally deduplicates articles.

4. Local RAG archive (archiver)
   - Builds or updates two local vector indexes under:
       cache/vector/<TICKER>/
         • news/          → chunked news articles
         • deterministic/ → a stable “latest facts” document
   - Uses a manifest of document hashes to avoid re-embedding unchanged content.

5. Retrieval (searcher)
   - Runs the orchestrator’s queries against both indexes.
   - Collects top-k chunks, filters old news, deduplicates, and sorts by similarity.

6. Report writer (advisor)
   - Sends a compact snapshot of deterministic facts + retrieved evidence to Bedrock.
   - Forces structured output (rating, confidence, key points, risks, gaps, citations).
   - Requires inline [chunk_id] citations for any evidence-based claims.

7. Validator
   - Performs a lightweight citation audit: checks which report lines contain
     at least one valid source URL from the evidence set.
   - Appends a short “Claim Audit” section to the markdown report.

8. Output
   - The final markdown report is saved as a formatted PDF using ReportLab.
   - All intermediate artifacts (API responses, embeddings, indexes) are persisted
     under the local cache/ directory for reproducibility and speed.

Design goals:
- Favor reliability over speed (aggressive caching + backoff).
- Keep deterministic data separate from LLM reasoning.
- Make every factual claim traceable to a source chunk.
- Allow re-runs without rebuilding embeddings unless content changed.

This file can be executed directly; the graph is invoked at the bottom with an
initial question, and a PDF report is written to ./reports/.
"""


# ---------------------------------- Imports ----------------------------------
import asyncio
from contextlib import contextmanager
import datetime as dt
from datetime import datetime, timezone
from dotenv import load_dotenv
import hashlib
import json
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.node_parser import SentenceSplitter
from langgraph.graph import StateGraph, END
from llama_index.core import load_index_from_storage, Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.bedrock import BedrockEmbedding
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
import random
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
import requests
from textwrap import shorten
import time
from typing import Any, Dict, List, Optional, TypedDict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Supress pdf parsing error messages
logging.getLogger("pypdf").setLevel(logging.ERROR)


# ---------------------------------- Variables ----------------------------------
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent


# ---------------------------------- Pipeline Knobs ----------------------------------
# Environment variables path
DOTENV_PATH = parent_dir / ".env"

# Email for sec.gov endpoint
EMAIL = "david125tran@gmail.com"

# Environment variable names from .env
AWS_REGION_ENV = "AWS_REGION"
BASE_MODEL_ENV = "BASE_MODEL"
MODEL_PROVIDER_ENV = "MODEL_PROVIDER"
ALPHA_VANTAGE_KEY = "ALPHA_VANTAGE_KEY"
FINNHUB_API_KEY = "FINNHUB_API_KEY"

# LLM settings
LLM_TEMPERATURE = 0.0

# Retries and backoffs for LLM api calls
LLM_MAX_RETRIES = 50
LLM_BACKOFF_BASE_SLEEP_S = 2
LLM_BACKOFF_MAX_SLEEP_S = 60.0

# News knobs (lookback window, max articles, dedup)
NEWS_LOOKBACK_DAYS = 365                        # Look for fresh articles
NEWS_MAX_ARTICLES = 100                         # Keep only the most recent N articles
NEWS_DEDUP_BY_URL = True                        # Dedup duplicate urls

# Embeddings settings
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_MIN_INTERVAL_S = 2.0                      # slower = safer
EMBED_BATCH_SIZE = 1                            # smaller batches reduce throttling

# Progress weights to gauge how far along the script is
PROGRESS_WEIGHTS = {
    "orchestrator": 5,                          
    "clarifier": 5,                             
    "data_analyst": 70,                         
    "news_fetcher": 20,                         
    "archiver": 10,                             
    "searcher": 10,                             
    "advisor": 15,                              
    "validator": 5                              
}
TOTAL_WEIGHT = sum(PROGRESS_WEIGHTS.values()) or 1


# ------------------------------ Logging Setup ----------------------------------
def get_logger(name: str = __name__):
    """
    Create (or return) a console logger with a consistent format.

    This helper avoids duplicate handlers if the logger is requested multiple times.
    Intended for scripts where you want readable, timestamped progress output.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("ai_pipeline")


@contextmanager
def log_timing(step: str):
    """
    Context manager that logs START/END messages with elapsed time.
    Use this to track runtime.
    """
    t0 = time.perf_counter()
    logger.info("START %s", step)
    try:
        yield
    finally:
        elapsed_s = time.perf_counter() - t0
        logger.info("END   %s (%.2fs)", step, elapsed_s)


# ---------------------------------- Load Environment Variables ----------------------------------
# Load environment variables
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# AWS region
aws_region = os.getenv(AWS_REGION_ENV)

# AI Model ARN ID
base_model = os.getenv(BASE_MODEL_ENV)

# AI Model provider
model_provider = os.getenv(MODEL_PROVIDER_ENV)

# AlphaVantage for factual stock data
alpha_vantage_api_key = os.getenv(ALPHA_VANTAGE_KEY)

# Finnhub for recent stock news
finnhub_api_key = os.getenv(FINNHUB_API_KEY)


# ---------------------------------- Helper Functions ----------------------------------
def get_llm():
    """
    Build a Bedrock chat client from environment configuration.

    Returns:
        ChatBedrockConverse: A LangChain chat model configured with model ARN,
        provider, region, and temperature.
    """
    return ChatBedrockConverse(
        model=base_model,
        provider=model_provider,
        region_name=aws_region,
        temperature=LLM_TEMPERATURE,
    )


def _is_throttle_error(e: Exception) -> bool:
    """
    Best-effort detector for rate limit / throttling errors.

    Bedrock and some HTTP layers return throttling in slightly different strings.
    This normalizes that into a single boolean so retry logic can be shared.
    """

    msg = str(e).lower()
    return (
        "throttl" in msg
        or "too many requests" in msg
        or "rate exceeded" in msg
        or "reached max retries" in msg
    )


def invoke_with_backoff(
        
    runnable,
    messages,
    *,
    step_name: str,
    max_retries: int = LLM_MAX_RETRIES,
    base_sleep_s: float = LLM_BACKOFF_BASE_SLEEP_S,
    max_sleep_s: float = LLM_BACKOFF_MAX_SLEEP_S,
):
    """
    Invoke a LangChain runnable with retry + exponential backoff on throttling.

    This is designed for Bedrock Converse calls (including structured output).
    Non-throttling failures are raised immediately.

    Args:
        runnable: LangChain runnable (e.g., llm.with_structured_output(...)).
        messages: List of LangChain messages (SystemMessage/HumanMessage).
        step_name: Label used for logs.
        max_retries: Maximum retry attempts before giving up.
        base_sleep_s: Base sleep for exponential backoff (2^n scaling).
        max_sleep_s: Upper bound on backoff sleep.

    Returns:
        The runnable.invoke(...) result.
    """

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return runnable.invoke(messages)
        except Exception as e:
            last_err = e
            msg = str(e)

            is_throttle = (
                "ThrottlingException" in msg
                or "Too many requests" in msg
                or "reached max retries" in msg
                or "throttl" in msg.lower()
            )

            if not is_throttle:
                raise  # not throttling, bubble up

            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            logger.warning(
                "%s | Bedrock throttled. Sleep %.1fs (attempt %d/%d)",
                step_name, sleep_s, attempt, max_retries
            )
            time.sleep(sleep_s)

    raise last_err


def progress_init(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the state contains a progress tracker with sane defaults.

    The progress structure is used across LangGraph nodes to compute a coarse
    percent-complete using PROGRESS_WEIGHTS.
    """

    if "progress" not in state or not isinstance(state.get("progress"), dict):
        state["progress"] = {"done": [], "weight_done": 0, "pct": 0.0, "stage": "start"}
    return state


def progress_mark(state: Dict[str, Any], node: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Mark a node as completed and update weighted percent progress.

    Args:
        state: Current AgentState dict.
        node: Node name to record as completed.
        extra: Optional metadata to store in progress (counts, flags, etc.).

    Returns:
        A new state dict with updated `progress`.
    """

    state = dict(state)
    p = dict(state.get("progress") or {})
    p.setdefault("done", [])
    p.setdefault("weight_done", 0)
    p.setdefault("pct", 0.0)

    if node not in p["done"]:
        p["done"].append(node)
        p["weight_done"] += PROGRESS_WEIGHTS.get(node, 1)

    p["stage"] = node
    den = TOTAL_WEIGHT or 1
    p["pct"] = round(100.0 * p["weight_done"] / den, 1)

    if extra:
        for k, v in extra.items():
            p[k] = v

    state["progress"] = p
    logger.info("PROGRESS %s%% | stage=%s", p["pct"], node)
    return state
    

def fetch_finnhub_company_news_last_12m(
    *,
    ticker: str,
    finnhub_api_key: str,
    cache_root: Path,
    today_utc: Optional[dt.date] = None,
    ttl_s: int = 60 * 60 * 24,   # 24 hour cache for news
    max_attempts: int = 6,
) -> Dict[str, Any]:
    """
    Pull company news from Finnhub for the last 12 months, with disk caching.

    Behavior:
      - Uses a simple TTL cache on disk. Fresh cache wins.
      - If the HTTP request fails, falls back to stale cache if available.
      - Filters articles to the expected time window, sorts newest-first.
      - Optionally deduplicates by URL and caps to NEWS_MAX_ARTICLES.

    Returns:
        Dict containing ticker, window dates, pulled_at, a normalized list of
        articles, and a `source` block describing URL + cache metadata.
    """


    tkr = (ticker or "").strip().upper()
    if not tkr:
        return {"ticker": None, "articles": [], "error": "missing ticker"}

    if not finnhub_api_key:
        return {"ticker": tkr, "articles": [], "error": "missing finnhub_api_key"}

    if today_utc is None:
        today_utc = dt.datetime.utcnow().date()

    start_date = today_utc - dt.timedelta(days=365)
    from_s = start_date.isoformat()
    to_s = today_utc.isoformat()

    pulled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    window_key = "company_news"

    provider = "finnhub"
    key = f"company_news_{window_key}"
    path = cache_root / provider / tkr / f"{key}.json"

    def is_fresh(p: Path, ttl: int) -> bool:
        try:
            return (time.time() - p.stat().st_mtime) <= ttl
        except Exception:
            return False

    def read_json(p: Path) -> Optional[Dict[str, Any]]:
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def write_json_atomic(p: Path, data: Dict[str, Any]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    def http_get_json(url: str) -> Any:
        sleep_s = 1.0
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                r = requests.get(
                    url,
                    timeout=30.0,
                    headers={
                        "User-Agent": "stock-research-bot/1.0",
                        "Accept": "application/json",
                    },
                )

                status = r.status_code
                ct = (r.headers.get("Content-Type") or "").lower()
                text_head = (r.text or "")[:240].replace("\n", " ")

                # hard-fail (don’t waste retries)
                if status in (401, 402, 403):
                    raise RuntimeError(f"HTTP {status} ct={ct} body_head={text_head!r}")

                # retry common transient errors
                if status in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {status} ct={ct} body_head={text_head!r}")

                # if not JSON, fail with context
                if "json" not in ct:
                    raise RuntimeError(f"non_json_response HTTP {status} ct={ct} body_head={text_head!r}")

                try:
                    return r.json()
                except Exception as je:
                    raise RuntimeError(f"json_decode_failed HTTP {status} ct={ct} body_head={text_head!r}") from je

            except Exception as e:
                last_err = e
                jitter = random.uniform(0, 0.25 * sleep_s)
                delay = min(20.0, sleep_s + jitter)
                time.sleep(delay)
                sleep_s *= 2

        raise last_err


    # 1) Cache hit
    if path.exists() and is_fresh(path, ttl_s):
        cached = read_json(path)
        if cached is not None:
            cached.setdefault("source", {})
            cached["source"]["cache"] = {"hit": True, "path": str(path), "ttl_s": ttl_s}
            return cached

    # 2) Fetch
    base = "https://finnhub.io/api/v1"
    # Company news endpoint takes symbol + from/to + token :contentReference[oaicite:6]{index=6}
    url = f"{base}/company-news?symbol={requests.utils.quote(tkr)}&from={from_s}&to={to_s}&token={requests.utils.quote(finnhub_api_key)}"

    try:
        payload = http_get_json(url)
    except Exception as e:
        # fallback to stale cache if present
        if path.exists():
            cached = read_json(path)
            if cached is not None:
                cached.setdefault("source", {})
                cached["source"]["cache"] = {"hit": True, "stale": True, "path": str(path)}
                cached["source"]["error"] = f"fetch_failed_using_stale_cache: {e}"
                return cached
        return {"ticker": tkr, "articles": [], "error": f"fetch_failed: {e}"}

    # Finnhub articles commonly include "datetime" as unix seconds.
    min_ts = int(dt.datetime.combine(start_date, dt.time.min).replace(tzinfo=dt.timezone.utc).timestamp())
    max_ts = int(dt.datetime.combine(today_utc, dt.time.max).replace(tzinfo=dt.timezone.utc).timestamp())

    articles = []
    if isinstance(payload, list):
        for a in payload:
            if not isinstance(a, dict):
                continue
            ts = a.get("datetime")
            try:
                ts_i = int(ts) if ts is not None else None
            except Exception:
                ts_i = None

            if ts_i is not None and (ts_i < min_ts or ts_i > max_ts):
                continue
            articles.append(a)

    # Sort newest first
    articles.sort(key=lambda a: int(a.get("datetime") or 0), reverse=True)

    if NEWS_DEDUP_BY_URL:
        seen = set()
        deduped = []
        for a in articles:
            u = (a.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            deduped.append(a)
        articles = deduped

    articles = articles[:NEWS_MAX_ARTICLES] 

    out = {
        "ticker": tkr,
        "from": from_s,
        "to": to_s,
        "pulled_at": pulled_at,
        "articles": articles,
        "source": {
            "provider": "Finnhub",
            "url": url,
            "cache": {"hit": False, "path": str(path), "ttl_s": ttl_s},
        },
    }

    write_json_atomic(path, out)
    return out


def build_citation_map(evidence_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Build a chunk_id -> citation metadata map from evidence rows.

    The advisor model emits citations like [chunk_id]. This map lets the report
    renderer replace them with readable labels and links.
    """

    cmap: Dict[str, Dict[str, str]] = {}
    for r in evidence_rows:
        cid = (r.get("chunk_id") or "").strip()
        if not cid:
            continue
        meta = r.get("metadata") or {}
        source_type = (meta.get("source_type") or "").strip()
        title = (meta.get("title") or "").strip()
        url = (meta.get("url") or "").strip()
        published_at = (meta.get("published_at") or "").strip()
        source = (meta.get("source") or meta.get("provider") or "").strip()

        label = ""
        if source_type == "news":
            if title and url:
                label = f"{title} — {source or 'News'} ({published_at or 'n.d.'})"
            elif url:
                label = f"{source or 'News'} ({published_at or 'n.d.'})"
        else:
            label = source or meta.get("provider") or "Deterministic"

        cmap[cid] = {
            "label": label,
            "url": url,
            "source_type": source_type,
        }
    return cmap


def replace_citations(md: str, cmap: Dict[str, Dict[str, str]]) -> str:
    """
    Replace inline [chunk_id] citations in markdown with human-readable links.

    For mapped chunk ids, emits:
      - [[Source: <label>]](<url>) if a URL is available
      - [Source: <label>] otherwise

    Unmapped citations are preserved as [unmapped:<chunk_id>].
    """

    def repl(m: re.Match) -> str:
        cid = m.group(1).strip()
        info = cmap.get(cid)
        if not info:
            return f"[unmapped:{cid}]"
        url = (info.get("url") or "").strip()
        label = (info.get("label") or cid).strip()
        if url:
            return f"[[Source: {label}]]({url})"
        return f"[Source: {label}]"

    return re.sub(r"\[([^\]]+::\d+)\]", repl, md)


def print_run_summary(out: Dict[str, Any]) -> None:
    """
    Print a console-friendly summary of the pipeline output.

    Intended for debugging:
      - Ticker / price
      - progress status
      - deterministic coverage + cache root
      - most recent filings
      - top news headlines
      - warnings preview
    """

    ticker = out.get("ticker")
    needs_clar = bool(out.get("needs_clarification"))
    clar_q = out.get("clarification_question") or ""
    price = out.get("current_price")

    progress = out.get("progress") or {}
    pct = progress.get("pct")
    stage = progress.get("stage")
    done = progress.get("done") or []

    det_meta = out.get("deterministic_meta") or {}
    coverage = det_meta.get("coverage") or {}
    warnings_list = det_meta.get("warnings") or []
    cache_root = det_meta.get("cache_root")

    det = out.get("deterministic") or {}
    filings = (det.get("filings_index") or {}).get("filings") or {}
    latest_10k = filings.get("latest_10k") or {}
    latest_10q = filings.get("latest_10q") or {}

    def yn(v: bool) -> str:
        return "OK" if v else "—"

    print("\n" + "=" * 88)
    print("RUN SUMMARY")
    print("=" * 88)

    # Headline
    price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "n/a"
    print(f"Ticker: {ticker or 'n/a'}   |   Price: {price_str}   |   Clarify: {needs_clar}")

    # Clarification
    if needs_clar:
        print(f"Question needed: {clar_q}")

    # Progress
    if pct is not None or stage is not None:
        print(f"Progress: {pct if pct is not None else 'n/a'}%   |   Stage: {stage or 'n/a'}   |   Done: {', '.join(done) or 'n/a'}")

    # Deterministic coverage
    if coverage:
        # Pretty print sorted by key
        cov_items = sorted(coverage.items(), key=lambda kv: kv[0])
        cov_str = " | ".join([f"{k}:{yn(bool(v))}" for k, v in cov_items])
        print(f"Coverage: {cov_str}")
    else:
        print("Coverage: n/a")

    # Cache root
    if cache_root:
        print(f"Cache: {cache_root}")

    # Filings
    if latest_10k:
        print(f"Latest 10-K: {latest_10k.get('filing_date')}   |   {latest_10k.get('primary_url')}")
    if latest_10q:
        print(f"Latest 10-Q: {latest_10q.get('filing_date')}   |   {latest_10q.get('primary_url')}")

    # Warnings (preview)
    if warnings_list:
        print("\nWarnings:")
        for w in warnings_list[:8]:
            print(f"  - {shorten(str(w), width=140, placeholder='...')}")
        if len(warnings_list) > 8:
            print(f"  ... ({len(warnings_list) - 8} more)")
    else:
        print("\nWarnings: none")

    # News preview
    news = out.get("news") or {}
    articles = news.get("articles") or []
    news_meta = out.get("news_meta") or {}
    cache_path = news_meta.get("cache_path") or ((news.get("source") or {}).get("cache") or {}).get("path")

    print("\nNews:")
    print(f"  Articles: {len(articles)}   |   Cache file: {cache_path or 'n/a'}   |   Error: {news.get('error') or 'none'}")

    def fmt_ts(ts):
        try:
            return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
        except Exception:
            return "n/a"

    for a in articles[:8]:
        d = fmt_ts(a.get("datetime"))
        src = (a.get("source") or "").strip() or "n/a"
        head = (a.get("headline") or "").strip() or "n/a"
        url = (a.get("url") or "").strip() or "n/a"
        print(f"  - {d} | {src} | {head}")
        print(f"    {url}")

    print("=" * 88 + "\n")


# ---------------------------------- Save Report as PDF ----------------------------------
def save_report_pdf(out: Dict[str, Any], *, script_dir: Path) -> Optional[Path]:
    """
    Render the final report markdown into a simple PDF using ReportLab.

    Supports:
      - '## ' headings
      - '- ' bullets
      - inline bold via **text**
      - source links of the form [[Source: label]](url)

    Returns:
        Path to the generated PDF, or None if no report content exists.
    """

    report = out.get("report") or {}
    md = (report.get("report_markdown") or "").strip()
    if not md:
        return None

    ticker = (out.get("ticker") or "UNKNOWN").strip().upper()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    reports_dir = script_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = reports_dir / f"{ticker}_due_diligence_{ts}.pdf"

    def _escape_para(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _convert_source_links(line: str) -> str:
        def repl(m: re.Match) -> str:
            label = m.group(1).strip()
            url = m.group(2).strip()
            return f'<a href="{url}">Source: {_escape_para(label)}</a>'
        return re.sub(r"\[\[Source:\s*([^\]]+)\]\]\((https?://[^)]+)\)", repl, line)


    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=f"{ticker} Due Diligence",
        author="Stock Research Bot",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=HexColor("#111827"),
        spaceAfter=16,
    )

    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        textColor=HexColor("#111827"),
        spaceBefore=14,
        spaceAfter=8,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        textColor=HexColor("#111827"),
        spaceAfter=6,
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        textColor=HexColor("#374151"),
        spaceAfter=6,
    )

    elements: List[Any] = []

    # ---- Cover ----
    elements.append(Paragraph(f"{ticker} Due Diligence Report", title_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} UTC", small))
    elements.append(Paragraph(f"<b>Rating:</b> {report.get('rating','n/a')} &nbsp;&nbsp; <b>Confidence:</b> {report.get('confidence','n/a')}", small))
    elements.append(PageBreak())

    # ---- Body ----
    bullet_buf: List[ListItem] = []

    def flush_bullets():
        nonlocal bullet_buf
        if bullet_buf:
            elements.append(ListFlowable(bullet_buf, bulletType="bullet", leftIndent=14))
            bullet_buf = []
            elements.append(Spacer(1, 6))

    for raw in md.splitlines():
        line = (raw or "").strip()

        if not line:
            flush_bullets()
            elements.append(Spacer(1, 8))
            continue

        if line.startswith("## "):
            flush_bullets()
            elements.append(Paragraph(_escape_para(line[3:]), h2))
            continue

        if line.startswith("- "):
            txt = _escape_para(line[2:].strip())
            txt = _convert_source_links(txt)
            bullet_buf.append(ListItem(Paragraph(txt, body)))
            continue

        flush_bullets()

        txt = _escape_para(line)
        txt = _convert_source_links(txt)
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)

        elements.append(Paragraph(txt, body))

    flush_bullets()
    doc.build(elements)
    return pdf_path



# ---------------------------------- Pydantic Output Validation ----------------------------------
class OrchestratorOutput(BaseModel):
    ticker: Optional[str] = Field(
        default=None,
        description="Single extracted ticker (uppercase) if exactly one is present; otherwise null."
    )
    found_multiple: bool = Field(
        description="True if the user mentioned more than one ticker."
    )
    needs_clarification: bool = Field(
        description="True if no ticker was explicitly provided or if ambiguity requires user input."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Ask exactly ONE question when clarification is needed."
    )


class RateLimitedEmbedder(BedrockEmbedding):
    _min_interval_s: float = PrivateAttr(default=2.0)
    _batch_size: int = PrivateAttr(default=1)
    _last: float = PrivateAttr(default=0.0)

    def __init__(self, *args, min_interval_s=2.0, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval_s = float(min_interval_s)
        self._batch_size = int(batch_size)
        self._last = 0.0

    def _sleep_spacing(self):
        now = time.time()
        dt = now - self._last
        if dt < self._min_interval_s:
            time.sleep(self._min_interval_s - dt)
        self._last = time.time()

    async def _asleep_spacing(self):
        now = time.time()
        dt = now - self._last
        if dt < self._min_interval_s:
            await asyncio.sleep(self._min_interval_s - dt)
        self._last = time.time()

    def _embed_batch_with_retries(self, parent_fn, batch_texts, *, max_attempts=8):
        sleep_s = 1.0
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                self._sleep_spacing()
                return parent_fn(batch_texts)
            except Exception as e:
                last_err = e
                if not _is_throttle_error(e):
                    raise

                # exponential backoff + jitter
                jitter = random.uniform(0, 0.25 * sleep_s)
                delay = min(60.0, sleep_s + jitter)
                time.sleep(delay)
                sleep_s *= 2

        raise last_err

    # ---- sync variants ----
    def get_text_embedding_batch(self, texts, **kwargs):
        out = []
        parent_batch = super().get_text_embedding_batch

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            res = self._embed_batch_with_retries(lambda t: parent_batch(t, **kwargs), batch)
            out.extend(res)

        return out

    def get_text_embeddings(self, texts, **kwargs):
        return self.get_text_embedding_batch(texts, **kwargs)

    def get_text_embedding(self, text, **kwargs):
        parent_one = super().get_text_embedding
        res = self._embed_batch_with_retries(lambda t: [parent_one(t[0], **kwargs)], [text])
        return res[0]

    def embed_documents(self, texts, **kwargs):
        return self.get_text_embedding_batch(texts, **kwargs)

    def embed_query(self, text, **kwargs):
        return self.get_text_embedding(text, **kwargs)

    # ---- async variants ----
    async def aget_text_embedding_batch(self, texts, **kwargs):
        out = []
        parent_abatch = super().aget_text_embedding_batch  # coroutine

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]

            sleep_s = 1.0
            last_err = None
            for attempt in range(1, 9):
                try:
                    await self._asleep_spacing()
                    out.extend(await parent_abatch(batch, **kwargs))
                    break
                except Exception as e:
                    last_err = e
                    if not _is_throttle_error(e):
                        raise
                    jitter = random.uniform(0, 0.25 * sleep_s)
                    delay = min(60.0, sleep_s + jitter)
                    await asyncio.sleep(delay)
                    sleep_s *= 2
            else:
                raise last_err

        return out

    async def aget_text_embeddings(self, texts, **kwargs):
        return await self.aget_text_embedding_batch(texts, **kwargs)


class ReportOutput(BaseModel):
    rating: str = Field(description="One of: Buy, Hold, Sell")
    confidence: float = Field(description="0.0 to 1.0 confidence")
    market_sentiment: str = Field(description="Short summary of sentiment based on news evidence")
    key_points: List[str] = Field(description="Top bullets that drove the conclusion")
    risks: List[str] = Field(description="Key risks supported by evidence (or empty if unknown)")
    gaps: List[str] = Field(description="Important missing info that limited the analysis")
    citations_used: List[str] = Field(description="List of chunk_ids used")
    report_markdown: str = Field(description="Full markdown report with inline [chunk_id] citations")


# ---------------------------------- Define Embedder Model ----------------------------------
# LlamaIndex global settings.  Use AWS tital-embed-text-v2:0 embedder
Settings.embed_model = RateLimitedEmbedder(
    model=EMBED_MODEL_ID,
    region_name=aws_region,
    min_interval_s=EMBED_MIN_INTERVAL_S,
    batch_size=EMBED_BATCH_SIZE,
)

# ---------------------------------- Define State ----------------------------------
logger.info("Defining AgentState schema")

# Shared state passed between LangGraph nodes.
class AgentState(TypedDict, total=False):
    # ---- Inputs ----
    question: str

    # ---- Progress / tracing ----
    progress: Dict[str, Any]

    # ---- orchestrator(): Routing ----
    ticker: Optional[str]
    needs_clarification: bool
    clarification_question: str
    queries: List[str]

    # ---- data_analyst(): Deterministic data outputs ----
    current_price: Optional[float]                 # convenience
    deterministic: Dict[str, Any]                  # payload
    deterministic_meta: Dict[str, Any]             # coverage, warnings, sources, cache_root

    # ---- news_fetcher(): Search recent news ----
    news: Dict[str, Any]                 # Finnhub payload (articles, dates, source, etc.)
    news_meta: Dict[str, Any]            # optional (counts, warnings)

    # ---- archiver(): RAG set up ----
    index_meta: Dict[str, Any]

    # ---- searcher(): RAG search ----
    evidence: List[Dict[str, Any]]
    evidence_meta: Dict[str, Any]

    # ---- advisor(): Report writer ----
    report: Dict[str, Any]

    # ---- validator(): Report writer ----
    validation: Dict[str, Any]


logger.info("AgentState defined")


# ---------------------------------- Define Node Functions ----------------------------------
logger.info("Defining graph node functions")

# Define the key nodes, which represents the functions that perform specific tasks in the graph
# They receive the current state and return a modified state

def orchestrator(state: AgentState) -> AgentState:
    """
    Extract a single stock ticker from the user's question and generate
    a small set of retrieval-oriented queries.

    Uses an LLM with structured output to enforce:
      - exactly one ticker, or
      - a single clarification question if missing/ambiguous

    Writes:
      - ticker, needs_clarification, clarification_question, queries
      - progress markers
    """

    with log_timing("orchestrator"):
        state = progress_init(dict(state))
        question = state.get("question") or ""

        llm = get_llm()
        extractor = llm.with_structured_output(OrchestratorOutput)

        prompt = (
            "You extract stock ticker symbols from user text.\n"
            "Return structured output only.\n"
            "Rules:\n"
            "- Tickers are typically 1-5 letters like TSLA, AAPL, NVDA.\n"
            "- Normalize by uppercasing and stripping any '$'.\n"
            "- If user includes MORE THAN ONE ticker: found_multiple=true, needs_clarification=true, ask them to pick ONE.\n"
            "- If no ticker is provided: needs_clarification=true and ask for one ticker.\n"
            "- Ask exactly ONE question when clarification is needed."
        )

        result: OrchestratorOutput = invoke_with_backoff(
            extractor,
            [SystemMessage(content=prompt), HumanMessage(content=question)],
            step_name="orchestrator",
            max_retries=LLM_MAX_RETRIES,
            base_sleep_s=LLM_BACKOFF_BASE_SLEEP_S,
            max_sleep_s=LLM_BACKOFF_MAX_SLEEP_S,
        )

        if result.needs_clarification or result.found_multiple or not result.ticker:
            out: AgentState = {
                **state,
                "ticker": None,
                "needs_clarification": True,
                "clarification_question": (
                    result.clarification_question
                    or "Please provide exactly one stock ticker symbol (e.g., NVDA)."
                ),
                "queries": [],
            }
            return progress_mark(out, "orchestrator", {"clarify": True})

        ticker = (result.ticker or "").strip().upper()

        report_queries = [
            f"{ticker} snapshot price volume market cap",
            f"{ticker} recent price trend last 3 months and 1 year",
            f"{ticker} financial health free cash flow debt net debt liquidity",
            f"{ticker} revenue growth and profitability margin trend",
            f"{ticker} most important recent company news last 90 days",
            f"{ticker} market sentiment headlines tone investor concerns optimism",
            f"{ticker} key risks mentioned in filings and recent disclosures",
            f"{ticker} near term catalysts upcoming events guidance product launches",
            f"{ticker} bull case vs bear case summary from available evidence",
            f"{ticker} investment conclusion is it a good buy based on evidence",
        ]

        out: AgentState = {
            **state,
            "ticker": ticker,
            "needs_clarification": False,
            "clarification_question": "",
            "queries": report_queries,
        }
        return progress_mark(out, "orchestrator", {"ticker": ticker, "query_count": len(report_queries)})



def route_after_orchestrator(state: AgentState) -> str:
    """
    Decide the next node after orchestrator.

    Returns:
        "clarifier" if a ticker is missing/ambiguous, otherwise "data_analyst".
    """

    return "clarifier" if state.get("needs_clarification") else "data_analyst"


def clarifier(state: AgentState) -> AgentState:
    """
    Terminal node for ambiguous requests.

    Leaves the pipeline in a state that clearly asks the user for exactly one
    ticker symbol, and marks progress accordingly.
    """

    with log_timing("clarifier"):
        state = progress_init(dict(state))
        msg = state.get("clarification_question") or "Please provide exactly one stock ticker symbol."
        logger.warning("CLARIFY: %s", msg)
        out = {**state, "needs_clarification": True, "clarification_question": msg}
        return progress_mark(out, "clarifier", {"clarify": True})


def data_analyst(state: AgentState) -> AgentState:
    """
    Deterministic data pull for the selected ticker (Alpha Vantage + SEC EDGAR).

    Responsibilities:
      - Fetch quote, daily adjusted history, overview, financial statements
      - Derive basic metrics (market cap calc, net debt, FCF TTM)
      - Map ticker -> CIK and fetch SEC submissions index
      - Cache each endpoint response on disk with per-endpoint TTLs
      - Fall back to stale cache on transient failures

    Writes:
      - deterministic: consolidated payload
      - deterministic_meta: coverage + warnings + sources + cache_root
      - current_price: convenience field (if available)
    """

    with log_timing("data_analyst"):
        state = progress_init(dict(state))

        ticker = (state.get("ticker") or "").strip().upper()
        if not ticker:
            out = {**state, "deterministic": {}, "deterministic_meta": {"warnings": ["missing ticker"]}}
            return progress_mark(out, "data_analyst", {"det_ok": False})

        api_key = alpha_vantage_api_key
        if not api_key:
            out = {
                **state,
                "deterministic": {},
                "deterministic_meta": {"warnings": ["missing ALPHA_VANTAGE_KEY in environment"]},
            }
            return progress_mark(out, "data_analyst", {"det_ok": False})

        pulled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # ------------------------------ cache ------------------------------
        cache_root = script_dir / "cache"

        ttl_seconds = {
            ("alphavantage", "GLOBAL_QUOTE"): 60 * 60 * 3,                      #  3 h
            ("alphavantage", "TIME_SERIES_DAILY_ADJUSTED"): 60 * 60 * 12,       # 12 h
            ("alphavantage", "OVERVIEW"): 60 * 60 * 24,                         # 24 h
            ("alphavantage", "INCOME_STATEMENT"): 60 * 60 * 24,                 # 24 h
            ("alphavantage", "BALANCE_SHEET"): 60 * 60 * 24,                    # 24 h
            ("alphavantage", "CASH_FLOW"): 60 * 60 * 24,                        # 24 h
            ("sec", "company_tickers"): 60 * 60 * 24 * 7,                       #  7 d
            ("sec", "submissions"): 60 * 60 * 12,                               # 1 2h
        }

        def now_ts() -> float:
            return time.time()

        def safe_float(x):
            try:
                if x is None:
                    return None
                if isinstance(x, (int, float)):
                    return float(x)
                s = str(x).strip()
                if s == "" or s.lower() in {"none", "null", "nan"}:
                    return None
                return float(s)
            except Exception:
                return None

        def safe_int(x):
            try:
                if x is None:
                    return None
                if isinstance(x, int):
                    return int(x)
                s = str(x).strip().replace(",", "")
                if s == "" or s.lower() in {"none", "null", "nan"}:
                    return None
                return int(round(float(s)))
            except Exception:
                return None

        def cache_path(provider: str, tkr: str, key: str) -> Path:
            safe_tkr = re.sub(r"[^A-Za-z0-9_\-\.]", "_", tkr)
            safe_key = re.sub(r"[^A-Za-z0-9_\-\.]", "_", key)
            return cache_root / provider / safe_tkr / f"{safe_key}.json"

        def read_json(path: Path) -> Optional[Dict[str, Any]]:
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(path)

        def is_fresh(path: Path, ttl_s: int) -> bool:
            try:
                age = now_ts() - path.stat().st_mtime
                return age <= float(ttl_s)
            except Exception:
                return False

        def http_get_json(url: str, *, timeout_s: float = 25.0, max_attempts: int = 6) -> Dict[str, Any]:
            sleep_s = 1.0
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    r = requests.get(
                        url,
                        timeout=timeout_s,
                        headers={
                            "User-Agent": f"deterministic-data-analyst/1.0 (contact: {EMAIL})",
                            "Accept": "application/json,text/plain,*/*",
                        },
                    )
                    try:
                        payload = r.json()
                    except Exception:
                        payload = {"_raw_text": r.text[:2000], "_status_code": r.status_code}

                    if r.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"HTTP {r.status_code}")

                    return payload
                except Exception as e:
                    last_err = e
                    jitter = random.uniform(0, 0.25 * sleep_s)
                    delay = min(30.0, sleep_s + jitter)
                    logger.warning(
                        "HTTP retry %d/%d url=%s err=%s sleep=%.1fs",
                        attempt, max_attempts, url, e, delay
                    )
                    time.sleep(delay)
                    sleep_s *= 2
            raise last_err

        def av_url(function: str, **params) -> str:
            base = "https://www.alphavantage.co/query"
            parts = [f"function={function}", f"apikey={api_key}"]
            for k, v in params.items():
                if v is None:
                    continue
                parts.append(f"{k}={requests.utils.quote(str(v))}")
            return base + "?" + "&".join(parts)

        def av_error(payload: Dict[str, Any]) -> Optional[str]:
            if not isinstance(payload, dict):
                return "Alpha Vantage payload not a dict"
            if "Error Message" in payload:
                return payload.get("Error Message")
            if "Information" in payload:
                return payload.get("Information")
            if "Note" in payload:
                return payload.get("Note")
            return None

        def cached_fetch(
            *,
            provider: str,
            tkr: str,
            key: str,
            url: str,
            ttl_fallback_s: int = 3600,
            allow_stale_on_error: bool = True,
            refuse_cache_on_error_payload: bool = True,
        ) -> Dict[str, Any]:
            """
            Cache lookup by (provider, ticker, key). If fresh -> return.
            Else fetch; if fetch fails, optionally return stale cache.

            refuse_cache_on_error_payload: For AlphaVantage, don't overwrite cache with rate-limit messages.
            """
            ttl = ttl_seconds.get((provider, key), ttl_fallback_s)
            path = cache_path(provider, tkr, key)

            # Fast path: fresh cache
            if path.exists() and is_fresh(path, ttl):
                cached = read_json(path)
                if cached is not None:
                    logger.info("CACHE HIT  | %s/%s/%s", provider, tkr, key)
                    return cached

            # Fetch path
            logger.info("CACHE MISS | %s/%s/%s", provider, tkr, key)
            try:
                payload = http_get_json(url)

                if provider == "alphavantage" and refuse_cache_on_error_payload:
                    err = av_error(payload)
                    if err:
                        # Prefer any cache we already have (even stale) vs caching an error payload
                        if allow_stale_on_error and path.exists():
                            cached = read_json(path)
                            if cached is not None:
                                logger.warning("AV error for %s/%s (%s). Using stale cache.", tkr, key, err)
                                return cached
                        return payload

                # Normal: cache it
                write_json_atomic(path, payload)
                return payload

            except Exception as e:
                if allow_stale_on_error and path.exists():
                    cached = read_json(path)
                    if cached is not None:
                        logger.warning("Fetch failed for %s/%s (%s). Using stale cache.", tkr, key, e)
                        return cached
                raise

        # ------------------------------ deterministic payload ------------------------------
        warnings_list: List[str] = []
        sources: Dict[str, Any] = {}
        coverage: Dict[str, bool] = {}

        deterministic: Dict[str, Any] = {
            "ticker": ticker,
            "pulled_at": pulled_at,
            "market_snapshot": {},
            "price_history": {},
            "corporate_actions": {},
            "capital_structure": {},
            "financials": {},
            "derived_metrics": {},
            "filings_index": {},
            "sanity_checks": {},
        }

        # ------------------------------ 1) Quote ------------------------------
        try:
            url = av_url("GLOBAL_QUOTE", symbol=ticker)
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="GLOBAL_QUOTE", url=url)
            err = av_error(payload)
            sources["alphavantage_global_quote"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage GLOBAL_QUOTE: {err}")
                coverage["market_snapshot"] = False
            else:
                gq = payload.get("Global Quote") or payload.get("Global quote") or {}
                price = safe_float(gq.get("05. price") or gq.get("price"))
                prev_close = safe_float(gq.get("08. previous close"))
                open_ = safe_float(gq.get("02. open"))
                high = safe_float(gq.get("03. high"))
                low = safe_float(gq.get("04. low"))
                volume = safe_int(gq.get("06. volume"))
                latest_trading_day = gq.get("07. latest trading day")

                deterministic["market_snapshot"] = {
                    "price": price,
                    "previous_close": prev_close,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "latest_trading_day": latest_trading_day,
                    "as_of": latest_trading_day or pulled_at,
                    "source": {"provider": "Alpha Vantage", "endpoint": "GLOBAL_QUOTE", "url": url},
                }
                state["current_price"] = price
                coverage["market_snapshot"] = True

        except Exception as e:
            warnings_list.append(f"GLOBAL_QUOTE failed: {e}")
            coverage["market_snapshot"] = False

        # ------------------------------ 2) Daily adjusted history ------------------------------
        try:
            url = av_url("TIME_SERIES_DAILY_ADJUSTED", symbol=ticker, outputsize="full")
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="TIME_SERIES_DAILY_ADJUSTED", url=url, ttl_fallback_s=60*60*12)
            err = av_error(payload)
            sources["alphavantage_daily_adjusted"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage DAILY_ADJUSTED: {err}")
                coverage["price_history_daily"] = False
            else:
                ts = payload.get("Time Series (Daily)") or {}
                dates = sorted(ts.keys())
                last_dates = dates[-1600:] if len(dates) > 1600 else dates

                daily = []
                for d in last_dates:
                    row = ts.get(d) or {}
                    daily.append(
                        {
                            "date": d,
                            "open": safe_float(row.get("1. open")),
                            "high": safe_float(row.get("2. high")),
                            "low": safe_float(row.get("3. low")),
                            "close": safe_float(row.get("4. close")),
                            "adjusted_close": safe_float(row.get("5. adjusted close")),
                            "volume": safe_int(row.get("6. volume")),
                            "dividend_amount": safe_float(row.get("7. dividend amount")),
                            "split_coefficient": safe_float(row.get("8. split coefficient")),
                        }
                    )

                deterministic["price_history"] = {
                    "daily": daily,
                    "prices_are_split_adjusted": True,
                    "provider": "Alpha Vantage",
                    "endpoint": "TIME_SERIES_DAILY_ADJUSTED",
                    "url": url,
                    "pulled_at": pulled_at,
                    "points": len(daily),
                }
                coverage["price_history_daily"] = True

        except Exception as e:
            warnings_list.append(f"TIME_SERIES_DAILY_ADJUSTED failed: {e}")
            coverage["price_history_daily"] = False

        # ------------------------------ 3) Company overview ------------------------------
        try:
            url = av_url("OVERVIEW", symbol=ticker)
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="OVERVIEW", url=url, ttl_fallback_s=60*60*24)
            err = av_error(payload)
            sources["alphavantage_overview"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage OVERVIEW: {err}")
                coverage["overview"] = False
            else:
                shares = safe_int(payload.get("SharesOutstanding"))
                market_cap = safe_float(payload.get("MarketCapitalization"))
                beta = safe_float(payload.get("Beta"))

                deterministic["capital_structure"] = {
                    "shares_outstanding": shares,
                    "market_cap_reported": market_cap,
                    "beta": beta,
                    "currency": payload.get("Currency"),
                    "exchange": payload.get("Exchange"),
                    "company_name": payload.get("Name"),
                    "sector": payload.get("Sector"),
                    "industry": payload.get("Industry"),
                    "as_of": pulled_at,
                    "source": {"provider": "Alpha Vantage", "endpoint": "OVERVIEW", "url": url},
                }
                coverage["overview"] = True

        except Exception as e:
            warnings_list.append(f"OVERVIEW failed: {e}")
            coverage["overview"] = False

        # ------------------------------ 4) Financial statements ------------------------------
        def normalize_reports(payload: Dict[str, Any]) -> Dict[str, Any]:
            annual = payload.get("annualReports") or []
            quarterly = payload.get("quarterlyReports") or []
            return {
                "annual": annual[:6] if isinstance(annual, list) else [],
                "quarterly": quarterly[:12] if isinstance(quarterly, list) else [],
            }

        # Income
        try:
            url = av_url("INCOME_STATEMENT", symbol=ticker)
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="INCOME_STATEMENT", url=url, ttl_fallback_s=60*60*24)
            err = av_error(payload)
            sources["alphavantage_income_statement"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage INCOME_STATEMENT: {err}")
                coverage["income_statement"] = False
            else:
                deterministic["financials"]["income_statement"] = {
                    **normalize_reports(payload),
                    "source": {"provider": "Alpha Vantage", "endpoint": "INCOME_STATEMENT", "url": url, "pulled_at": pulled_at},
                }
                coverage["income_statement"] = True

        except Exception as e:
            warnings_list.append(f"INCOME_STATEMENT failed: {e}")
            coverage["income_statement"] = False

        # Balance
        try:
            url = av_url("BALANCE_SHEET", symbol=ticker)
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="BALANCE_SHEET", url=url, ttl_fallback_s=60*60*24)
            err = av_error(payload)
            sources["alphavantage_balance_sheet"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage BALANCE_SHEET: {err}")
                coverage["balance_sheet"] = False
            else:
                deterministic["financials"]["balance_sheet"] = {
                    **normalize_reports(payload),
                    "source": {"provider": "Alpha Vantage", "endpoint": "BALANCE_SHEET", "url": url, "pulled_at": pulled_at},
                }
                coverage["balance_sheet"] = True

        except Exception as e:
            warnings_list.append(f"BALANCE_SHEET failed: {e}")
            coverage["balance_sheet"] = False

        # Cash flow
        try:
            url = av_url("CASH_FLOW", symbol=ticker)
            payload = cached_fetch(provider="alphavantage", tkr=ticker, key="CASH_FLOW", url=url, ttl_fallback_s=60*60*24)
            err = av_error(payload)
            sources["alphavantage_cash_flow"] = {"url": url, "pulled_at": pulled_at, "error": err}

            if err:
                warnings_list.append(f"Alpha Vantage CASH_FLOW: {err}")
                coverage["cash_flow"] = False
            else:
                deterministic["financials"]["cash_flow"] = {
                    **normalize_reports(payload),
                    "source": {"provider": "Alpha Vantage", "endpoint": "CASH_FLOW", "url": url, "pulled_at": pulled_at},
                }
                coverage["cash_flow"] = True

        except Exception as e:
            warnings_list.append(f"CASH_FLOW failed: {e}")
            coverage["cash_flow"] = False

        # ------------------------------ 5) Corporate actions (derive from daily adjusted) ------------------------------
        try:
            daily = (deterministic.get("price_history") or {}).get("daily") or []
            splits, dividends = [], []

            for row in daily:
                sc = row.get("split_coefficient")
                da = row.get("dividend_amount")
                if sc is not None and sc != 1.0:
                    splits.append({"date": row.get("date"), "split_coefficient": sc})
                if da is not None and da != 0.0:
                    dividends.append({"date": row.get("date"), "dividend_amount": da})

            deterministic["corporate_actions"] = {
                "splits": splits[:50],
                "dividends": dividends[:200],
                "derived_from": "TIME_SERIES_DAILY_ADJUSTED",
                "as_of": pulled_at,
            }
            coverage["corporate_actions"] = True

        except Exception as e:
            warnings_list.append(f"Corporate actions derivation failed: {e}")
            coverage["corporate_actions"] = False

        # ------------------------------ 6) Derived metrics ------------------------------
        try:
            price = (deterministic.get("market_snapshot") or {}).get("price")
            shares_out = (deterministic.get("capital_structure") or {}).get("shares_outstanding")
            mc_calc = (float(price) * float(shares_out)) if (price is not None and shares_out is not None) else None

            bal_q = ((deterministic.get("financials") or {}).get("balance_sheet") or {}).get("quarterly") or []
            latest_bal = bal_q[0] if bal_q else {}

            cash = safe_float(latest_bal.get("cashAndCashEquivalentsAtCarryingValue"))
            st_debt = safe_float(latest_bal.get("shortTermDebt"))
            lt_debt = safe_float(latest_bal.get("longTermDebt"))
            total_debt = (float(st_debt or 0.0) + float(lt_debt or 0.0)) if (st_debt is not None or lt_debt is not None) else None
            net_debt = (float(total_debt) - float(cash)) if (total_debt is not None and cash is not None) else None

            cf_q = ((deterministic.get("financials") or {}).get("cash_flow") or {}).get("quarterly") or []
            ocf_ttm = capex_ttm = fcf_ttm = None
            if len(cf_q) >= 4:
                ocf_vals, capex_vals = [], []
                for i in range(4):
                    ocf_vals.append(safe_float(cf_q[i].get("operatingCashflow")) or 0.0)
                    capex_vals.append(safe_float(cf_q[i].get("capitalExpenditures")) or 0.0)
                ocf_ttm = float(sum(ocf_vals))
                capex_ttm = float(sum(capex_vals))
                fcf_ttm = ocf_ttm + capex_ttm  # capex often negative

            deterministic["derived_metrics"] = {
                "market_cap_calculated": mc_calc,
                "cash_and_equivalents": cash,
                "total_debt": total_debt,
                "net_debt": net_debt,
                "operating_cashflow_ttm": ocf_ttm,
                "capex_ttm": capex_ttm,
                "free_cash_flow_ttm": fcf_ttm,
                "as_of": pulled_at,
                "notes": [
                    "market_cap_calculated = price * shares_outstanding (if both available)",
                    "TTM values are the sum of the most recent 4 quarterly reports (if present)",
                    "FCF uses AlphaVantage sign convention: FCF = OCF + CapEx",
                ],
            }
            coverage["derived_metrics"] = True

        except Exception as e:
            warnings_list.append(f"Derived metrics failed: {e}")
            coverage["derived_metrics"] = False

        # ------------------------------ 7) SEC filings index (with cache) ------------------------------
        def sec_map_ticker_to_cik(tkr: str) -> Optional[int]:
            url = "https://www.sec.gov/files/company_tickers.json"
            payload = cached_fetch(provider="sec", tkr="__GLOBAL__", key="company_tickers", url=url, ttl_fallback_s=60*60*24*7)

            if not isinstance(payload, dict):
                return None

            tkr_u = tkr.upper()
            for _, row in payload.items():
                if str(row.get("ticker", "")).upper() == tkr_u:
                    cik = row.get("cik_str")
                    try:
                        return int(cik)
                    except Exception:
                        return None
            return None

        def sec_recent_filings(cik: int) -> Dict[str, Any]:
            cik10 = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik10}.json"

            # Cache keyed by ticker + "submissions" so you get what you asked for (ticker-level caching),
            # but the filename still stays stable.
            payload = cached_fetch(provider="sec", tkr=ticker, key="submissions", url=url, ttl_fallback_s=60*60*12)

            out = {
                "cik": cik,
                "company_name": payload.get("name") if isinstance(payload, dict) else None,
                "sic": payload.get("sic") if isinstance(payload, dict) else None,
                "fiscal_year_end": payload.get("fiscalYearEnd") if isinstance(payload, dict) else None,
                "filings": {"latest_10k": None, "latest_10q": None, "recent_8k": []},
                "source": {"provider": "SEC EDGAR submissions", "url": url, "pulled_at": pulled_at},
            }

            recent = (((payload.get("filings") or {}).get("recent")) or {}) if isinstance(payload, dict) else {}
            forms = recent.get("form") or []
            accession = recent.get("accessionNumber") or []
            filing_dates = recent.get("filingDate") or []
            primary_docs = recent.get("primaryDocument") or []

            n = min(len(forms), len(accession), len(filing_dates), len(primary_docs))
            items = []
            for i in range(n):
                items.append(
                    {
                        "form": forms[i],
                        "accessionNumber": accession[i],
                        "filingDate": filing_dates[i],
                        "primaryDocument": primary_docs[i],
                    }
                )

            def doc_url(cik_int: int, acc: str, primary: str) -> str:
                return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/{primary}"

            for item in items:
                f = str(item.get("form", "")).upper()
                if out["filings"]["latest_10k"] is None and f == "10-K":
                    out["filings"]["latest_10k"] = {
                        "form": "10-K",
                        "filing_date": item.get("filingDate"),
                        "accession": item.get("accessionNumber"),
                        "primary_document": item.get("primaryDocument"),
                        "primary_url": doc_url(cik, item.get("accessionNumber", ""), item.get("primaryDocument", "")),
                    }
                if out["filings"]["latest_10q"] is None and f == "10-Q":
                    out["filings"]["latest_10q"] = {
                        "form": "10-Q",
                        "filing_date": item.get("filingDate"),
                        "accession": item.get("accessionNumber"),
                        "primary_document": item.get("primaryDocument"),
                        "primary_url": doc_url(cik, item.get("accessionNumber", ""), item.get("primaryDocument", "")),
                    }
                if f == "8-K" and len(out["filings"]["recent_8k"]) < 10:
                    out["filings"]["recent_8k"].append(
                        {
                            "form": "8-K",
                            "filing_date": item.get("filingDate"),
                            "accession": item.get("accessionNumber"),
                            "primary_document": item.get("primaryDocument"),
                            "primary_url": doc_url(cik, item.get("accessionNumber", ""), item.get("primaryDocument", "")),
                        }
                    )

                if out["filings"]["latest_10k"] and out["filings"]["latest_10q"] and len(out["filings"]["recent_8k"]) >= 5:
                    break

            return out

        try:
            cik = sec_map_ticker_to_cik(ticker)
            if not cik:
                warnings_list.append("SEC filings: could not map ticker to CIK")
                coverage["sec_filings"] = False
            else:
                sec_out = sec_recent_filings(cik)
                deterministic["filings_index"] = sec_out
                sources["sec_filings"] = sec_out.get("source")
                coverage["sec_filings"] = True

        except Exception as e:
            warnings_list.append(f"SEC filings pull failed: {e}")
            coverage["sec_filings"] = False

        # ------------------------------ 8) Sanity checks ------------------------------
        try:
            issues = []
            price = (deterministic.get("market_snapshot") or {}).get("price")
            shares_out = (deterministic.get("capital_structure") or {}).get("shares_outstanding")
            mc_rep = (deterministic.get("capital_structure") or {}).get("market_cap_reported")
            mc_calc = (deterministic.get("derived_metrics") or {}).get("market_cap_calculated")

            if price is None:
                issues.append("missing current price")
            if shares_out is None:
                issues.append("missing shares outstanding")
            if mc_rep is not None and mc_calc is not None and mc_rep > 0:
                if abs(mc_rep - mc_calc) / mc_rep > 0.25:
                    issues.append("market cap mismatch >25% (reported vs calculated)")

            deterministic["sanity_checks"] = {"ok": len(issues) == 0, "issues": issues, "as_of": pulled_at}

        except Exception as e:
            warnings_list.append(f"Sanity checks failed: {e}")

        # ------------------------------ finalize ------------------------------
        deterministic_meta = {
            "ticker": ticker,
            "pulled_at": pulled_at,
            "sources": sources,
            "coverage": coverage,
            "warnings": warnings_list,
            "cache_root": str(cache_root),
        }

        out = {**state, "deterministic": deterministic, "deterministic_meta": deterministic_meta}

        if out.get("current_price") is None:
            out["current_price"] = (deterministic.get("market_snapshot") or {}).get("price")

        return progress_mark(
            out,
            "data_analyst",
            {
                "ticker": ticker,
                "det_ok": True,
                "det_warnings": len(warnings_list),
                "det_coverage_true": sum(1 for v in coverage.values() if v),
                "det_coverage_total": len(coverage),
            },
        )


def news_fetcher(state: AgentState) -> AgentState:
    """
    Fetch and cache recent company news for the ticker using Finnhub.

    Writes:
      - news: normalized Finnhub payload
      - news_meta: counts + cache hit/stale flags
    """

    with log_timing("news_fetcher"):
        state = progress_init(dict(state))
        ticker = (state.get("ticker") or "").strip().upper()

        if not ticker:
            out = {**state, "news": {"articles": [], "error": "missing ticker"}}
            return progress_mark(out, "news_fetcher", {"news_ok": False})

        if not finnhub_api_key:
            out = {**state, "news": {"articles": [], "error": "missing FINNHUB_API_KEY"}}
            return progress_mark(out, "news_fetcher", {"news_ok": False})

        news = fetch_finnhub_company_news_last_12m(
            ticker=ticker,
            finnhub_api_key=finnhub_api_key,
            cache_root=script_dir / "cache",
        )

        articles = news.get("articles") or []
        cache = (news.get("source") or {}).get("cache") or {}

        # This is the line you want every run:
        logger.info(
            "NEWS | ticker=%s | n=%d | cache_hit=%s | stale=%s | error=%s",
            ticker,
            len(articles),
            cache.get("hit"),
            cache.get("stale"),
            news.get("error"),
        )

        out = {
            **state,
            "news": news,
            "news_meta": {
                "article_count": len(articles),
                "has_error": bool(news.get("error")),
                "cache_hit": bool(cache.get("hit")),
                "cache_path": cache.get("path"),
                "stale_cache": bool(cache.get("stale")),
            },
        }
        return progress_mark(out, "news_fetcher", {"news_ok": not bool(news.get("error")), "news_n": len(articles)})


def archiver(state: AgentState) -> AgentState:
    """
    Build or incrementally update local vector indexes for retrieval.

    Creates two indexes under cache/vector/<TICKER>/:
      - news/: article headline+summary chunks w/ metadata (url, published_at, etc.)
      - deterministic/: a stable "latest facts" doc derived from deterministic payload

    Uses per-doc hashes in a manifest to avoid re-embedding unchanged content.
    """

    with log_timing("archiver"):
        state = progress_init(dict(state))

        ticker = (state.get("ticker") or "").strip().upper()
        if not ticker:
            out = {**state, "index_meta": {"error": "missing ticker"}}
            return progress_mark(out, "archiver", {"index_ok": False})

        base_dir = script_dir / "cache" / "vector" / ticker
        news_dir = base_dir / "news"
        det_dir = base_dir / "deterministic"
        news_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def sha(s: str) -> str:
            return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

        def load_manifest(path: Path) -> Dict[str, str]:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def save_manifest(path: Path, data: Dict[str, str]) -> None:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

        def try_load_index(persist_dir: Path) -> Optional[VectorStoreIndex]:
            try:
                sc = StorageContext.from_defaults(persist_dir=str(persist_dir))
                return load_index_from_storage(sc)
            except Exception:
                return None

        splitter = SentenceSplitter(chunk_size=900, chunk_overlap=120)

        def article_doc(a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            url = (a.get("url") or "").strip()
            title = (a.get("headline") or "").strip()
            summary = (a.get("summary") or "").strip()
            source = (a.get("source") or "").strip()

            try:
                ts_i = int(a.get("datetime")) if a.get("datetime") is not None else None
            except Exception:
                ts_i = None

            if not title and not summary:
                return None

            published_at = (
                datetime.fromtimestamp(ts_i, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts_i else None
            )

            text = "\n".join([x for x in (title, summary) if x])
            doc_id = sha(url) if url else sha(f"{title}|{published_at or ''}")

            doc_hash = sha(
                json.dumps(
                    {"title": title, "summary": summary, "url": url, "published_at": published_at},
                    sort_keys=True,
                )
            )

            meta = {
                "source_type": "news",
                "provider": "Finnhub",
                "ticker": ticker,
                "title": title,
                "url": url,
                "source": source,
                "published_at": published_at,
                "published_ts": ts_i,
                "doc_id": doc_id,
                "doc_hash": doc_hash,
                "retrieved_at": now_iso,
            }
            return {"text": text, "metadata": meta}

        def deterministic_facts(det: Dict[str, Any]) -> Dict[str, Any]:
            facts: List[str] = []
            ms = det.get("market_snapshot") or {}
            dm = det.get("derived_metrics") or {}
            cs = det.get("capital_structure") or {}
            filings = ((det.get("filings_index") or {}).get("filings")) or {}

            def add(k: str, v: Any) -> None:
                if v is None:
                    return
                s = str(v).strip()
                if not s or s.lower() in {"none", "null", "nan"}:
                    return
                facts.append(f"{k}: {s}")

            add("price", ms.get("price"))
            add("previous_close", ms.get("previous_close"))
            add("volume", ms.get("volume"))
            add("as_of", ms.get("as_of"))

            add("shares_outstanding", cs.get("shares_outstanding"))
            add("market_cap_reported", cs.get("market_cap_reported"))
            add("market_cap_calculated", dm.get("market_cap_calculated"))
            add("net_debt", dm.get("net_debt"))
            add("free_cash_flow_ttm", dm.get("free_cash_flow_ttm"))

            if filings.get("latest_10k"):
                add("latest_10k_filing_date", filings["latest_10k"].get("filing_date"))
                add("latest_10k_url", filings["latest_10k"].get("primary_url"))

            if filings.get("latest_10q"):
                add("latest_10q_filing_date", filings["latest_10q"].get("filing_date"))
                add("latest_10q_url", filings["latest_10q"].get("primary_url"))

            as_of = det.get("pulled_at") or now_iso
            text = "\n".join(facts) if facts else "No deterministic facts were available."

            # stable id so you update in-place instead of accumulating a new deterministic doc every run
            doc_id = sha(f"det::{ticker}::latest")
            doc_hash = sha(text)

            meta = {
                "source_type": "deterministic",
                "provider": "AlphaVantage+SEC",
                "ticker": ticker,
                "as_of": as_of,
                "doc_id": doc_id,
                "doc_hash": doc_hash,
                "retrieved_at": now_iso,
            }
            return {"text": text, "metadata": meta}


        def to_nodes(doc: Dict[str, Any], prefix: str) -> List[TextNode]:
            text = doc["text"]
            meta = doc["metadata"]

            chunks = splitter.split_text(text) if text else []
            if not chunks:
                chunks = [""]

            base_id = meta.get("doc_id") or sha(meta.get("title", "") + "|" + meta.get("url", ""))

            nodes: List[TextNode] = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{prefix}::{ticker}::{base_id}::{i}"
                m = dict(meta)
                m["chunk_index"] = i
                m["chunk_id"] = chunk_id
                nodes.append(TextNode(text=chunk, metadata=m, id_=chunk_id))
            return nodes

        news_articles = (state.get("news") or {}).get("articles") or []
        news_docs = [d for d in (article_doc(a) for a in news_articles) if d is not None]

        det_payload = state.get("deterministic") or {}
        det_doc = deterministic_facts(det_payload)

        news_manifest_path = news_dir / "manifest.json"
        det_manifest_path = det_dir / "manifest.json"
        news_manifest = load_manifest(news_manifest_path)
        det_manifest = load_manifest(det_manifest_path)

        def changed(docs: List[Dict[str, Any]], manifest: Dict[str, str]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for d in docs:
                doc_id = d["metadata"].get("doc_id")
                doc_hash = d["metadata"].get("doc_hash")
                if not doc_id or not doc_hash:
                    out.append(d)
                    continue
                if manifest.get(doc_id) != doc_hash:
                    out.append(d)
            return out

        news_changed = changed(news_docs, news_manifest)
        det_changed = changed([det_doc], det_manifest)

        news_nodes: List[TextNode] = []
        for d in news_changed:
            news_nodes.extend(to_nodes(d, "news"))

        det_nodes: List[TextNode] = []
        for d in det_changed:
            det_nodes.extend(to_nodes(d, "det"))

        news_index = try_load_index(news_dir)
        det_index = try_load_index(det_dir)

        def upsert(persist_dir: Path, index: Optional[VectorStoreIndex], nodes: List[TextNode]) -> VectorStoreIndex:
            if index is None:
                index = VectorStoreIndex(nodes=nodes)
            else:
                index.insert_nodes(nodes)
            index.storage_context.persist(persist_dir=str(persist_dir))
            return index

        if news_nodes:
            news_index = upsert(news_dir, news_index, news_nodes)
            for d in news_changed:
                news_manifest[d["metadata"]["doc_id"]] = d["metadata"]["doc_hash"]
            save_manifest(news_manifest_path, news_manifest)

        if det_nodes:
            det_index = upsert(det_dir, det_index, det_nodes)
            for d in det_changed:
                det_manifest[d["metadata"]["doc_id"]] = d["metadata"]["doc_hash"]
            save_manifest(det_manifest_path, det_manifest)

        logger.info(
            "INDEX | ticker=%s | news_docs=%d upserted=%d nodes=%d | det_upserted=%d det_nodes=%d",
            ticker,
            len(news_docs),
            len(news_changed),
            len(news_nodes),
            len(det_changed),
            len(det_nodes),
        )

        out = {
            **state,
            "index_meta": {
                "ticker": ticker,
                "news": {
                    "persist_dir": str(news_dir),
                    "docs_total": len(news_docs),
                    "docs_upserted": len(news_changed),
                    "nodes_upserted": len(news_nodes),
                },
                "deterministic": {
                    "persist_dir": str(det_dir),
                    "docs_total": 1,
                    "docs_upserted": len(det_changed),
                    "nodes_upserted": len(det_nodes),
                },
            },
        }

        ok = bool(news_index or det_index)
        return progress_mark(out, "archiver", {"index_ok": ok, "news_nodes": len(news_nodes), "det_nodes": len(det_nodes)})


def searcher(state: AgentState) -> AgentState:
    """
    Retrieve relevant evidence chunks from the local vector indexes.

    For each query generated by orchestrator:
      - retrieve top-k from deterministic index (small)
      - retrieve top-k from news index (slightly larger)
      - filter old news outside NEWS_LOOKBACK_DAYS
      - dedupe by chunk_id
      - sort by similarity score

    Writes:
      - evidence: list of chunks w/ metadata and score
      - evidence_meta: summary of retrieval coverage
    """

    with log_timing("searcher"):
        state = progress_init(dict(state))

        ticker = (state.get("ticker") or "").strip().upper()
        if not ticker:
            out = {**state, "evidence": [], "evidence_meta": {"error": "missing ticker"}}
            return progress_mark(out, "searcher", {"evidence_ok": False})

        queries = state.get("queries") or []
        if not queries:
            out = {**state, "evidence": [], "evidence_meta": {"error": "no queries"}}
            return progress_mark(out, "searcher", {"evidence_ok": False})

        base_dir = script_dir / "cache" / "vector" / ticker
        news_dir = base_dir / "news"
        det_dir = base_dir / "deterministic"

        def load_index(persist_dir: Path) -> Optional[VectorStoreIndex]:
            try:
                sc = StorageContext.from_defaults(persist_dir=str(persist_dir))
                return load_index_from_storage(sc)
            except Exception:
                return None

        news_index = load_index(news_dir)
        det_index = load_index(det_dir)

        if news_index is None and det_index is None:
            out = {**state, "evidence": [], "evidence_meta": {"error": "no indexes found"}}
            return progress_mark(out, "searcher", {"evidence_ok": False})

        def collect(index: VectorStoreIndex, q: str, *, top_k: int) -> List[Dict[str, Any]]:
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(q) or []

            out_rows: List[Dict[str, Any]] = []
            for n in nodes:
                node = getattr(n, "node", n)
                meta = dict(getattr(node, "metadata", {}) or {})
                if meta.get("source_type") == "news":
                    ts_i = meta.get("published_ts")
                    if isinstance(ts_i, int):
                        cutoff = int((time.time() - (NEWS_LOOKBACK_DAYS * 86400)))
                        if ts_i < cutoff:
                            continue
                chunk_id = meta.get("chunk_id") or getattr(node, "id_", None) or getattr(node, "node_id", None)

                out_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "score": float(getattr(n, "score", 0.0) or 0.0),
                        "text": (getattr(node, "text", "") or "").strip(),
                        "metadata": meta,
                        "query": q,
                    }
                )
            return out_rows

        evidence: List[Dict[str, Any]] = []
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue

            if det_index is not None:
                evidence.extend(collect(det_index, q, top_k=3))
            if news_index is not None:
                evidence.extend(collect(news_index, q, top_k=5))

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in evidence:
            cid = row.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            deduped.append(row)

        deduped.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

        out = {
            **state,
            "evidence": deduped,
            "evidence_meta": {
                "ticker": ticker,
                "queries": len(queries),
                "chunks": len(deduped),
                "has_news_index": bool(news_index),
                "has_det_index": bool(det_index),
            },
        }
        return progress_mark(out, "searcher", {"evidence_ok": True, "evidence_chunks": len(deduped)})


def advisor(state: AgentState) -> AgentState:
    """
    Write the due diligence report using an LLM with structured output.

    Inputs:
      - deterministic snapshot (compact facts)
      - evidence chunks from retrieval

    Constraints enforced in the system prompt:
      - only use provided evidence/facts
      - do not guess missing info
      - cite claims using inline [chunk_id]

    Post-processing:
      - convert [chunk_id] citations to readable [[Source: ...]](url) links
    """

    with log_timing("advisor"):
        state = progress_init(dict(state))

        ticker = (state.get("ticker") or "").strip().upper()
        if not ticker:
            out = {**state, "report": {"error": "missing ticker"}}
            return progress_mark(out, "advisor", {"report_ok": False})

        det = state.get("deterministic") or {}
        evidence = state.get("evidence") or []

        det_meta = state.get("deterministic_meta") or {}
        coverage = det_meta.get("coverage") or {}
        warnings_list = det_meta.get("warnings") or []

        def compact_market_snapshot(d: Dict[str, Any]) -> Dict[str, Any]:
            ms = d.get("market_snapshot") or {}
            cs = d.get("capital_structure") or {}
            dm = d.get("derived_metrics") or {}
            return {
                "price": ms.get("price"),
                "previous_close": ms.get("previous_close"),
                "volume": ms.get("volume"),
                "as_of": ms.get("as_of"),
                "company_name": cs.get("company_name"),
                "sector": cs.get("sector"),
                "industry": cs.get("industry"),
                "shares_outstanding": cs.get("shares_outstanding"),
                "market_cap_reported": cs.get("market_cap_reported"),
                "market_cap_calculated": dm.get("market_cap_calculated"),
                "net_debt": dm.get("net_debt"),
                "free_cash_flow_ttm": dm.get("free_cash_flow_ttm"),
            }

        snapshot = compact_market_snapshot(det)

        def shrink_chunks(rows: List[Dict[str, Any]], max_items: int = 35, max_chars: int = 900) -> List[Dict[str, Any]]:
            out_rows = []
            for r in rows[:max_items]:
                txt = (r.get("text") or "").strip()
                if len(txt) > max_chars:
                    txt = txt[:max_chars].rstrip() + "…"
                out_rows.append(
                    {
                        "chunk_id": r.get("chunk_id"),
                        "source_type": ((r.get("metadata") or {}).get("source_type") or ""),
                        "published_at": ((r.get("metadata") or {}).get("published_at") or ""),
                        "title": ((r.get("metadata") or {}).get("title") or ""),
                        "url": ((r.get("metadata") or {}).get("url") or ""),
                        "text": txt,
                    }
                )
            return out_rows

        evidence_compact = shrink_chunks(evidence, max_items=35, max_chars=900)

        llm = get_llm()
        writer = llm.with_structured_output(ReportOutput)

        system = (
            "You are writing a high-level stock due diligence report.\n"
            "You MUST base the report only on the provided deterministic facts and evidence chunks.\n"
            "If information is missing, say it is missing. Do not guess.\n"
            "\n"
            "CITATIONS:\n"
            "- Any claim that comes from evidence chunks must include an inline citation like: [chunk_id].\n"
            "- Do not cite anything you did not see in the evidence.\n"
            "\n"
            "RATING:\n"
            "- Pick exactly one: Buy / Hold / Sell.\n"
            "- If evidence is thin or mixed, default to Hold with lower confidence.\n"
            "\n"
            "OUTPUT:\n"
            "- Return structured output only."
        )

        user_payload = {
            "ticker": ticker,
            "user_question": state.get("question") or "",
            "deterministic_snapshot": snapshot,
            "coverage": coverage,
            "warnings": warnings_list[:10],
            "evidence_chunks": evidence_compact,
            "report_format": [
                "## Executive summary",
                "## Snapshot",
                "## Market sentiment (news-driven)",
                "## Financial quality (facts-driven)",
                "## Risks",
                "## Conclusion",
            ],
        }

        result: ReportOutput = invoke_with_backoff(
            writer,
            [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
            ],
            step_name="advisor",
            max_retries=LLM_MAX_RETRIES,
            base_sleep_s=LLM_BACKOFF_BASE_SLEEP_S,
            max_sleep_s=LLM_BACKOFF_MAX_SLEEP_S,
        )

        citation_map = build_citation_map(state.get("evidence") or [])
        pretty_md = replace_citations(result.report_markdown, citation_map)

        out = {
            **state,
            "report": {
                **result.model_dump(),
                "report_markdown": pretty_md,
                "citation_map": citation_map,
            },
        }
        return progress_mark(out, "advisor", {"report_ok": True, "rating": result.rating, "confidence": result.confidence})


def validator(state: AgentState) -> AgentState:
    """
    Lightweight citation audit for the final markdown report.

    Checks each non-empty, non-heading line and flags it VALID if it contains
    at least one source URL that exists in the report's citation_map.

    Appends a 'Claim Audit' section to the report markdown with totals and
    per-line validity.
    """

    with log_timing("validator"):
        state = progress_init(dict(state))

        report = state.get("report") or {}
        md = (report.get("report_markdown") or "").strip()
        cmap = report.get("citation_map") or {}

        if not md:
            out = {**state, "validation": {"error": "missing report_markdown"}}
            return progress_mark(out, "validator", {"valid_ok": False})

        valid_urls = {
            (v.get("url") or "").strip()
            for v in cmap.values()
            if (v.get("url") or "").strip()
        }

        def extract_urls(line: str) -> List[str]:
            return re.findall(r"\[\[Source:[^\]]+\]\]\((https?://[^)]+)\)", line)
        
        def is_structural_line(s: str) -> bool:
            # Section lead-ins like "Key risks include:" or "Valuation outlook:"
            if s.endswith(":") and "http" not in s:
                return True

            # Very short glue lines that aren't real factual claims
            if len(s) < 30 and "Source:" not in s:
                return True

            return False       

        # Validate per-line (cleaner than sentence splitting for markdown)
        candidate_lines: List[str] = []
        for ln in md.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if is_structural_line(s):
                continue

            candidate_lines.append(s)

        audit_rows: List[Dict[str, Any]] = []
        valid_count = 0

        for line in candidate_lines:
            urls = extract_urls(line)
            is_valid = any(u in valid_urls for u in urls)

            # Accept deterministic sources even if they don't have URLs
            if not is_valid:
                if "Source: AlphaVantage+SEC" in line or "Source: Deterministic" in line:
                    is_valid = True
            audit_rows.append({"claim": line, "sourced": bool(is_valid), "urls": urls})
            if is_valid:
                valid_count += 1

        total = len(audit_rows)
        pct = (100.0 * valid_count / total) if total else 0.0

        audit_md_lines: List[str] = []
        audit_md_lines.append("\n---\n")
        audit_md_lines.append("## Claim Audit\n")
        audit_md_lines.append(f"- Claims checked: {total}\n")
        audit_md_lines.append(f"- Claims sourced: {valid_count} ({pct:.1f}%)\n")
        audit_md_lines.append("\n### Claims\n")

        for i, row in enumerate(audit_rows, start=1):
            status = "SOURCED" if row["sourced"] else "UNSOURCED"
            audit_md_lines.append(f"{i}. **{status}** — {row['claim']}\n")

        new_md = md.rstrip() + "\n" + "".join(audit_md_lines)

        out = {
            **state,
            "report": {**report, "report_markdown": new_md},
            "validation": {
                "claims_total": total,
                "claims_valid": valid_count,
                "pct_sourced": pct,
            },
        }
        return progress_mark(out, "validator", {"valid_ok": True, "claims_total": total, "claims_valid": valid_count})



# ---------------------------------- Build Graph ----------------------------------
logger.info("Building LangGraph pipeline")

builder = StateGraph(AgentState)


# ---------------------------------- Register Nodes ----------------------------------
builder.add_node("orchestrator", orchestrator)
builder.add_node("clarifier", clarifier)
builder.add_node("data_analyst", data_analyst)
builder.add_node("news_fetcher", news_fetcher)
builder.add_node("archiver", archiver)
builder.add_node("searcher", searcher)
builder.add_node("advisor", advisor)
builder.add_node("validator", validator)


# ---------------------------------- Wire Edges ----------------------------------
builder.set_entry_point("orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"clarifier": "clarifier", "data_analyst": "data_analyst"},
)

builder.add_edge("clarifier", END)
builder.add_edge("data_analyst", "news_fetcher")
builder.add_edge("news_fetcher", "archiver")
builder.add_edge("archiver", "searcher")
builder.add_edge("searcher", "advisor")
builder.add_edge("advisor", "validator")
builder.add_edge("validator", END)

graph = builder.compile()


# ---------------------------------- Initialize ----------------------------------
logger.info("Invoking graph with initial state")

# Initialize 
initial_state = {"question": "How is the NVDA stock?"}
out = graph.invoke(initial_state)


# ---------------------------------- Print Node Outputs ----------------------------------
p = out.get("progress") or {}
logger.info(
    "PIPELINE COMPLETE | final_pct=%s%% | last_stage=%s | nodes_run=%s",
    p.get("pct"),
    p.get("stage"),
    ",".join(p.get("done", [])),
)

# news preview
news = out.get("news") or {}
articles = news.get("articles") or []
cache = (news.get("source") or {}).get("cache") or {}
logger.info("NEWS (final) | n=%d | cache_hit=%s | error=%s", len(articles), cache.get("hit"), news.get("error"))


# ---------------------------------- Save as PDF ----------------------------------
saved_pdf = save_report_pdf(out, script_dir=script_dir)

if saved_pdf:
    logger.info("Saved report PDF: %s", saved_pdf)
