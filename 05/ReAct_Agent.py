# ./ReAct_Agent.py

# ---------------------------------- Imports ----------------------------------
import boto3
from colorama import Fore, Style, init
init(autoreset=True)
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hashlib
import json
from llama_index.core import Document, VectorStoreIndex
from google import genai
import os
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List, Literal, Optional, Type


# ---------------------------------- Config & Env ----------------------------------
LOCATION_HOME = "Durham, NC"
LOCATION_WORK = "Burlington, NC"

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
ENV_PATH = PARENT_DIR / "05.env"

load_dotenv(ENV_PATH)


AWS_REGION = os.getenv("AWS_REGION")
AWS_BASE_MODEL = os.getenv("AWS_BASE_MODEL")
EMBEDDING_MODEL_ID = os.getenv("AWS_EMBEDDING_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


WARDROBE_KNOWLEDGE_BASE = SCRIPT_DIR / "knowledge_base" / "wardrobe.xlsx"


# ------------------------------------ Functions ----------------------------------
def print_banner(text: str):
    banner = f"{'= ' * 10}{text} {'= ' * 10}"
    print("\n" + banner + "\n")


def convert_json_to_toon(json_str: str) -> str:
    data = json.loads(json_str)
    lines = [
        "activity,city,time,temp (fahrenheit),condition",
        f"leaving home,{data['location_home_5am']['location']},{data['location_home_5am']['time']},{data['location_home_5am']['temp_f']},{data['location_home_5am']['condition']}",
        f"arriving work,{data['location_work_6am']['location']},{data['location_work_6am']['time']},{data['location_work_6am']['temp_f']},{data['location_work_6am']['condition']}",
        f"leaving work,{data['location_work_4pm']['location']},{data['location_work_4pm']['time']},{data['location_work_4pm']['temp_f']},{data['location_work_4pm']['condition']}",
        f"arriving home,{data['location_home_3pm']['location']},{data['location_home_3pm']['time']},{data['location_home_3pm']['temp_f']},{data['location_home_3pm']['condition']}"
    ]
    return "\n".join(lines)


def estimate_token_count(formatting_style: str, text: str) -> int:
    """
    Estimates the token count for a given text.
    This is a simple approximation assuming 1 token ~ 4 characters.
    """
    print(
        f"\n{Fore.CYAN}Estimating token count for the following text in {Fore.YELLOW}{formatting_style}{Fore.CYAN} format:{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}The text length is {len(text)} characters.  This is approximately {len(text) / 4:.2f} tokens.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.MAGENTA}The Claude Opus 4.5 model costs $5/million tokens.{Style.RESET_ALL}"
    )
    print(
        f"{Fore.BLUE}So this text would cost approximately ${(len(text) / 4) * (5 / 1_000_000):.6f} to process.{Style.RESET_ALL}"
    )


def read_wardrobe_knowledge_base():
    """
    Reads the wardrobe knowledge base from the Excel file.
    Returns a list of dicts representing the wardrobe items.
    """
    import pandas as pd

    df = pd.read_excel(WARDROBE_KNOWLEDGE_BASE)
    return df.to_dict(orient="records")


def classify_article_bin(df, article_col="Article"):
    """
    Classifies each clothing article as 'Upper', 'Lower', or 'Feet' using nearest neighbor in embedding space.
    Adds a new column 'Bin' to the dataframe.
    """
    # Reference articles for each bin
    reference = {
        "Upper": ["T-Shirt", "Sweater", "Jacket", "Shirt", "Blazer", "Hoodie", "Vest", "Coat"],
        "Lower": ["Pants", "Jeans", "Chinos", "Shorts", "Trousers", "Skirt"],
        "Feet": ["Slippers", "Shoes", "Boots", "Sneakers", "Loafers", "Sandals", "Oxfords"]
    }

    # Flatten reference articles and keep mapping
    ref_articles = []
    ref_bins = []
    for bin_name, articles in reference.items():
        for a in articles:
            ref_articles.append(a.lower())
            ref_bins.append(bin_name)

    # Prepare vectorizer
    vectorizer = TfidfVectorizer().fit(ref_articles + df[article_col].str.lower().tolist())

    # Embed reference articles
    ref_vecs = vectorizer.transform(ref_articles)

    # Classify each article
    bins = []
    for article in df[article_col].astype(str):
        vec = vectorizer.transform([article.lower()])
        sims = cosine_similarity(vec, ref_vecs)[0]
        best_idx = sims.argmax()
        bins.append(ref_bins[best_idx])

    df["Bin"] = bins
    return df


# ---------------------------------- AWS Vector Embeddings ----------------------------------
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


def preview_chunk_text(
    chunk_strings: Dict[str, str],
    bin_name: str,
    *,
    max_lines: int = 15,
    show_chars: bool = True,
) -> None:
    """
    Prints a readable preview of a chunk (including header) without dumping the whole thing.
    """
    if bin_name not in chunk_strings:
        raise KeyError(f"bin_name={bin_name!r} not found. Available: {list(chunk_strings.keys())}")

    text = chunk_strings[bin_name].strip("\n")
    lines = text.splitlines()

    print_banner(f"Chunk Preview: {bin_name}")
    if show_chars:
        print(f"Chars: {len(text):,} | Lines: {len(lines):,}\n")

    # show header + first rows
    for i, line in enumerate(lines[:max_lines], start=1):
        print(f"{i:02d}: {line}")

    if len(lines) > max_lines:
        print(f"\n... ({len(lines) - max_lines} more lines)")


def store_embeddings_llamaindex(records, output_path="wardrobe_llama_index.json"):
    """
    Stores wardrobe chunk embeddings locally using LlamaIndex.
    Each record should have 'id', 'text', 'embedding', and 'metadata'.
    """
    docs = []
    for rec in records:
        # Create a Document with embedding and metadata
        doc = Document(
            text=rec["text"],
            doc_id=rec["id"],
            embedding=rec["embedding"],
            metadata=rec["metadata"]
        )
        docs.append(doc)

    # Build the index
    index = VectorStoreIndex.from_documents(docs)

    # Save index to disk (as JSON)
    index.save_to_disk(output_path)
    print(f"Saved LlamaIndex vector store to {output_path}")


# ---------------------------------- Weather Fetching ----------------------------------
def get_weather_for_locations_tomorrow():
    """
    Fetches weather for LOCATION_HOME at 5 am and LOCATION_WORK at 4 pm for tomorrow.
    Returns a dict with weather info for both locations.
    """
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    results = {}

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    def fetch_weather(location, target_hour):
        params = {
            "key": WEATHER_API_KEY,
            "q": location,
            "days": 2,  # Get today and tomorrow
            "aqi": "no",
            "alerts": "no"
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        # Find the forecast for tomorrow
        for day in data["forecast"]["forecastday"]:
            if day["date"] == tomorrow:
                for hour in day["hour"]:
                    if hour["time"].endswith(f"{target_hour:02d}:00"):
                        return {
                            "location": location,
                            "time": hour["time"],
                            "temp_f": hour["temp_f"],
                            "condition": hour["condition"]["text"]
                        }
        return None

    results["location_home_5am"] = fetch_weather(LOCATION_HOME, 5)
    results["location_work_6am"] = fetch_weather(LOCATION_WORK, 6)
    results["location_work_4pm"] = fetch_weather(LOCATION_WORK, 16)
    results["location_home_3pm"] = fetch_weather(LOCATION_HOME, 15)

    # ---- Print JSON + token estimate ----
    print_banner("Weather (JSON)")
    results_json = json.dumps(results, indent=2)
    print(results_json)
    estimate_token_count("json", results_json)

    # ---- Print Toon + token estimate ----
    print_banner("Weather (Toon)")
    results_toon = convert_json_to_toon(results_json)
    print(results_toon)
    estimate_token_count("toon", results_toon)

    # ---- Savings summary (same math you used before) ----
    json_tokens_est = len(results_json) / 4
    toon_tokens_est = len(results_toon) / 4

    token_saved = json_tokens_est - toon_tokens_est
    pct_saved = (token_saved / json_tokens_est) * 100 if json_tokens_est else 0.0

    print(
        f"\n{Fore.GREEN}By converting from JSON -> Toon, "
        f"estimated token reduction: {token_saved:.2f} tokens ({pct_saved:.2f}%).{Style.RESET_ALL}"
    )

    return results


# ---------------------------------- ReAct Agent ----------------------------------
class ReActAgent:
    def __init__(self, wardrobe_df, get_weather_func):
        self.wardrobe_df = wardrobe_df
        self.get_weather = get_weather_func

    def plan_outfit(self, system_prompt: str):
        print_banner("ReAct Agent Planning")
        print(f"{Fore.YELLOW}System prompt: {system_prompt}{Style.RESET_ALL}")

        # Step 1: Get weather for tomorrow
        weather = self.get_weather()
        print(f"{Fore.CYAN}Weather for tomorrow:{Style.RESET_ALL}")
        for k, v in weather.items():
            print(f"{k}: {v}")

        # Step 2: Filter wardrobe for work-appropriate items
        work_items = self.wardrobe_df[
            self.wardrobe_df["Tags/Metadata"].str.contains("work-appropriate", case=False, na=False)
        ]
        # Pick a work outfit (simple: pick first upper, lower, feet)
        work_upper = work_items[work_items["Bin"] == "Upper"]["Article"].iloc[0] if not work_items[work_items["Bin"] == "Upper"].empty else "No upper found"
        work_lower = work_items[work_items["Bin"] == "Lower"]["Article"].iloc[0] if not work_items[work_items["Bin"] == "Lower"].empty else "No lower found"
        work_feet = work_items[work_items["Bin"] == "Feet"]["Article"].iloc[0] if not work_items[work_items["Bin"] == "Feet"].empty else "No feet found"

        # Step 3: Filter wardrobe for casual (not work-appropriate) items for night
        casual_items = self.wardrobe_df[
            ~self.wardrobe_df["Tags/Metadata"].str.contains("work-appropriate", case=False, na=False)
        ]
        night_upper = casual_items[casual_items["Bin"] == "Upper"]["Article"].iloc[0] if not casual_items[casual_items["Bin"] == "Upper"].empty else "No upper found"
        night_lower = casual_items[casual_items["Bin"] == "Lower"]["Article"].iloc[0] if not casual_items[casual_items["Bin"] == "Lower"].empty else "No lower found"
        night_feet = casual_items[casual_items["Bin"] == "Feet"]["Article"].iloc[0] if not casual_items[casual_items["Bin"] == "Feet"].empty else "No feet found"

        # Step 4: Output plan
        print(f"\n{Fore.GREEN}Work outfit plan for tomorrow morning:{Style.RESET_ALL}")
        print(f"Upper: {work_upper}")
        print(f"Lower: {work_lower}")
        print(f"Feet: {work_feet}")

        print(f"\n{Fore.MAGENTA}Night outfit plan for tomorrow evening:{Style.RESET_ALL}")
        print(f"Upper: {night_upper}")
        print(f"Lower: {night_lower}")
        print(f"Feet: {night_feet}")

        return {
            "work": {"upper": work_upper, "lower": work_lower, "feet": work_feet},
            "night": {"upper": night_upper, "lower": night_lower, "feet": night_feet},
            "weather": weather
        }



# ----------------------------------Main ----------------------------------
# # -------- Fetch Weather --------
# print_banner("Weather Fetching Test")
# print("Fetching weather for tomorrow at specified times...")
# # results = get_weather_for_locations_tomorrow()
# # results = json.dumps(results, indent=2)


# results = {
#   "location_home_5am": {
#     "location": "Durham, NC",
#     "time": "2026-02-05 05:00",
#     "temp_f": 31.7,
#     "condition": "Light freezing rain"
#   },
#   "location_work_6am": {
#     "location": "Burlington, NC",
#     "time": "2026-02-05 06:00",
#     "temp_f": 30.2,
#     "condition": "Light freezing rain"
#   },
#   "location_work_4pm": {
#     "location": "Burlington, NC",
#     "time": "2026-02-05 16:00",
#     "temp_f": 36.9,
#     "condition": "Partly Cloudy "
#   },
#   "location_home_3pm": {
#     "location": "Durham, NC",
#     "time": "2026-02-05 15:00",
#     "temp_f": 37.4,
#     "condition": "Sunny"
#   }
# }


# results_json = json.dumps(results, indent=2)
# print(results_json)
# estimate_token_count("json", results_json)

# # -------- Convert to Toon-style Representation --------
# print_banner("Conversion to Toon-style Representation to Reduce Token Count")
# results_toon = convert_json_to_toon(results_json)

# print(results_toon)
# estimate_token_count("toon", results_toon)

# print(f"\nBy converting from JSON object to Toon-style representation, we reduced the token count by approximately {(len(results_json) / 4) - (len(results_toon) / 4):.2f} tokens.")
# print(f"This is a cost savings of approximately: {(((len(results_json) / 4) - (len(results_toon) / 4)) / (len(results_json) / 4)) * 100:.2f}%.")




# -------- Load Wardrobe Knowledge Base Test --------
print_banner("Load Wardrobe Knowledge Base")
wardrobe_items = read_wardrobe_knowledge_base()
print(f"Loaded {len(wardrobe_items)} wardrobe items from the knowledge base.")
print("Sample items:")
for item in wardrobe_items[:5]:
    print(item)

print("\n")
print(f"Classifying wardrobe items into 'Upper', 'Lower', and 'Feet' bins by using nearest neighbor in embedding space...\n")
df = pd.DataFrame(wardrobe_items)  # Convert list of dicts to DataFrame
df = classify_article_bin(df)
print(df[["Article", "Bin"]])


# -------- Chunking Wardrobe Knowledge Base --------
print_banner("Wardrobe Knowledge Base Chunking")
print("\nChunking strategy: segregating table chunks by 'Bin' category while maintaining the table header.\n")
# Chunk by 'Bin'
bins = ["Upper", "Lower", "Feet"]
chunks = {bin_name: df[df["Bin"] == bin_name] for bin_name in bins}

# Convert each chunk to CSV string
chunk_strings = {}
for bin_name, chunk_df in chunks.items():
    chunk_strings[bin_name] = chunk_df.to_csv(index=False)

for bin_name, chunk_str in chunk_strings.items():
    print("= = = = = " + f"Chunk for Bin: {bin_name}" + " = = = = =")
    print(chunk_str)
    estimate_token_count("toon", chunk_str)
    print("\n")


# -------- Embedding Wardrobe Knowledge Base Chunks with AWS Bedrock --------
print_banner("Embedding Wardrobe Knowledge Base Chunks with AWS Bedrock")

records = embed_unique_chunks_bedrock(
    chunk_strings,
    region=AWS_REGION,
    embedding_model_id=EMBEDDING_MODEL_ID,
    kb_name="wardrobe knowledge base"
)

print(f"Embedded {len(records)} unique chunks.")
print(records[0]["id"], records[0]["metadata"].keys())

print("\nSample embedded chunk metadata:")
for record in records:
    print(f"ID: {record['id']}")
    print(f"Metadata: {record['metadata']}")
    print(f"Embedding (first 5 values): {record['embedding'][:5]}")
    print("-----")
    break  # Just show one sample   

print("\nPreviewing chunk texts:")
preview_chunk_text(chunk_strings, "Upper", max_lines=20)

# -------- Store Embeddings with LlamaIndex --------
print_banner("Storing Embeddings with LlamaIndex")
store_embeddings_llamaindex(records, "wardrobe_llama_index.json")




# # -------- ReAct Agent Outfit Planning --------
agent = ReActAgent(df, get_weather_for_locations_tomorrow)
system_prompt = "Plan my outfits for tomorrow: work-appropriate for work, casual for night."
agent.plan_outfit(system_prompt)