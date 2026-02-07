# ./ReAct Agent.py

# ---------------------------------- Imports ----------------------------------
import boto3
from botocore.exceptions import ClientError
from colorama import Fore, Style, init
init(autoreset=True)
from datetime import datetime, timedelta
from docx import Document as DocxDocument
from dotenv import load_dotenv
import json
import os
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
import random
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import socket
import time
from typing import Any, Callable, Dict, Literal, Optional, Type

from employee_handbook_kb import (
    chunk_employee_handbook,
    embed_handbook_chunks,
    search_handbook_chunks,
)

from wardrobe_kb import (
    classify_article_bin,
    evaluate_binning_confusion,
    print_banner,
    read_wardrobe_knowledge_base,
    wardrobe_json_to_toon,
)



# ---------------------------------- Config & Env ----------------------------------
LOCATION_HOME = "Durham, NC"
LOCATION_WORK = "Burlington, NC"

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

ipaddress = socket.gethostbyname(socket.gethostname())
if (ipaddress == "192.168.0.102"):
    ENV_PATH = PARENT_DIR / "05.env"
else:
    ENV_PATH = SCRIPT_DIR / ".env"

load_dotenv(ENV_PATH)

# AWS Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION")
AWS_BASE_MODEL = os.getenv("AWS_BASE_MODEL")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
EMBEDDING_MODEL_ID = os.getenv("AWS_EMBEDDING_MODEL_ID")

# Weather API Key
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Detect running under VS Code (used to auto-enable HTML export)
RUNNING_UNDER_VSCODE = bool(os.getenv("VSCODE_PID") or os.getenv("TERM_PROGRAM", "").lower() == "vscode")

# Knowledge base paths
WARDROBE_KNOWLEDGE_BASE = SCRIPT_DIR / "knowledge_base" / "wardrobe.xlsx"
EMPLOYEE_HANDBOOK = SCRIPT_DIR / "knowledge_base" / "employee_handbook.docx"



# ---------------------------------- Utilities ----------------------------------
def safe_json_dumps(obj: Any, *, max_chars: int = 2500) -> str:
    # Try to convert to JSON for pretty printing to the console, but if it fails, 
    # just convert to string.
    try:
        pretty_string = json.dumps(obj, indent=2, default=str)
    except Exception:
        pretty_string = str(obj)
    if len(pretty_string) > max_chars:
        return pretty_string[:max_chars] + "\n... (truncated)"
    return pretty_string


def estimate_token_count(formatting_style: str, text: str) -> int:
    """
    Estimates the token count for a given text assuming 1 token is about 4 characters.
    Also print out cost estimates for processing with Claude Opus 4.5.
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

    return int(len(text) / 4)


def normalize_action_name(action_text: str) -> str:
    """
    Normalize whatever action the LLM writes as an "Action" into one of the real tool names.
    This is for easier parsing and to allow the model some flexibility in how it refers to the tools.
    This is because LLMs are sloppy with tool naming.  This acts as a tolerance layer.  
    """
    llm_action = (action_text or "").strip()
    llm_action = re.sub(r"^Action\s*:\s*", "", llm_action, flags=re.IGNORECASE).strip()
    llm_action = re.sub(r"^call\s+", "", llm_action, flags=re.IGNORECASE).strip()
    llm_action = re.sub(r"\(.*\)\s*$", "", llm_action).strip()

    lower = llm_action.lower()
    if "weather" in lower:
        return "get_weather"
    if "wardrobe" in lower or "vector" in lower or "clothes" in lower:
        return "search_wardrobe"
    if "policy" in lower or "dress code" in lower or "handbook" in lower:
        return "search_handbook"

    return llm_action



# ---------------------------------- Presentation Layer ----------------------------------
def print_outfit_presentation(outfit: Dict[str, Any], day: str, weather: Dict[str, Any], policy_text: str = "") -> None:
    """
    Print the outfit recommendation in a friendly format to the console.
    """
    print_banner("👔 Tomorrow’s Outfit Recommendation (AI Assisted)", color=Fore.CYAN)

    # Outfit picks
    top = outfit.get("Tops") or outfit.get("Top") or "—"
    bottom = outfit.get("Bottoms") or outfit.get("Bottom") or "—"
    shoes = outfit.get("Footwear") or outfit.get("Shoes") or "—"
    reasoning = (outfit.get("Reasoning") or "").strip()

    print(f"📅 Day: {Fore.YELLOW}{day}{Style.RESET_ALL}\n")

    # Weather summary
    print("🌤️ Commute Weather Snapshot:")
    print(weather_to_human_friendly(weather) if weather else "Weather data unavailable.")
    print()

    # Policy summary (truncated)
    if policy_text:
        short_policy = " ".join(policy_text.split())
        short_policy = (short_policy[:220] + "...") if len(short_policy) > 220 else short_policy
        print(f"🏢 Workplace Dress Guidance: {short_policy}\n")

    # Final outfit
    print("✅ Recommended Outfit:")
    print(f"  • Top:      {Fore.CYAN}{top}{Style.RESET_ALL}")
    print(f"  • Bottoms:  {Fore.CYAN}{bottom}{Style.RESET_ALL}")
    print(f"  • Footwear: {Fore.CYAN}{shoes}{Style.RESET_ALL}\n")

    # Reasoning 
    if reasoning:
        print(f"🧠 AI's Reasoning: {" ".join(reasoning.split())}")

    # What AI did (simple explanation)
    print("🤖 What the AI did (in plain English):")
    print("  1) Looked up tomorrow’s weather via an API")
    print("  2) Checked your company dress code from the handbook")
    print("  3) Searched your wardrobe list for good matches")
    print("  4) Picked one complete outfit and explained why\n")


def weather_to_human_friendly(weather: Dict[str, Any]) -> str:
    """
    Convert weather JSON into a human-friendly summary for the commute times.
    """
    if not weather:
        return "Weather data unavailable."

    lines = []

    def line(label, key):
        w = weather.get(key)
        if not w:
            return
        temp = round(w.get("temp_f", 0))
        cond = w.get("condition", "").strip()
        lines.append(f"{label}: {temp}°F and {cond}")

    line("🌅 Leaving home", "location_home_5am")
    line("🏢 Arriving work", "location_work_6am")
    line("🏢 Leaving work", "location_work_4pm")
    line("🏠 Arriving home", "location_home_3pm")

    return "\n".join(lines)


def weather_json_to_toon(json_str: str) -> str:
    """
    Convert JSON weather data to Toon format (CSV-like) for downstream token efficiency.
    """

    data = json.loads(json_str)
    lines = [
        "activity,city,time,temp (fahrenheit),condition",
        f"leaving home,{data['location_home_5am']['location']},{data['location_home_5am']['time']},{data['location_home_5am']['temp_f']},{data['location_home_5am']['condition']}",
        f"arriving work,{data['location_work_6am']['location']},{data['location_work_6am']['time']},{data['location_work_6am']['temp_f']},{data['location_work_6am']['condition']}",
        f"leaving work,{data['location_work_4pm']['location']},{data['location_work_4pm']['time']},{data['location_work_4pm']['temp_f']},{data['location_work_4pm']['condition']}",
        f"arriving home,{data['location_home_3pm']['location']},{data['location_home_3pm']['time']},{data['location_home_3pm']['temp_f']},{data['location_home_3pm']['condition']}"
    ]
    return "\n".join(lines)



# ---------------------------------- External Integrations ----------------------------------
# def get_weather_for_locations_tomorrow(*, verbose: bool = False) -> Dict[str, Any]:
#     """
#     Use this function to return mock weather data for testing without hitting the real API.
#     Uncomment the real API call code and comment out the mock data to use the real function.
#     """

#     test_data = {
#         "location_home_5am": {
#             "location": "Durham, NC",
#             "time": "2026-02-06 05:00",
#             "temp_f": 29.5,
#             "condition": "Clear "
#         },
#         "location_work_6am": {
#             "location": "Burlington, NC",
#             "time": "2026-02-06 06:00",
#             "temp_f": 27.2,
#             "condition": "Clear "
#         },
#         "location_work_4pm": {
#             "location": "Burlington, NC",
#             "time": "2026-02-06 16:00",
#             "temp_f": 38.8,
#             "condition": "Overcast "
#         },
#         "location_home_3pm": {
#             "location": "Durham, NC",
#             "time": "2026-02-06 15:00",
#             "temp_f": 43.7,
#             "condition": "Sunny"
#         }
#     }

#     return test_data


def get_weather_for_locations_tomorrow(verbose: bool = False):
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
    if verbose:
        print_banner("🌤️ Weather API Call", color=Fore.BLUE)
        # ---- Print JSON + token estimate ----
        print_banner("Weather (JSON)")
        results_json = json.dumps(results, indent=2)
        print(results_json)
        estimate_token_count("json", results_json)
        # ---- Print Toon + token estimate ----
        print_banner("Weather (Toon)")
        results_toon = weather_json_to_toon(results_json)
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


def invoke_bedrock_with_backoff(
    client,
    *,
    model_id: str,
    body_bytes: bytes,
    content_type: str = "application/json",
    accept: str = "application/json",
    max_attempts: int = 8,
    base_delay_s: float = 1.0,
    max_delay_s: float = 20.0,
):
    """Invoke Bedrock with explicit backoff for throttling/transient errors."""
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.invoke_model(
                modelId=model_id,
                body=body_bytes,
                contentType=content_type,
                accept=accept,
            )
        except ClientError as e:
            last_err = e
            code = (
                e.response.get("Error", {}).get("Code")
                if hasattr(e, "response") and isinstance(e.response, dict)
                else None
            )
            retryable = code in {
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
            }

            if not retryable or attempt >= max_attempts:
                raise

            # Exponential backoff + jitter
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay = delay * (0.7 + random.random() * 0.6)
            # print(f"Warning: Bedrock {code} (attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s")
            time.sleep(delay)

    if last_err:
        raise last_err
    raise RuntimeError("Bedrock invoke failed unexpectedly")


def call_llm_text(prompt: str, *, max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Calls an AWS Bedrock Claude model and returns raw text."""
    if not (AWS_REGION and AWS_BASE_MODEL):
        raise RuntimeError(
            "No Bedrock LLM configured. Set AWS_REGION and AWS_BASE_MODEL (or BASE_MODEL) to your Bedrock model id/ARN."
        )

    provider = (MODEL_PROVIDER or "").strip().lower()
    # Accept either a standard Claude model id (anthropic.*) or an inference-profile ARN that contains 'anthropic'
    is_claude = AWS_BASE_MODEL.startswith("anthropic.") or ("anthropic" in AWS_BASE_MODEL.lower()) or provider in {"claude", "anthropic"}
    if not is_claude:
        raise RuntimeError(
            f"Model provider not recognized as Claude. BASE_MODEL/AWS_BASE_MODEL={AWS_BASE_MODEL!r}, MODEL_PROVIDER={MODEL_PROVIDER!r}. "
            "This script currently supports Anthropic Claude on Bedrock."
        )

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    resp = invoke_bedrock_with_backoff(
        client,
        model_id=AWS_BASE_MODEL,
        body_bytes=json.dumps(body).encode("utf-8"),
    )

    payload = json.loads(resp["body"].read())
    content = payload.get("content", [])
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get("text", "") or ""
    return payload.get("completion", "") or ""



# ---------------------------------- Knowledge Bases ----------------------------------
def read_policy_text() -> str:
    """
    Reads the workplace attire policy from the DOCX file as plain text.
    """

    doc = DocxDocument(str(EMPLOYEE_HANDBOOK))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)


def get_attire_policy(
    handbook_records,
    *,
    query: str = "dress code attire policy business casual",
    top_k: int = 3,
) -> str:
    """
    Do semantic retrieval against the handbook chunks and return a compact policy snippet.
    """
    hits = search_handbook_chunks(AWS_REGION, EMBEDDING_MODEL_ID, query, handbook_records, top_k=top_k)
    policy = "\n\n".join([h["text"] for h in hits if h.get("text")])
    return policy.strip()


def build_wardrobe_df(*, preview: bool = False) -> pd.DataFrame:
    """
    Load the wardrobe knowledge base from Excel, classify each item into a bin, and return a DataFrame with the predictions.
    """
    wardrobe_items = read_wardrobe_knowledge_base(WARDROBE_KNOWLEDGE_BASE)
    df_raw = pd.DataFrame(wardrobe_items)

    # Classify ONCE
    df_pred = classify_article_bin(df_raw.copy())

    # Evaluate using that same prediction
    evaluate_binning_confusion(df_raw, df_pred)

    if preview:
        print(f"Loaded {len(wardrobe_items)} wardrobe 👕 items from the Excel file 𝄜 (knowledge base).")
        print("\n🔍 Classified bins preview:")
        print(df_pred[["Article", "Bin"]].head(10))

    return df_pred


# ---------------------------------- Tooling ----------------------------------
class SearchWardrobeArgs(BaseModel):
    """
    Validates the JSON schema for the search_wardrobe tool.
    """
    query: str = Field(..., description="Search query, e.g. 'warm tops for freezing rain'.")
    bin: Literal["Tops", "Bottoms", "Footwear"] = Field(..., description="Which cabinet to search.")
    top_k: int = Field(default=5, ge=1, le=10)
    work_appropriate: Optional[bool] = Field(default=True, description="Filter to work appropriate items.")


class SearchHandbookArgs(BaseModel):
    """
    Validates tool inputs for handbook search.
    """
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class WardrobePlanner:
    """
    A simple wrapper around the wardrobe DataFrame that provides policy retrieval and cabinet-local search functionality.
    """
    def __init__(self, wardrobe_df: pd.DataFrame, policy_text: str):
        # Store wardrobe DataFrame and policy text
        self.wardrobe_df = wardrobe_df.copy()
        self.policy_text = policy_text or ""

        def row_text(r):
            """
            Combine key attributes into one search string for vectorization.
            """
            parts = [
                str(r.get("Article", "")),
                str(r.get("Style", "")),
                str(r.get("Tags/Metadata", "")),
                str(r.get("Color", "")),
                str(r.get("Pattern", "")),
            ]
            return " ".join([p for p in parts if p])

        self.wardrobe_df["_search_text"] = self.wardrobe_df.apply(row_text, axis=1).astype(str)

    def get_policy_for_day(self, day_type: str) -> str:
        """
        For simplicity, this example just returns the same policy text regardless of day_type.
        """
        return self.policy_text

    def search_bin(
        self,
        *,
        query: str,
        bin_name: str,
        top_k: int = 5,
        work_appropriate: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        This function mimics a semantic search within a specific wardrobe bin or cabinet.  It allows the
        LLM to only search in one bin at a time ("Tops", "Bottoms", or "Footwear") and optionally filter
        to only work-appropriate items.
        """
        # If the LLM is calling this function, we can trust that bin_name is valid due to the Pydantic schema validation in SearchWardrobeArgs.

        # Filter the DataFrame to the specified bin
        df = self.wardrobe_df[self.wardrobe_df["Bin"].astype(str).str.lower() == bin_name.lower()].copy()
        if df.empty:
            return df

        # Optionally, filter the bin to only work-appropriate items if requested by the LLM
        if work_appropriate is not None:
            desired = "yes" if work_appropriate else "no"
            df = df[df["Work Appropriate"].astype(str).str.lower() == desired]

        if df.empty:
            return df

        # Vectorize the search text and query, compute cosine similarity, and return top_k results
        corpus = df["_search_text"].tolist()
        vectorizer = TfidfVectorizer().fit(corpus + [query])
        corpus_vecs = vectorizer.transform(corpus)
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, corpus_vecs)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return df.iloc[top_idx]


class ToolSpec:
    """
    A small wrapper that binds:
        * Tool name
        * Tool description
        * Pydantic model for tool args validation
        * Python function to run
    """
    def __init__(
        self,
        name: str,
        description: str,
        args_model: Type[BaseModel],
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.func = func

    def call(self, raw_args: Dict[str, Any]) -> Any:
        parsed = self.args_model(**(raw_args or {}))
        return self.func(**pydantic_model_dump(parsed))


class GetWeatherArgs(BaseModel):
    """
    Tool args schema for get_weather.
    """
    verbose: bool = Field(default=True, description="If true, tool prints debug details.")


def pydantic_model_dump(obj: BaseModel) -> Dict[str, Any]:
    """
    Compatibility wrapper obj.model_dump()
    """
    return obj.model_dump()


def pydantic_model_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Provide JSON schema to the LLM for tool argument validation. 
    """
    return model.model_json_schema()


# ---------------------------------- Tooling ----------------------------------
def parse_final_outfit_json(text: str) -> Dict[str, Any]:
    """
    Extract/parse the agent's Final Answer JSON safely.
    Accepts raw JSON or text containing a JSON object.
    """

    # Strip the text
    raw = (text or "").strip()

    # Strip common markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    # If it's already pure JSON, try to parse it
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Otherwise, try to extract the first {...} block
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        raise ValueError("Agent did not return JSON.")

    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Final Answer JSON was not an object.")
    return obj


def describe_tool_action(action: str, action_input: Dict[str, Any]) -> str:
    """
    Convert a raw tool call into a human-friendly narration line.
    """

    action = action.lower()

    if action == "get_weather":
        return "🌤️ Checking tomorrow’s weather forecast"

    if action == "search_wardrobe":
        query = action_input.get("query", "clothes")
        bin_name = action_input.get("bin")

        if bin_name:
            return f"👔 Looking through your {bin_name.lower()} drawer for: '{query}'"
        return f"👔 Looking through your wardrobe for: '{query}'"

    if action == "search_handbook":
        query = action_input.get("query", "policy")
        return f"📖 Reviewing company policy about: '{query}'"

    return f"🤖 Using tool: {action}"



# ---------------------------------- ReAct Agent ----------------------------------
class ReActAgent:
    """
    A simple ReAct agent implementation that can use tools with structured inputs and produce a final answer.
    The agent is designed to follow strict formatting rules for its thoughts, actions, and final answer, 
    and the prompt provides detailed instructions and constraints to guide its behavior.  The agent interacts 
    with the tools via a JSON schema-based interface, and the code includes robust parsing of the LLM's 
    responses to extract the intended actions and inputs.
    """
    def __init__(
        self,
        *,
        tools: Dict[str, ToolSpec],
        llm: Callable[[str], str],
        show_thoughts: bool = False,
        max_steps: int = 8,
    ):
        self.tools = tools
        self.llm = llm
        self.show_thoughts = show_thoughts
        self.max_steps = max_steps

    def build_prompt(self, question: str, scratchpad: str) -> str:
        """
        Build the prompt for the LLM, including tool descriptions and instructions.
        """
        tool_lines = []
        for name, tool in self.tools.items():
            schema = json.dumps(pydantic_model_schema(tool.args_model), indent=2)
            tool_lines.append(
                f"Tool: {name}\n"
                f"Description: {tool.description}\n"
                f"Args JSON Schema: {schema}\n"
            )

        tools_text = "\n".join(tool_lines)

        return (
            "You are a ReAct-style assistant that can use tools.\n\n"
            "Rules:\n"
            "- You may think briefly, then take ONE action, then wait for an observation.\n"
            "- Thought must be 1 sentence and high-level; do not reveal detailed step-by-step reasoning.\n"
            "- When you use a tool, respond EXACTLY in this format:\n"
            "  Thought: <brief>\n"
            "  Action: <tool_name>\n"
            "  Action Input: <json>\n"
            "- When you are done, respond EXACTLY in this format:\n"
            "  Final Answer: <your answer>\n"
            "- Action must be one of the valid tools below.\n\n"

            "Outfit planning constraint (must follow):\n"
            "- To recommend an outfit, you MUST search ONE cabinet at a time using search_wardrobe.\n"
            "- You MUST call search_wardrobe exactly 3 times, with bin=Tops, Bottoms, Footwear (any order).\n"
            "- Do NOT call Final Answer until you have done all 3 searches and received observations for each.\n"
            "- Only after you have those 3 observations, produce the final outfit using one item from each cabinet.\n\n"

            "Final Answer format (MUST follow):\n"
            "- Final Answer MUST be valid JSON only (no extra text).\n"
            "- JSON schema:\n"
            '  {"Tops": "<article>", "Bottoms": "<article>", "Footwear": "<article>", "Reasoning": "<1-2 sentences>"}\n\n'

            f"{tools_text}\n"
            f"User Question: {question}\n\n"
            f"Scratchpad so far:\n{scratchpad}\n"
        )


    def parse_llm_reply(self, text: str) -> Dict[str, Any]:
        """
        Parse the LLM's reply to determine if it's a Thought/Action or a Final Answer, and extract the relevant parts.
        """
        raw = (text or "").strip()
        if not raw:
            return {"type": "error", "message": "Empty LLM response"}

        # Do a quick regex check to see if the LLM is trying to give a Final Answer
        m_final = re.search(r"^\s*Final Answer\s*:\s*(.*)\s*$", raw, flags=re.IGNORECASE | re.DOTALL)
        if m_final:
            return {"type": "final", "final": m_final.group(1).strip()}

        # Try to extract Thought, Action, and Action Input using regex
        thought_match = re.search(r"^\s*Thought\s*:\s*(.*)$", raw, flags=re.IGNORECASE | re.MULTILINE)
        action_match = re.search(r"^\s*Action\s*:\s*(.*)$", raw, flags=re.IGNORECASE | re.MULTILINE)
        input_match = re.search(
            r"^\s*Action Input\s*:\s*(.*)$",
            raw,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        # Normalize the action name and parse the action input JSON if present
        thought = thought_match.group(1).strip() if thought_match else ""
        action_raw = action_match.group(1).strip() if action_match else ""
        action = normalize_action_name(action_raw)

        action_input: Dict[str, Any] = {}
        if input_match:
            candidate = input_match.group(1).strip()
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"```\s*$", "", candidate)

            if "Observation:" in candidate:
                candidate = candidate.split("Observation:")[0].strip()
            if "Thought:" in candidate:
                candidate = candidate.split("Thought:")[0].strip()

            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        action_input = parsed
                    else:
                        action_input = {"value": parsed}
                except Exception:
                    action_input = {"query": candidate}

        if not action:
            return {"type": "error", "message": f"Could not parse Action from: {raw}"}

        return {
            "type": "action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "raw": raw,
        }

    def run(self, question: str) -> str:
        """
        Entry  point to run the ReAct agent loop.  It will keep calling the LLM and tools until 
        it gets a Final Answer or hits the max step limit.
        """
        print_banner(f"🤖 ReAct AI Agent Starting...")
        print(f"User: {question}\n")

        # Initialize an empty scratchpad to keep track of the conversation history and tool interactions for the LLM's context.
        scratchpad = ""

        for _step in range(1, self.max_steps + 1):
            prompt = self.build_prompt(question, scratchpad)
            reply = self.llm(prompt)
            parsed = self.parse_llm_reply(reply)

            if parsed["type"] == "final":
                print(f"Final Answer: {parsed['final']}")
                return parsed["final"]

            if parsed["type"] == "error":
                msg = parsed.get("message", "Unknown parse error")
                print(f"Final Answer: I couldn't proceed: {msg}")
                return f"I couldn't proceed: {msg}"

            thought = parsed.get("thought", "")
            if self.show_thoughts and thought:
                print(f"🤖 LLM Thought: {thought}")

            action = parsed["action"]
            action_input = parsed.get("action_input", {})
            friendly_line = describe_tool_action(action, action_input)
            print(f"\n{friendly_line}...")

            tool = self.tools.get(action)
            if not tool:
                observation = {
                    "error": f"Unknown tool: {action}",
                    "valid_tools": list(self.tools.keys()),
                }
            else:
                try:
                    observation = tool.call(action_input)
                except ValidationError as ve:
                    observation = {
                        "error": "Tool input validation failed",
                        "details": json.loads(ve.json()),
                    }
                except Exception as e:
                    observation = {"error": str(e)}

            obs_text = safe_json_dumps(observation)
            # print(f"Observation: {obs_text}\n")

            scratchpad += (
                f"Thought: {thought}\n"
                f"Action: {action}\n"
                f"Action Input: {json.dumps(action_input)}\n"
                f"Observation: {obs_text}\n\n"
            )

        print("Final Answer: I hit the max step limit without finishing.")
        return "I hit the max step limit without finishing."



# ---------------------------------- Tool Wiring ----------------------------------
def make_tools(df: Optional[pd.DataFrame], policy_text: str) -> Dict[str, ToolSpec]:
    """
    Define the tools that the ReAct agent can use, including:
        - get_weather:      Fetches weather data for the commute times via API
        - search_wardrobe:  Performs a semantic search over the wardrobe DataFrame for a specific cabinet and query.
        - search_handbook:  Performs semantic search over the employee handbook chunks.
    """
    planner = WardrobePlanner(df, policy_text=policy_text) if df is not None else None

    print_banner("Placed Employee Handbook into our 'Smart Library' for Semantic Search (Vector Embeddings)")
    handbook_chunks = chunk_employee_handbook(EMPLOYEE_HANDBOOK)

    print(f"{Fore.YELLOW}Preview of first handbook chunk 📃:{Style.RESET_ALL}")
    print(f"----------------------------------------------------------------")
    print(f"{handbook_chunks[0]['text'][:200]}...\n")
    print(f"----------------------------------------------------------------")
    print(f"{Fore.YELLOW}Preview of second handbook chunk 📃:{Style.RESET_ALL}")
    print(f"----------------------------------------------------------------")
    print(f"{handbook_chunks[1]['text'][:200]}...\n")
    print(f"----------------------------------------------------------------")
    print(f"Employee handbook total chunks: {len(handbook_chunks)}")

    handbook_records = embed_handbook_chunks(AWS_REGION, EMBEDDING_MODEL_ID, SCRIPT_DIR, handbook_chunks)

    def tool_get_weather(verbose: bool = True) -> Dict[str, Any]:
        return get_weather_for_locations_tomorrow(verbose=verbose)

    def tool_search_wardrobe(
        query: str,
        bin: str,
        top_k: int = 5,
        work_appropriate: Optional[bool] = True,
    ) -> Dict[str, Any]:
        print_banner("🧺 Looking into the wardrobe cabinet...")
        print(f"🧍 Opening the '{Fore.CYAN}{bin}{Style.RESET_ALL}' drawer...")
        print(f"🔎 Looking for items that match: {Fore.YELLOW}{query}{Style.RESET_ALL}")
        print("🏢 Keeping only work-appropriate pieces." if work_appropriate else "🎉 Including casual / off-duty pieces too.")
        print()

        results = planner.search_bin(query=query, bin_name=bin, top_k=top_k, work_appropriate=work_appropriate)

        print("👀 Inside the drawer, you see:\n")
        if results.empty:
            print("   (Hmm… nothing here fits that description.)\n")
        else:
            for _, row in results.iterrows():
                article = row.get("Article", "Unknown item")
                color = row.get("Color", "")
                pattern = row.get("Pattern", "")
                wa = row.get("Work Appropriate", "Unknown")
                print(f"   • {Fore.CYAN}{article}{Style.RESET_ALL} — {color} {pattern} | Work OK: {wa}")

        print("\n🧺 Closing the drawer.\n")

        out = []
        for _, row in results.iterrows():
            out.append(
                {
                    "Article": row.get("Article"),
                    "Style": row.get("Style"),
                    "Color": row.get("Color"),
                    "Pattern": row.get("Pattern"),
                    "Tags/Metadata": row.get("Tags/Metadata"),
                    "Bin": row.get("Bin"),
                    "Work Appropriate": row.get("Work Appropriate"),
                }
            )

        return {"count": len(out), "results": out}


    def tool_search_handbook(query: str, top_k: int = 3) -> Dict[str, Any]:
        print_banner("📖 Employee Handbook Retrieval", color=Fore.MAGENTA)
        results = search_handbook_chunks(AWS_REGION, EMBEDDING_MODEL_ID, query, handbook_records, top_k=top_k)
        for i, chunk in enumerate(results, 1):
            print(f"[Human] Handbook Section {i}:\n{chunk['text']}\n")
        return {"count": len(results), "chunks": [c["text"] for c in results]}

    return {
        "get_weather": ToolSpec(
            name="get_weather",
            description="Get tomorrow's weather for home/work commute windows.",
            args_model=GetWeatherArgs,
            func=tool_get_weather,
        ),
        "search_wardrobe": ToolSpec(
            name="search_wardrobe",
            description="Search wardrobe items (semantic search over wardrobe knowledge base).",
            args_model=SearchWardrobeArgs,
            func=tool_search_wardrobe,
        ),
        "search_handbook": ToolSpec(
            name="search_handbook",
            description="Semantic search employee handbook sections (e.g., dress policy).",
            args_model=SearchHandbookArgs,
            func=tool_search_handbook,
        ),
    }, handbook_records



# ---------------------------------- Output / Export ----------------------------------
def generate_stunning_html(
    planned_outfit: Dict[str, str],
    day_of_week: str,
    weather: Dict[str, Any],
    policy_text: str,
    *,
    wardrobe_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Ask the LLM to generate a nice standalone HTML card for the planned outfit,
    then save it locally. If the model output isn't valid HTML, fall back to a
    simple built-in template.
    """

    prompt = (
        "Create a standalone HTML page (with inline CSS) for a 'Planned Attire' card.\n"
        "Include:\n"
        "- The day of the week\n"
        "- The selected outfit (top, bottom, footwear)\n"
        "- The commute weather\n"
        "- A short 2–6 line poem that matches the outfit and weather vibe\n\n"
        "Return ONLY raw HTML. No explanations. No markdown fences.\n\n"
        f"Day: {day_of_week}\n"
        f"Outfit: {json.dumps(planned_outfit, ensure_ascii=False)}\n"
        f"Weather: {json.dumps(weather, ensure_ascii=False)}\n"
        f"Policy context (optional): {policy_text[:1200]}\n"
    )

    raw = call_llm_text(prompt, max_tokens=4000, temperature=0.9)

    # Clean up common markdown formatting issues
    html = (raw or "").strip()
    html = re.sub(r"^```(?:html)?\s*", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\s*```$", "", html)
    first_tag = html.find("<")
    if first_tag != -1:
        html = html[first_tag:].lstrip()

    # If the model didn't actually return HTML, use a simple fallback
    if "<html" not in html.lower():
        tops = planned_outfit.get("Tops", "—")
        bottoms = planned_outfit.get("Bottoms", "—")
        footwear = planned_outfit.get("Footwear", "—")

        html = f"""<!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Planned Attire</title>
        <style>
            body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; background:#fafafa; }}
            .card {{ max-width: 760px; margin: auto; background:white; padding:20px; border-radius:14px; box-shadow:0 8px 24px rgba(0,0,0,0.08); }}
            h1 {{ margin-top:0; }}
            pre {{ background:#f4f4f4; padding:12px; border-radius:8px; white-space:pre-wrap; }}
        </style>
        </head>
        <body>
        <div class="card">
            <h1>Planned Attire — {day_of_week}</h1>
            <p><strong>Top:</strong> {tops}<br>
            <strong>Bottom:</strong> {bottoms}<br>
            <strong>Footwear:</strong> {footwear}</p>

            <h2 style="font-size:16px;margin-top:18px;">Commute Weather</h2>
            <pre>{json.dumps(weather, indent=2, ensure_ascii=False)}</pre>

            <p style="margin-top:18px;"><em>
            Step into the day, dressed just right,<br>
            Layers for comfort, calm and light.<br>
            Weather may shift, but style stays true,<br>
            Ready for whatever comes into view.
            </em></p>
        </div>
        </body>
        </html>"""

    filename = f"{datetime.now().strftime('%m-%d-%y_%H%M')}_Planned_Attire.html"

    out_path = SCRIPT_DIR / filename
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML card to: {out_path}")
    return out_path



# ---------------------------------- Main ----------------------------------
# Run ReAct Agent
react_mode = True
# Print the model thoughts
show_thoughts = True
# Preview the wardrobe knowledge base
preview_kb = False
# Export the HTML card
export_html = True


def main() -> None:
    # Load the wardrobe knowledge base as a data frame
    df = build_wardrobe_df(preview=preview_kb)

    # Load the employee handbook knowledge base
    raw_handbook_text = read_policy_text()

    # Prepare tools
    tools, handbook_records = make_tools(df, raw_handbook_text)

    # Precompute shared context for HTML (and for fallback)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")

    try:
        weather = get_weather_for_locations_tomorrow(verbose=False)
    except Exception as e:
        print("Warning: weather fetch failed:", e)
        weather = {}

    # Pull a compact dress policy snippet once (for the HTML prompt context)
    try:
        always_policy = get_attire_policy(
            handbook_records,
            query="dress code policy business casual",
            top_k=3
        )
    except Exception as e:
        print("Warning: handbook retrieval failed:", e)
        always_policy = raw_handbook_text[:1200] if raw_handbook_text else ""

    outfit: Dict[str, Any] = {}

    # Run ReAct agent loop
    if react_mode:
        agent = ReActAgent(
            tools=tools,
            llm=lambda p: call_llm_text(p),
            show_thoughts=show_thoughts
        )

        final_text = agent.run("What should I wear tomorrow?")
        try:
            outfit = parse_final_outfit_json(final_text)
        except Exception as e:
            print("Warning: could not parse outfit JSON from agent:", e)
            outfit = {}

    # If agent failed, fallback to something minimal so HTML still works
    if not outfit:
        outfit = {"Tops": "—", "Bottoms": "—", "Footwear": "—", "Reasoning": "No outfit returned."}

    # Console output
    print_banner("✅ Final Answer", color=Fore.GREEN)
    print_outfit_presentation(outfit, tomorrow, weather, always_policy)

    # Export HTML
    if export_html:
        try:
            html_path = generate_stunning_html(outfit, tomorrow, weather, always_policy, wardrobe_df=df)
        except Exception as e:
            print("Warning: failed to generate/save HTML card:", e)


if __name__ == "__main__":
    main()