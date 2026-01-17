# ---------------------------------- Imports ----------------------------------
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from openai import OpenAI as OpenAIClient
import os
from pathlib import Path
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from typing import TypedDict, Dict, List, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")



# ---------------------------------- Paths & Settings ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = Path(parent_dir).parent

output_dir = Path(script_dir) / "output"
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualiziation of what's going on 

    Ex.
    Input:  "Device"
    Output:
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
            *                          Device                           *
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    """
    banner_len = len(text)
    mid = 49 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 50)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 50)


# ---------------------------------- PDF Reporting ----------------------------------
def save_agent_run_to_pdf(out: dict, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styleH = styles["Heading1"]
    styleH2 = styles["Heading2"]
    styleN = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Agent Run Output"
    )

    story = []

    # Title
    story.append(Paragraph("Market Research Agent - Run Output", styleH))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styleN))
    story.append(Spacer(1, 12))

    # Question
    story.append(Paragraph("User Question", styleH2))
    story.append(Paragraph(out.get("question", ""), styleN))
    story.append(Spacer(1, 12))

    # Planner
    story.append(Paragraph("Planner Agent - Plan", styleH2))
    plan = out.get("plan", [])
    if plan:
        for i, step in enumerate(plan, 1):
            story.append(Paragraph(f"{i}. {escape_for_pdf(step)}", styleN))
    else:
        story.append(Paragraph("(No plan returned)", styleN))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Planner Agent - Queries", styleH2))
    queries = out.get("queries", [])
    if queries:
        for i, q in enumerate(queries, 1):
            story.append(Paragraph(f"{i}. {escape_for_pdf(q)}", styleN))
    else:
        story.append(Paragraph("(No queries returned)", styleN))

    story.append(PageBreak())

    # Search
    story.append(Paragraph("Search Agent - Top Results", styleH2))
    search_results = out.get("search_results", [])
    if search_results:
        for r in search_results[:30]:  # cap to keep PDF reasonable
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")
            story.append(Paragraph(f"<b>{escape_for_pdf(title)}</b>", styleN))
            story.append(Paragraph(f"{escape_for_pdf(url)}", styleN))
            story.append(Paragraph(f"{escape_for_pdf(snippet)}", styleN))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("(No search results)", styleN))

    story.append(PageBreak())

    # Retriever
    story.append(Paragraph("Retriever Agent - Extracted Evidence Bullets", styleH2))
    bullets = out.get("retrieved_results", [])
    if bullets:
        for i, b in enumerate(bullets, 1):
            story.append(Paragraph(f"{i}. {escape_for_pdf(str(b))}", styleN))
    else:
        story.append(Paragraph("(No retrieved results)", styleN))

    story.append(PageBreak())

    # Synthesizer
    story.append(Paragraph("Synthesizer Agent - Draft Report", styleH2))
    raw = out.get("synthesized_report", "")
    safe = escape_for_pdf(raw)
    safe = safe.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    story.append(Paragraph(safe, styleN))

    story.append(PageBreak())

    # Validator
    story.append(Paragraph("Validator Agent - Validation Output", styleH2))

    validations = out.get("validations", [])
    if validations:
        for i, item in enumerate(validations, 1):
            story.append(Paragraph(f"{i}. {escape_for_pdf(item)}", styleN))
    else:
        story.append(Paragraph("(No validations returned)", styleN))

    doc.build(story)


def escape_for_pdf(text: str) -> str:
    """ReportLab Paragraph uses a mini-HTML parser; escape special chars."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


# ---------------------------------- Load Environment Variables ----------------------------------
print_banner("Load Environment Variables")

# Load environment variables and create OpenAI client
load_dotenv(dotenv_path = grandparent_dir / ".env", override=True)



openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# View the first few characters in the key
print(f"OpenAI API Key: {openai_api_key[:15]}...")
print(f"Google API Key: {google_api_key[:15]}...")
print(f"Tavily API Key: {tavily_api_key[:15]}...")

# Configure APIs
openai_client = OpenAIClient(api_key = openai_api_key)
genai.configure(api_key = google_api_key)

# Initialize gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")




# ---------------------------------- Variables ----------------------------------
print_banner("Variables")
# Set up tavily search tool to equip LLM with tooling
tavily_search_tool = TavilySearchResults(max_results = 3)
# List of tools for this step
tools_list_single = [tavily_search_tool]


# ---------------------------------- Pydantic Output Validation ----------------------------------
class PlanOutput(BaseModel):
    plan: List[str] = Field(description="3-6 step plan")
    queries: List[str] = Field(description="5-10 high quality web search queries")


class RetrieverOutput(BaseModel):
    retrieved_results: List[str] = Field(description="15-35 high quality snippets of text")


class SynthesizerOutput(BaseModel):
    synthesized_report: str = Field(
        description="Clear, professional written report in normal prose (no markdown)"
    )


class ValidatorOutput(BaseModel):
    validations: List[str] = Field(
        description="One item per line. Each item must start with 'VALID:' or 'INVALID:'."
    )


# ---------------------------------- Define State ----------------------------------
print_banner("Define State")
# Define a state that includes: 
# 1. input_text: The original text to be summarized
# 2. summary: The generated summary of the input text
# The state is like a container that stores and passes data between different parts of our workflow
# Each node receives and returns a state object, and the State can include messages, variables, memory, etc.
class AgentState(TypedDict):
    """
    State container for the text summarization workflow.
    Holds the original input text and the generated summary.
    """
    question: str
    plan: List[str]
    queries: List[str]
    search_results: List[Dict[str, Any]]
    retrieved_results: List[str]
    synthesized_report: str
    validations: List[str]

print("Defined AgentState")


# ---------------------------------- Define Node Functions ----------------------------------
print_banner("Define Node Functions")

# Configure the OpenAI Client (using LangChain's wrapper)
# GPT-4o is generally better with tool calling
llm = ChatOpenAI(model = "gpt-4.1-mini", temperature = 0, streaming = True)
print("LangChain OpenAI Chat Model configured.")


# Define the key nodes, which represents the functions that perform specific tasks in the graph
# They receive the current state and return a modified state

def plan(state: AgentState) -> AgentState:
    question = state["question"]

    planner_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    planner = planner_llm.with_structured_output(PlanOutput)

    prompt = (
        "You are a planner for a market research agent.\n"
        "Given the user question, produce:\n"
        "1) A short step-by-step plan (3-6 bullets)\n"
        "2) 5-10 high-quality web search queries (include synonyms, date qualifiers if relevant)\n"
    )

    data: PlanOutput = planner.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"User question: {question}")
        ]
    )

    return {
        **state,
        "plan": data.plan,
        "queries": data.queries,
    }


def search(state: AgentState) -> AgentState:
    queries = state.get("queries", [])
    all_results = []

    for q in queries:
        # TavilySearchResults returns a list[dict] like:
        # [{"title": "...", "url": "...", "content": "..."}]
        results = tavily_search_tool.invoke({"query": q})
        for r in results:
            r["query"] = q  # keep provenance
        all_results.extend(results)

    return {
        **state,
        "search_results": all_results
    }


def retrieve(state: AgentState) -> AgentState:
    search_results = state.get("search_results", [])

    retriever_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    retriever = retriever_llm.with_structured_output(RetrieverOutput)

    prompt = (
        "You are a retriever for a market research agent. Given the snippets of relevant text, \n"
        "extract the key facts, quotes/snippets, and metadata, produce a short list of the most \n"
        "relevant key facts, quotes/snippets, and metadata (15-35 bullets).\n"
    )

    data: RetrieverOutput = retriever.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Snippets of relevant text: {search_results}")
        ]
    )

    return {
        **state,
        "retrieved_results": data.retrieved_results,
    }


def synthesize(state: AgentState) -> AgentState:
    retrieved_results = state.get("retrieved_results", [])

    synthesizer_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    synthesizer = synthesizer_llm.with_structured_output(SynthesizerOutput)

    prompt = (
        "You are a business-style report writer.\n"
        "Write a clear, professional narrative report in normal paragraphs.\n"
        "Do not use markdown formatting.\n"
        "Include URLs in parentheses where appropriate."
    )

    data: SynthesizerOutput = synthesizer.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Snippets of relevant context: {retrieved_results}")
        ]
    )

    return {
        **state,
        "synthesized_report": data.synthesized_report,
    }


def validate(state: AgentState) -> AgentState:
    retrieved_results = state.get("retrieved_results", [])
    report = state.get("synthesized_report", "")

    validator_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    validator = validator_llm.with_structured_output(ValidatorOutput)

    prompt = (
        "You are a report validator.\n"
        "Extract the main claims from the report and validate each claim against the provided sources.\n"
        "Return ONLY a list of strings.\n"
        "Rules:\n"
        "- 10-30 items\n"
        "- Each item must start with 'VALID:' or 'INVALID:'\n"
        "- One sentence per item\n"
    )

    data: ValidatorOutput = validator.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"REPORT:\n{report}\n\nSOURCES:\n{retrieved_results}")
        ]
    )

    return {
        **state,
        "validations": data.validations,
    }


# ---------------------------------- Build Graph ----------------------------------
print_banner("Build Graph")
# Initialize AgentState
builder = StateGraph(AgentState)

# Build nodes
builder.add_node("planner", plan)
builder.add_node("searcher", search)
builder.add_node("retriever", retrieve)
builder.add_node("synthesizer",  synthesize)
builder.add_node("validator",  validate)

# Connect nodes to functions
builder.set_entry_point("planner")
builder.add_edge("planner", "searcher")
builder.add_edge("searcher", "retriever")
builder.add_edge("retriever", "synthesizer")
builder.add_edge("synthesizer", "validator")
builder.add_edge("validator", END)

# Compile the graph
graph = builder.compile()


# ---------------------------------- Initialize ----------------------------------
print_banner("Initialize")

# Initialize 
initial_state = {"question": "Summarize recent trends in GLP-1 obesity drugs and their market impact."}
out = graph.invoke(initial_state)
print_banner("plan")
print(out["plan"])

print_banner("searcher")
print(len(out["search_results"]), "results")
print(out["search_results"][:5])

print_banner("retriever")
print(len(out["retrieved_results"]), "retrieved_results")
print(out["retrieved_results"][:5])

print_banner("synthesizer")
print(len(out["synthesized_report"]), "synthesized_report")
print(out["synthesized_report"][:200])

print_banner("validator")
print(len(out.get("validations", [])), "validations")
print(out.get("validations", [])[:5])


# ---------------------------------- Write PDF ----------------------------------
pdf_path = output_dir / "agent_run.pdf"
save_agent_run_to_pdf(out, pdf_path)
print(f"\nSaved PDF to: {pdf_path}")
