# AI_Agents 🤖
---
### About 🧩: 
- In this repo, I play around with AI Agentization.  Each different folder is a different project.  

---
## ⭐ 01: Market Research Multi-Agent System (LangGraph) 
- **Project Overview:** This repository contains a prototype **multi-agent AI research system** built with **LangGraph** that demonstrates how a structured team of LLM agents can collaboratively reason, search the web, extract evidence, synthesize insights, and validate claims.  The goal of this project is to explore how agentic workflows can turn an open-ended question into a well-structured, evidence-based report in a transparent and auditable way.  I intentionally structured this as a multi-agent system rather than sending everything to one model in a huge prompt and having long context degradation (inducing hallucinations).  By splitting the tasks, I get cleaner reasoning, better grounding, and more predictable outputs.  
- **Highlights:** Given a user question (e.g., *“Summarize recent trends in GLP-1 obesity drugs and their market impact”*), the system runs through a sequence of specialized agents:
    - **Planner Agent:** Breaks the question into:
        - A clear step-by-step plan  
        - A set of high-quality web search queries  
    - **Search Agent:** Uses the Tavily web search API to gather real, up-to-date sources relevant to the question.
    - **Retriever Agent:** Reads the search results and extracts:
        - Key facts  
        - Relevant quotes  
        - Important evidence snippets  
    - **Synthesizer Agent:** Writes a clear, professional narrative report in plain prose based on the retrieved evidence.
    - **Validator Agent:** Checks the synthesized report against the original sources and produces a list of:
        - `VALID:` claims that are supported  
        - `INVALID:` claims that are not clearly backed by evidence  

This creates a transparent pipeline where you can trace how a final answer was constructed.
- **Architecture:** The system is implemented as a **stateful graph** using LangGraph.  Each node:
    - Receives a shared state object  
    - Modifies only its relevant fields  
    - Passes the updated state forward  
- **Technologies:**
    - **Python 3.12**
    - **LangGraph** (for agent orchestration)
    - **LangChain + OpenAI (GPT-4.1-mini)**  
    - **Tavily Web Search API**
    - **Pydantic** (for structured LLM outputs)
    - **ReportLab** (for PDF reporting)
- **Output Artifacts:**
    - **Console output** showing each agent’s intermediate results  
    - **A structured PDF report** containing:
        - The original question  
        - The planner’s reasoning  
        - Web search results  
        - Extracted evidence  
        - Final synthesized report  
        - Claim validation results  

<br><img src="https://github.com/david125tran/AI_Agents/blob/main/01/output/screenshot.png?raw=true" width="500"/>

- **Mermaid Diagram:**
```mermaid
flowchart LR
    U[User Question] --> P

    subgraph LangGraph Workflow
        P[Planner Agent]
        S[Search Agent]
        R[Retriever Agent]
        SY[Synthesizer Agent]
        V[Validator Agent]
    end

    P --> S
    S --> R
    R --> SY
    SY --> V

    P -.->|updates| State
    S -.->|updates| State
    R -.->|updates| State
    SY -.->|updates| State
    V -.->|updates| State

    State[(Shared State Store)]

    V --> PDF[PDF Report]
    V --> Console[Console Output]
```

---

## ⭐ 02: Stock Due Diligence Agent (LangGraph + AWS Bedrock + Local RAG)
- **Project Overview:**  This project explores 🤖 **AI agentization as a system design pattern**, rather than treating LLMs as single-shot answer generators. Instead of asking one model a broad question like “What stock should I buy right now?”-which typically produces an opaque, non-reproducible response-this system is intentionally structured as a **multi-agent workflow** where each agent has a narrow, well-defined responsibility.  

  The result is a production-style stock due diligence agent that transforms a natural-language question into a structured, cited, auditable PDF report, with clear separation between: (1) Deterministic financial facts, (2) Recent market news, (3) Semantic evidence retrieval, and (4) LLM reasoning.
  
## 📄 Example Output Due Diligence PDF Reports
  `Please ignore the warnings in the report.  I was getting throttled by Alpha Vantage API and appending warnings to the report, lol.`

<table>
  <tr>
    <td align="center">
      <b>📟 NVDA</b><br/>
      <a href="./02/reports/NVDA_due_diligence_20260125.pdf">➡️ Open Full PDF</a><br/><br/>
      <img src="./02/images/NVDA.png" width="350"/>
    </td>
    <td align="center">
      <b>✨ PLTR</b><br/>
      <a href="./02/reports/PLTR_due_diligence_20260125.pdf">➡️ Open Full PDF</a><br/><br/>
      <img src="./02/images/PLTR.png" width="350"/>
    </td>
    <td align="center">
      <b>🚗 TSLA</b><br/>
      <a href="./02/reports/TSLA_due_diligence_20260125.pdf">➡️ Open Full PDF</a><br/><br/>
      <img src="./02/images/TSLA.png" width="350"/>
    </td>
  </tr>
</table>

- **Why multi-agentization instead of 💬 “just ask ChatGPT?”:**
  - Asking a general LLM `“What stock should I buy?”` has several fundamental limitations:
    - **No control over evidence**. The model is trained on old outdated data & the data it sees may not be factual
    - **No separation between facts and reasoning**.  Market data, news, and analysis are blended together in one opaque response.
    - **No auditability**
    - **High variance**. The same question asked twice can yield materially different answers.

  - Rather than relying on a single model to “do everything at once,” the workflow decomposes the task into specialized agents coordinated by LangGraph. This mirrors how real research and analytics teams operate: data collection first, evidence curation second, analysis last.
    - Deterministic agents fetch structured financial data from trusted APIs (Alpha Vantage + SEC).
    - A dedicated news agent gathers recent, bounded market context with explicit recency constraints.
    - The LLM is used strictly for reasoning and synthesis, not data acquisition.
    - A validator agent performs a lightweight claim audit, making reasoning inspectable rather than implicit.
  - This design produces outputs that are **more stable, more explainable, and more reusable** than a single-shot LLM answer.
  - I built this agentic workflow to create an auditable research system to do due dillegence on stocks for me because I just don't have the time to do the research.  I stay pretty busy between work, jiu-jitsu, and constantly programming.  

- **🏗️ Intentional Design Choices:**
  - **Separation of concerns:**  I intentionally segregated factual data from news (which could cause noise)
    - Factual/deterministic data is collected from Alpha Vantage + SEC through API calls
    - Recent news is collected from FinnHub API call because Finnhub allows me to select articles from a date range through the call 

  - **Deliberate chunking strategy:**  
    - Two separate vector indexes:
        - `deterministic/` - **Financial facts** (Alpha Vantage + SEC data API call)
        - `news/` → **Recent news articles** (Finnhub API call).  The system only surfaces news from the last 365 days and caps total articles, avoiding “recency noise."
    - I then added metadata tags so that I could filter my retrievals later on prior to sending data to the LLM

  - **Semantic Retrieval with Metadata Guardrails:**  
    - Rather than relying on pure similarity search alone, this system uses semantic retrieval as a candidate generator and then **applies metadata-based guardrails to control what evidence is allowed to reach the LLM** (i.e. filter for news recency). Query embeddings retrieve the most relevant chunks from local vector indexes, after which results are filtered and constrained using structured metadata. In particular, retrieval is split across two dedicated indexes-one for deterministic facts and one for recent news-and additional hard filters are applied for news recency, chunk identity deduplication, and source attribution. This ensures the LLM reasons over evidence that is not only semantically relevant, but also recent, stable, and contextually valid. By combining vector similarity with deterministic metadata constraints, the retrieval layer avoids stale or misleading context while remaining fast and production-ready.
    - Retrieval is run separately over:
      - a *deterministic facts index* (for stable truths), and  
      - a *news index* (for recent developments).  

  - **Reliability & Observability:**  
    - For this script, I was caching API call data to local memory with time to live variables to save on $ and to perform less API calls.  This is because I'm using the free tier for the API calls.
    - When the API calls would fail, I scripted in a graceful fallback to using older/stale cache data
    - Exponential backoff on Bedrock throttling.  
    - Progress tracking across the entire graph for observability.

  - **Claim traceability:**  
    - The report requires inline citations like `[chunk_id]`.  
    - A validator agent performs a lightweight “claim audit,” marking which lines are supported by at least one verifiable source URL.

- **🤖 Agent roles (what the agent does end-to-end):**
  - **Orchestrator Agent**  
     - Extracts exactly one ticker from the user question (or asks a single clarification).  
     - Generates targeted retrieval queries.

  - **Deterministic Analyst Agent**  
     - Pulls structured data from Alpha Vantage (prices, overview, financials) and SEC filings.  
     - Normalizes and summarizes this into a compact snapshot.

  - **News Agent**  
     - Fetches up to 12 months of company news from Finnhub with caching + deduplication.

  - **Archiver Agent (Local RAG layer)**  
     - Builds two local vector indexes:
       - `news/` → chunked recent articles  
       - `deterministic/` → latest financial facts  
     - Uses content hashes to avoid redundant embeddings.

  - **Retriever Agent**  
     - Runs semantic search over both indexes.  
     - Filters old news, deduplicates chunks, and keeps only the most relevant evidence.

  - **Advisor Agent**  
     - Receives a compact market snapshot + curated evidence.  
     - Produces a structured report (rating, risks, key drivers, confidence, gaps) with citations.

  - **Validator Agent**  
     - Audits each claim for source presence and appends a “Claim Audit” section.

- **Architecture:**  
  The system is implemented as a **stateful LangGraph workflow** where each node:
  - Reads from a shared `AgentState`  
  - Writes only its relevant outputs  
  - Updates progress for observability  
  - Hands off to the next specialized agent

- **Technologies:**
  - **Python 3.12**
  - **LangGraph** (agent orchestration)
  - **LangChain + AWS Bedrock** (Claude-style chat + embeddings)
  - **LlamaIndex** (vector storage & retrieval)
  - **Alpha Vantage + Finnhub + SEC EDGAR APIs**
  - **Pydantic** (structured LLM outputs)
  - **ReportLab** (PDF generation)
  - **Disk-based caching + manifests**

- **Output Artifacts:**
  - Rich console logs showing progress and failures  
  - A **sourced PDF report** including:
    - Original question  
    - Ticker extraction result  
    - Deterministic financial snapshot  
    - Retrieved evidence  
    - LLM-written investment view  
    - Claim validation results  

- **Initial Problems:**  
  I originally began this project about a month ago by leaning heavily on general web search (`Tavily`) as the primary knowledge source. Early versions of the system tried to answer investment questions purely from live web results, but I quickly ran into two practical problems:  
  1) **Recency control was too weak.** Tavily returned a mix of high-quality recent sources and noisy, low-signal content, and I didn’t have a reliable way to systematically discriminate between “material, decision-relevant news” and background chatter.  
  2) **Structured financial data was hard to scrape cleanly.** Key facts like prices, cash flows, balance sheets, and filings were scattered across websites in tables, PDFs, or interactive dashboards that were brittle to parse with generic scraping.  

  After a few iterations, I deliberately **re-architected the entire system** around a deterministic data layer first-pulling structured data from trusted APIs (Alpha Vantage + SEC EDGAR) and treating news as a separate, curated signal rather than the foundation of reasoning. This shift made the system far more reliable & repeatable.  I intentionally avoided brittle web table scraping in favor of structured APIs.  

- **Mermaid Diagram:**
```mermaid
flowchart LR
    U2[User Question] --> O2

    subgraph LangGraph Workflow 02
        O2[Orchestrator]
        D2[Deterministic Analyst]
        N2[News Fetcher]
        A2[Archiver]
        R2[Retriever]
        AD2[Advisor]
        V2[Validator]
    end

    O2 --> D2 --> N2 --> A2 --> R2 --> AD2 --> V2

    O2 -.->|updates| State2
    D2 -.->|updates| State2
    N2 -.->|updates| State2
    A2 -.->|updates| State2
    R2 -.->|updates| State2
    AD2 -.->|updates| State2
    V2 -.->|updates| State2

    State2[(Shared State Store)]

    V2 --> PDF2[PDF Report]
    V2 --> Console2[Console Output]
```
