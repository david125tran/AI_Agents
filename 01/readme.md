# Market Research Agent (LangGraph) — Node Flow

## Goal
Given a user question, the agent will:
1) plan the work,
2) search the web,
3) retrieve/extract key facts,
4) synthesize a structured report with citations,
5) validate that claims are grounded in cited sources (and fix gaps).

---

## Graph Overview

### Nodes
- **planner**: break the user request into a step-by-step plan + search queries
- **search**: run web search for the planned queries
- **retriever**: open top results and extract key facts + quotes/snippets + metadata
- **synthesizer**: write a report using extracted facts and attach citations
- **validator**: check every claim is supported by citations; flag gaps/hallucinations
- **revise** (optional loop): if validator finds gaps, refine queries and re-run search/retrieve

### State (shared across nodes)
- `question`: user’s original question
- `plan`: ordered steps (planner output)
- `queries`: list of search queries
- `search_results`: list of results `{title, url, snippet, score}`
- `sources`: list of opened sources `{url, title, published_date?, extracted_facts[]}`
- `notes`: normalized evidence objects `{claim, evidence, url, quote?, confidence}`
- `draft_report`: report text + citations
- `validation`: `{passed: bool, issues: [ ... ]}`
- `revision_round`: int (prevents infinite loops)

---

## Flow Diagram (Mermaid)

```mermaid
flowchart TD
  A[START: user question] --> B[planner]
  B --> C[search]
  C --> D[retriever]
  D --> E[synthesizer]
  E --> F[validator]

  F -->|passed| G[END: final report]
  F -->|issues found| H[revise queries + plan]
  H --> C

  style A fill:#f2f2f2,stroke:#999
  style G fill:#f2f2f2,stroke:#999
