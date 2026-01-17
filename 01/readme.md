# LLM-Fine-Tuning-Lab
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

- **Mermaid Diagram:**
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
