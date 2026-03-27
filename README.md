# Agentic NLP Workshop

A hands-on workshop exploring agentic NLP patterns using LangChain, RAG, and AI agents.

## Project Structure

```
agentic-nlp-workshop/
├── demo1_rag/              # Retrieval-Augmented Generation demo
│   └── rag_demo.py         #   PDF -> Chunks -> FAISS -> LLM Q&A
├── demo2_agent/            # AI Agent with tool use demo
│   └── agent_demo.py       #   ReAct agent + calculator + Wikipedia
├── demo3_capstone/         # Capstone project combining RAG + Agents
│   ├── research_agent.py   #   RAG + Wikipedia + summarisation agent
│   ├── hallucination_guard.py  # Fact-checking claims against sources
│   └── api.py              #   FastAPI wrapper with HTML test page
├── demo4_usecases/         # Real-world use case examples
│   ├── legal_agent.py      #   Indian IPC legal research assistant
│   ├── healthcare_triage.py#   Symptom triage with guardrails
│   ├── saas_copilot.py     #   Finixy SaaS customer support agent
│   └── run_all_demos.py    #   Run all use cases in sequence
├── utils/                  # Shared utility functions
├── data/                   # Sample data files
│   └── sample_doc.pdf      #   2-page PDF on AI and agents
├── start_workshop.sh       # Interactive launcher script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

## Setup

1. Create a conda environment:
   ```bash
   conda create -n workshop python=3.11 -y
   conda activate workshop
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-...
   ```

## Quick Start (Interactive Menu)

```bash
conda activate workshop
bash start_workshop.sh
```

This checks your `.env`, installs dependencies, and shows a numbered menu to pick any demo.

## Run Individual Demos

All commands assume you are in the project root (`agentic-nlp-workshop/`) with the `workshop` conda environment activated.

### Demo 1 -- RAG (Interactive Q&A)
```bash
python demo1_rag/rag_demo.py
```
Loads a PDF, embeds chunks into FAISS, then opens an interactive prompt. Type a question, get an answer. Switch prompt strategy with `mode 2` (few-shot) or `mode 3` (chain-of-thought).

### Demo 2 -- ReAct Agent (Tools + Memory)
```bash
python demo2_agent/agent_demo.py
```
Runs 3 preset questions showing the full Thought -> Action -> Observation loop. Uses a calculator tool, Wikipedia search, and conversation memory.

### Demo 3 -- Capstone Research Agent
```bash
python demo3_capstone/research_agent.py
```
Combines RAG + Wikipedia + GPT summarisation with ConversationSummaryMemory. Runs 3 research queries with formatted output (Sources, Answer, Key Points).

### Demo 3b -- Hallucination Guard
```bash
python demo3_capstone/hallucination_guard.py
```
Extracts factual claims from agent output, verifies each against source chunks, and flags unverified claims with `[UNVERIFIED]`.

### Demo 3c -- FastAPI Research Agent API
```bash
python demo3_capstone/api.py
```
Starts a FastAPI server on http://localhost:8000 with:
- `GET /` -- HTML test page
- `POST /query` -- JSON `{"question": "...", "session_id": "..."}`
- `GET /health` -- Health check
- `GET /sessions/{id}/history` -- Session history

### Demo 4a -- Legal Research Assistant
```bash
python demo4_usecases/legal_agent.py
```
Indian IPC section lookup with structured output (Legal Basis, Risk Level, Recommendation).

### Demo 4b -- Healthcare Triage
```bash
python demo4_usecases/healthcare_triage.py
```
Symptom-to-urgency mapping with guardrails that block "diagnose" / "cure" in responses.

### Demo 4c -- SaaS Support Copilot
```bash
python demo4_usecases/saas_copilot.py
```
Finixy accounting app FAQ agent with 15 FAQ entries and sliding window memory (k=5).

### Demo 4 -- All Use Cases
```bash
python demo4_usecases/run_all_demos.py
```
Runs a quick showcase of all three use cases in sequence.
