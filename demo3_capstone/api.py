"""
Demo 3 – FastAPI wrapper for the Research Agent
================================================
Exposes the capstone research agent over HTTP with session management,
CORS support, and a built-in HTML test page.

Run:  uvicorn demo3_capstone.api:app --reload --port 8000
  or: python demo3_capstone/api.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import requests as http_requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Load environment ────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Copy .env.example -> .env and add your key.")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationSummaryMemory

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "data" / "sample_doc.pdf")
FAISS_DIR = str(PROJECT_ROOT / "data" / "faiss_index")

# ── Shared LLM & embeddings ────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# ── In-memory session store ─────────────────────────────────────────
sessions: dict[str, dict] = {}

# ════════════════════════════════════════════════════════════════════
#  Vector store
# ════════════════════════════════════════════════════════════════════
vectorstore = None
retriever = None


def load_or_build_vectorstore():
    if Path(FAISS_DIR).exists():
        print("[VECTORSTORE] Loading existing FAISS index from", FAISS_DIR)
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    print("[VECTORSTORE] Building from PDF ...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_DIR)
    print(f"[VECTORSTORE] Index built and saved ({len(chunks)} chunks)")
    return vs


# ════════════════════════════════════════════════════════════════════
#  Tools (same as research_agent.py)
# ════════════════════════════════════════════════════════════════════
def rag_search(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in local store."
    results = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        results.append(f"[Chunk {i}, page {page}]: {doc.page_content}")
    return "\n\n".join(results)


rag_tool = Tool(
    name="rag_search",
    func=rag_search,
    description="Searches the local document store. Use this FIRST. Input: search query.",
)

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"


def wikipedia_search(topic: str) -> str:
    try:
        url = WIKI_API.format(topic=topic.strip().replace(" ", "_"))
        resp = http_requests.get(url, timeout=10, headers={"User-Agent": "agentic-nlp-workshop/1.0"})
        if resp.status_code == 404:
            return f"Wikipedia: no article found for '{topic}'."
        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "")
        sentences = extract.split(". ")
        summary = ". ".join(sentences[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."
        return summary if summary else "No summary available."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


wikipedia_tool = Tool(
    name="wikipedia_search",
    func=wikipedia_search,
    description="Searches Wikipedia for a topic summary. Use AFTER rag_search. Input: topic name.",
)

SUMMARISE_PROMPT = ChatPromptTemplate.from_template(
    "Summarise the following text into exactly 3 concise bullet points.\n\n"
    "Text:\n{text}\n\nBullet points:"
)
summarise_chain = SUMMARISE_PROMPT | llm | StrOutputParser()


def summarise_tool_func(text: str) -> str:
    return summarise_chain.invoke({"text": text})


summarise_tool = Tool(
    name="summarise",
    func=summarise_tool_func,
    description="Summarises long text into 3 bullet points. Input: full text.",
)

SYSTEM_PREFIX = (
    "You are a research assistant. Always search local docs first, then Wikipedia. "
    "Summarise findings in bullet points."
)


# ════════════════════════════════════════════════════════════════════
#  Per-session agent factory
# ════════════════════════════════════════════════════════════════════
def get_or_create_session(session_id: str) -> dict:
    if session_id not in sessions:
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        agent = initialize_agent(
            tools=[rag_tool, wikipedia_tool, summarise_tool],
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": SYSTEM_PREFIX},
        )
        sessions[session_id] = {"agent": agent, "memory": memory, "history": []}
    return sessions[session_id]


# ════════════════════════════════════════════════════════════════════
#  FastAPI app
# ════════════════════════════════════════════════════════════════════
app = FastAPI(title="Research Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global vectorstore, retriever
    print("[STARTUP] Pre-loading vector store ...")
    vectorstore = load_or_build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("[STARTUP] Ready on http://localhost:8000")


# ── Request / response models ──────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: dict
    key_points: str
    session_id: str


# ── POST /query ─────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    session = get_or_create_session(req.session_id)
    agent = session["agent"]

    # Gather sources
    local_results = rag_search(req.question)
    wiki_results = wikipedia_search(req.question)
    sources = {
        "local_docs": "Yes" if "Chunk" in local_results else "No relevant docs",
        "wikipedia": "Yes" if "Wikipedia:" not in wiki_results else "No article found",
    }

    # Run agent
    response = agent.invoke({"input": req.question})
    answer = response.get("output", str(response))

    # Key points
    combined = f"{local_results}\n\n{wiki_results}\n\nAgent answer: {answer}"
    key_points = summarise_chain.invoke({"text": combined})

    # Save to history
    session["history"].append({"question": req.question, "answer": answer, "key_points": key_points})

    return QueryResponse(
        answer=answer,
        sources=sources,
        key_points=key_points,
        session_id=req.session_id,
    )


# ── GET /health ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "gpt-3.5-turbo"}


# ── GET /sessions/{session_id}/history ──────────────────────────────
@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "exchanges": session["history"],
        "summary": session["memory"].buffer,
    }


# ── GET / — HTML test page ──────────────────────────────────────────
TEST_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Research Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
           display: flex; justify-content: center; padding: 2rem; }
    .container { max-width: 720px; width: 100%; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    .subtitle { color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.9rem; }
    .input-row { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
    input[type=text] { flex: 1; padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid #334155;
                       background: #1e293b; color: #e2e8f0; font-size: 1rem; outline: none; }
    input[type=text]:focus { border-color: #3b82f6; }
    button { padding: 0.6rem 1.5rem; border-radius: 6px; border: none; background: #3b82f6;
             color: #fff; font-size: 1rem; cursor: pointer; }
    button:hover { background: #2563eb; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result { background: #1e293b; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;
              border: 1px solid #334155; }
    .result h3 { color: #3b82f6; font-size: 0.85rem; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .result p, .result pre { white-space: pre-wrap; word-wrap: break-word; line-height: 1.6; }
    .spinner { display: none; color: #94a3b8; margin-bottom: 1rem; }
    .history-link { color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem; }
    .history-link a { color: #60a5fa; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Research Agent API</h1>
    <p class="subtitle">Ask a question -- the agent searches local docs, Wikipedia, then summarises.</p>

    <div class="input-row">
      <input type="text" id="question" placeholder="e.g. What is RAG?" autofocus />
      <button id="ask" onclick="askQuestion()">Ask</button>
    </div>

    <div class="spinner" id="spinner">Thinking...</div>
    <div id="results"></div>
    <div class="history-link">
      Session: <strong id="sid">default</strong> |
      <a href="#" onclick="viewHistory(); return false;">View history</a>
    </div>
  </div>

  <script>
    const sessionId = 'session_' + Math.random().toString(36).slice(2, 8);
    document.getElementById('sid').textContent = sessionId;

    const input = document.getElementById('question');
    input.addEventListener('keydown', e => { if (e.key === 'Enter') askQuestion(); });

    async function askQuestion() {
      const q = input.value.trim();
      if (!q) return;

      const btn = document.getElementById('ask');
      const spinner = document.getElementById('spinner');
      btn.disabled = true;
      spinner.style.display = 'block';

      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, session_id: sessionId })
        });
        const data = await res.json();

        const html = `
          <div class="result"><h3>Answer</h3><p>${esc(data.answer)}</p></div>
          <div class="result"><h3>Key Points</h3><pre>${esc(data.key_points)}</pre></div>
          <div class="result"><h3>Sources</h3>
            <p>Local docs: ${esc(data.sources.local_docs)}<br>
               Wikipedia: ${esc(data.sources.wikipedia)}</p></div>`;
        document.getElementById('results').innerHTML = html;
      } catch (err) {
        document.getElementById('results').innerHTML =
          `<div class="result"><h3>Error</h3><p>${esc(err.message)}</p></div>`;
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    async function viewHistory() {
      try {
        const res = await fetch(`/sessions/${sessionId}/history`);
        const data = await res.json();
        let html = `<div class="result"><h3>Session History (${data.exchanges.length} exchanges)</h3>`;
        for (const ex of data.exchanges) {
          html += `<p><strong>Q:</strong> ${esc(ex.question)}<br><strong>A:</strong> ${esc(ex.answer)}</p><hr style="border-color:#334155;margin:0.5rem 0">`;
        }
        html += `<p style="color:#94a3b8;margin-top:0.5rem"><strong>Summary:</strong> ${esc(data.summary)}</p></div>`;
        document.getElementById('results').innerHTML = html;
      } catch (err) {
        document.getElementById('results').innerHTML =
          `<div class="result"><h3>Error</h3><p>${esc(err.message)}</p></div>`;
      }
    }

    function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def test_page():
    return TEST_PAGE_HTML


# ════════════════════════════════════════════════════════════════════
#  Direct run
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Add project root to sys.path so uvicorn can resolve "demo3_capstone.api"
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    uvicorn.run("demo3_capstone.api:app", host="0.0.0.0", port=8000, reload=True)
