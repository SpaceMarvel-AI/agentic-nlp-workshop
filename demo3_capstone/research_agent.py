"""
Demo 3 -- Capstone: Domain-Specific Research Agent
==================================================
Combines RAG (local FAISS index) + Wikipedia search + GPT summarisation
into a single research pipeline with conversation summary memory.
"""

import os
import sys
import time
import warnings
from pathlib import Path

if not sys.warnoptions:
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-W", "ignore"] + sys.argv))

warnings.filterwarnings("ignore")

import requests
from dotenv import load_dotenv

# -- Load environment ------------------------------------------------------
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

# -- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "data" / "sample_doc.pdf")
FAISS_DIR = str(PROJECT_ROOT / "data" / "faiss_index")

# -- LLM ------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# =========================================================================
#  1. FAISS Vector Store -- reuse or rebuild
# =========================================================================
embeddings = OpenAIEmbeddings()


def load_or_build_vectorstore():
    if Path(FAISS_DIR).exists():
        print("[VECTORSTORE] Loading existing FAISS index from", FAISS_DIR)
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    print("[VECTORSTORE] No saved index found -- building from PDF ...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"[VECTORSTORE] Loaded {len(pages)} page(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"[VECTORSTORE] Created {len(chunks)} chunk(s)")

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_DIR)
    print(f"[VECTORSTORE] Index saved to {FAISS_DIR}")
    return vs


# -- Banner ----------------------------------------------------------------
print("\n" + "#" * 60)
print("  DEMO 3: Capstone -- Domain-Specific Research Agent")
print("  RAG + Wikipedia + Summarisation + Summary Memory")
print("#" * 60)
time.sleep(2)

try:
    vectorstore = load_or_build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    sys.exit(f"[ERROR] Vector store failed -- check OPENAI_API_KEY: {e}")
time.sleep(2)

# =========================================================================
#  2. Tools
# =========================================================================

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
        resp = requests.get(url, timeout=10, headers={"User-Agent": "agentic-nlp-workshop/1.0"})
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

# =========================================================================
#  3. Memory -- ConversationSummaryMemory
# =========================================================================
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# =========================================================================
#  4. Agent
# =========================================================================
agent = initialize_agent(
    tools=[rag_tool, wikipedia_tool, summarise_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_PREFIX},
)

# =========================================================================
#  5. Formatted output
# =========================================================================

def research(question: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  RESEARCH QUERY: {question}")
    print(f"{'=' * 60}\n")
    time.sleep(2)

    print("[STEP 1] Searching local docs ...")
    local_results = rag_search(question)
    time.sleep(2)

    print("[STEP 2] Searching Wikipedia ...")
    wiki_results = wikipedia_search(question)
    time.sleep(2)

    sources = {"local_docs": local_results, "wikipedia": wiki_results}

    print("[STEP 3] Agent reasoning ...\n")
    try:
        response = agent.invoke({"input": question})
        answer = response.get("output", str(response))
    except Exception as e:
        print(f"[ERROR] Agent call failed: {e}")
        return {"sources": sources, "answer": "Error", "key_points": ""}
    time.sleep(2)

    print("\n[STEP 4] Extracting key points ...")
    try:
        combined = f"{local_results}\n\n{wiki_results}\n\nAgent answer: {answer}"
        key_points = summarise_chain.invoke({"text": combined})
    except Exception as e:
        key_points = f"Summarisation failed: {e}"

    # -- Display -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  SOURCES USED")
    print(f"{'=' * 60}")
    print(f"  Local docs : {'Yes' if 'Chunk' in local_results else 'No relevant docs'}")
    print(f"  Wikipedia  : {'Yes' if 'Wikipedia:' not in wiki_results else 'No article found'}")

    print(f"\n{'=' * 60}")
    print("  ANSWER")
    print(f"{'=' * 60}")
    print(f"  {answer}")

    print(f"\n{'=' * 60}")
    print("  KEY POINTS")
    print(f"{'=' * 60}")
    for line in key_points.strip().split("\n"):
        print(f"  {line.strip()}")
    print(f"{'=' * 60}\n")

    return {"sources": sources, "answer": answer, "key_points": key_points}


# =========================================================================
#  6. Demo sequence
# =========================================================================
DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation and why is it important?",
    "What are AI agents and what frameworks exist for building them?",
    "Summarise everything we've discussed so far.",
]


def main():
    print(f"  Tools  : rag_search, wikipedia_search, summarise")
    print(f"  Memory : ConversationSummaryMemory")
    print(f"  System : \"{SYSTEM_PREFIX[:60]}...\"")
    print("#" * 60)
    time.sleep(2)

    for question in DEMO_QUESTIONS:
        research(question)

    print("=" * 60)
    print("  CONVERSATION SUMMARY (from ConversationSummaryMemory)")
    print("=" * 60)
    print(f"  {memory.buffer}")
    print()


if __name__ == "__main__":
    main()
