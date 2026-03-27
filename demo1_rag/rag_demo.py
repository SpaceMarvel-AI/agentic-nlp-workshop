"""
Demo 1 -- Retrieval-Augmented Generation (RAG) with FAISS & LangChain
=====================================================================
Loads a PDF, chunks it, embeds the chunks into a FAISS vector store,
and lets the user query the document interactively using three different
prompt strategies: zero-shot, few-shot, and chain-of-thought.
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

from dotenv import load_dotenv

# -- Load environment ------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Copy .env.example -> .env and add your key.")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -- Paths -----------------------------------------------------------------
PDF_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample_doc.pdf")

# -- Banner ----------------------------------------------------------------
print("\n" + "=" * 60)
print("  DEMO 1: Retrieval-Augmented Generation (RAG)")
print("  PDF -> Chunks -> Embeddings -> FAISS -> LLM Q&A")
print("=" * 60)
time.sleep(2)

# -- 1. Load PDF -----------------------------------------------------------
print("\n[LOADING] Reading PDF from", PDF_PATH)
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"[LOADING] Loaded {len(pages)} page(s)")
time.sleep(2)

# -- 2. Split into chunks --------------------------------------------------
print("\n[SPLITTING] Chunking with RecursiveCharacterTextSplitter (chunk_size=500) ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"[SPLITTING] Created {len(chunks)} chunk(s)")
time.sleep(2)

# -- 3. Embed & store in FAISS --------------------------------------------
print("\n[EMBEDDING] Creating OpenAI embeddings and building FAISS index ...")
try:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("[EMBEDDING] FAISS index ready (top-k = 3)")
except Exception as e:
    sys.exit(f"[ERROR] Embedding failed -- check your OPENAI_API_KEY: {e}")
time.sleep(2)

# -- 4. Prompt templates ---------------------------------------------------

ZERO_SHOT_TEMPLATE = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

FEW_SHOT_TEMPLATE = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context. "
    "Here are two examples of good answers:\n\n"
    "Example 1:\n"
    "  Q: What is machine learning?\n"
    "  A: Machine learning is a subset of AI that enables systems to learn and "
    "improve from data without being explicitly programmed.\n\n"
    "Example 2:\n"
    "  Q: What is RAG?\n"
    "  A: RAG (Retrieval-Augmented Generation) combines large language models with "
    "external knowledge retrieval to access up-to-date, domain-specific information.\n\n"
    "Now answer the following question in the same concise style.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

COT_TEMPLATE = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context. "
    "Think step by step: first identify the relevant facts from the context, "
    "then reason through them, and finally give a clear answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Step-by-step reasoning and answer:"
)

PROMPTS = {
    "1": ("Zero-Shot", ZERO_SHOT_TEMPLATE),
    "2": ("Few-Shot (2 examples)", FEW_SHOT_TEMPLATE),
    "3": ("Chain-of-Thought", COT_TEMPLATE),
}

# -- 5. Build retrieval chain helper ---------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(prompt_template):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )


# -- 6. Interactive CLI loop -----------------------------------------------
def main():
    print("\n" + "-" * 60)
    print("  Interactive Q&A over sample_doc.pdf")
    print("-" * 60)
    print("Prompt modes:")
    for key, (name, _) in PROMPTS.items():
        print(f"  [{key}] {name}")
    print("\nType 'mode <number>' to switch prompt (default: 1 -- Zero-Shot).")
    print("Type 'quit' to exit.\n")

    current_mode = "1"

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Handle mode switching
        if question.lower().startswith("mode "):
            choice = question.split(maxsplit=1)[1].strip()
            if choice in PROMPTS:
                current_mode = choice
                print(f"[MODE] Switched to {PROMPTS[current_mode][0]}\n")
            else:
                print(f"[MODE] Unknown mode '{choice}'. Choose 1, 2, or 3.\n")
            continue

        mode_name, prompt_template = PROMPTS[current_mode]
        print(f"\n[MODE] Using: {mode_name}")

        # Retrieve
        print("[RETRIEVING] Fetching top-3 relevant chunks ...")
        docs = retriever.invoke(question)
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"  chunk {i} (page {page}): {snippet}...")
        time.sleep(2)

        # Answer
        print("[ANSWER] Generating response with gpt-3.5-turbo ...")
        try:
            chain = build_chain(prompt_template)
            answer = chain.invoke(question)
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            continue

        print(f"\n{'-' * 40}")
        print(answer)
        print(f"{'-' * 40}\n")


if __name__ == "__main__":
    main()
