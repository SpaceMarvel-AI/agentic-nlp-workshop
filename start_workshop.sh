#!/usr/bin/env bash
# ============================================================
#  Agentic NLP Workshop -- Launcher Script
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# -- Check .env ------------------------------------------------------------
if [ ! -f .env ]; then
    echo ""
    echo "  ERROR: .env file not found!"
    echo "  Copy .env.example to .env and add your OpenAI API key:"
    echo ""
    echo "    cp .env.example .env"
    echo "    # then edit .env and set OPENAI_API_KEY=sk-..."
    echo ""
    exit 1
fi

echo ""
echo "  .env found -- API key configured."

# -- Install requirements --------------------------------------------------
echo ""
echo "  Installing/checking Python dependencies ..."
pip install -q -r requirements.txt
echo "  Dependencies ready."

# -- Menu ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Agentic NLP Workshop"
echo "============================================================"
echo ""
echo "  1) Demo 1 -- RAG: PDF Q&A with FAISS (interactive)"
echo "  2) Demo 2 -- ReAct Agent: Tools + Memory"
echo "  3) Demo 3 -- Capstone: Research Agent (RAG + Wikipedia)"
echo "  4) Demo 3b - Hallucination Guard"
echo "  5) Demo 3c - FastAPI Research Agent API (port 8000)"
echo "  6) Demo 4a - Use Case: Legal Research (Indian IPC)"
echo "  7) Demo 4b - Use Case: Healthcare Triage"
echo "  8) Demo 4c - Use Case: SaaS Support Copilot (Finixy)"
echo "  9) Demo 4  - Run ALL Use Cases"
echo "  0) Exit"
echo ""
read -rp "  Choose a demo [0-9]: " choice

case "$choice" in
    1) python demo1_rag/rag_demo.py ;;
    2) python demo2_agent/agent_demo.py ;;
    3) python demo3_capstone/research_agent.py ;;
    4) python demo3_capstone/hallucination_guard.py ;;
    5) python demo3_capstone/api.py ;;
    6) python demo4_usecases/legal_agent.py ;;
    7) python demo4_usecases/healthcare_triage.py ;;
    8) python demo4_usecases/saas_copilot.py ;;
    9) python demo4_usecases/run_all_demos.py ;;
    0) echo "  Goodbye!" ; exit 0 ;;
    *) echo "  Invalid choice." ; exit 1 ;;
esac
