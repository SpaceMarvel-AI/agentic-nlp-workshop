"""
Demo 2 -- ReAct Agent with Tools & Conversation Memory
======================================================
A LangChain ReAct agent equipped with two tools (calculator, wikipedia search)
and conversation memory.  The full Thought -> Action -> Observation loop is
printed so the audience can follow the agent's reasoning.
"""

import math
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

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import Tool

# -- 1. Tools --------------------------------------------------------------

# --- Calculator (safe eval) ---
SAFE_MATH_GLOBALS = {"__builtins__": {}, "abs": abs, "round": round, "int": int, "float": float}
SAFE_MATH_GLOBALS.update({name: getattr(math, name) for name in dir(math) if not name.startswith("_")})


def calculator(expression: str) -> str:
    """Evaluate a math expression safely (only math functions allowed)."""
    try:
        expression = expression.strip().strip("'\"")
        result = eval(expression, SAFE_MATH_GLOBALS, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description=(
        "Evaluates a math expression. Input must be a valid Python math "
        "expression (e.g. '1_000_000 / 4', 'sqrt(144)', 'log(100)')."
    ),
)

# --- Wikipedia Search ---
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"


def wikipedia_search(topic: str) -> str:
    """Fetch a 2-sentence summary from Wikipedia for the given topic."""
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
    description=(
        "Searches Wikipedia for a topic and returns a 2-sentence summary. "
        "Input should be a topic name (e.g. 'LangChain', 'Python programming')."
    ),
)

# -- 2. Memory -------------------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -- 3. Agent --------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = initialize_agent(
    tools=[calculator_tool, wikipedia_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# -- 4. Preset demo sequence -----------------------------------------------
DEMO_QUESTIONS = [
    "What is LangChain?",
    "How many tokens are in 1 million characters if average token is 4 chars?",
    "What was my last question?",
]


def run_demo():
    print("\n" + "=" * 60)
    print("  DEMO 2: ReAct Agent -- Tools + Conversation Memory")
    print("  Thought -> Action -> Observation reasoning loop")
    print("=" * 60)
    print(f"  Tools  : calculator, wikipedia_search")
    print(f"  Memory : ConversationBufferMemory")
    print(f"  Agent  : CONVERSATIONAL_REACT_DESCRIPTION (ReAct)")
    print("=" * 60)
    time.sleep(2)

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'#' * 60}")
        print(f"  DEMO QUESTION {i}/{len(DEMO_QUESTIONS)}")
        print(f"  >>> {question}")
        print(f"{'#' * 60}\n")
        time.sleep(2)

        try:
            response = agent.invoke({"input": question})
            answer = response.get("output", response)
        except Exception as e:
            print(f"[ERROR] Agent call failed: {e}")
            continue

        print(f"\n[ANSWER] {answer}\n")
        time.sleep(2)

    # Show final memory contents
    print("=" * 60)
    print("  MEMORY CONTENTS (conversation history)")
    print("=" * 60)
    for msg in memory.chat_memory.messages:
        role = msg.type.upper()
        print(f"  [{role}] {msg.content[:120]}{'...' if len(msg.content) > 120 else ''}")
    print()


if __name__ == "__main__":
    run_demo()
