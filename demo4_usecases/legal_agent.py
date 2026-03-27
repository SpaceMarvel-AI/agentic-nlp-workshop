"""
Use Case 1 -- Legal Research Assistant (Indian Law / IPC)
=========================================================
A ReAct agent with a hardcoded IPC section lookup tool that returns
structured responses with Legal Basis, Risk Level, and Recommendation.
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

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set.")

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferMemory

# -- IPC Sections Database ------------------------------------------------
IPC_SECTIONS = {
    302: "Punishment for murder. Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
    304: "Punishment for culpable homicide not amounting to murder. Imprisonment for life, or imprisonment up to 10 years, and fine.",
    306: "Abetment of suicide. If any person commits suicide, whoever abets the commission shall be punished with imprisonment up to 10 years and fine.",
    354: "Assault or criminal force to woman with intent to outrage her modesty. Imprisonment of not less than 1 year extendable to 5 years and fine.",
    376: "Punishment for rape. Rigorous imprisonment of not less than 10 years extendable to life imprisonment, and fine.",
    420: "Cheating and dishonestly inducing delivery of property. Imprisonment up to 7 years and fine.",
    498: "Enticing or taking away or detaining with criminal intent a married woman. Imprisonment up to 2 years, or fine, or both.",
    499: "Defamation. Whoever by words, signs, or visible representations makes or publishes any imputation concerning any person, with intent to harm.",
    500: "Punishment for defamation. Simple imprisonment up to 2 years, or fine, or both.",
    506: "Punishment for criminal intimidation. Imprisonment up to 2 years, or fine, or both. If threat is of death or grievous hurt: up to 7 years.",
}


def search_ipc(query: str) -> str:
    query_lower = query.lower().strip()
    for sec_num, desc in IPC_SECTIONS.items():
        if str(sec_num) in query_lower:
            return f"IPC Section {sec_num}: {desc}"
    matches = []
    for sec_num, desc in IPC_SECTIONS.items():
        if any(word in desc.lower() for word in query_lower.split()):
            matches.append(f"IPC Section {sec_num}: {desc}")
    if matches:
        return "\n\n".join(matches)
    return "No matching IPC sections found for the given query."


ipc_tool = Tool(
    name="search_ipc",
    func=search_ipc,
    description="Searches Indian Penal Code (IPC) sections by number or keyword.",
)

LEGAL_PROMPT = ChatPromptTemplate.from_template(
    "You are a legal research assistant specializing in Indian law (IPC). "
    "Based on the following legal context and question, provide a structured response.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Respond in EXACTLY this format:\n"
    "LEGAL BASIS: [cite the relevant IPC section(s) and their provisions]\n"
    "RISK LEVEL: [High / Medium / Low -- based on severity of punishment]\n"
    "RECOMMENDATION: [practical legal advice in 2-3 sentences]"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
legal_chain = LEGAL_PROMPT | llm | StrOutputParser()

SYSTEM_PROMPT = (
    "You are a legal research assistant specializing in Indian Penal Code (IPC). "
    "Always search for the relevant IPC section first using the search_ipc tool. "
    "Provide responses grounded in Indian law."
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[ipc_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
)


def ask_legal(question: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  LEGAL QUERY: {question}")
    print(f"{'=' * 60}\n")
    time.sleep(2)

    ipc_context = search_ipc(question)
    print(f"[IPC LOOKUP] {ipc_context[:100]}...\n")
    time.sleep(2)

    print("[AGENT] Reasoning ...")
    try:
        response = agent.invoke({"input": question})
        agent_answer = response.get("output", str(response))
    except Exception as e:
        print(f"[ERROR] Agent call failed: {e}")
        return {"agent_answer": "Error", "structured": ""}
    time.sleep(2)

    print("[STRUCTURING] Formatting response ...\n")
    try:
        structured = legal_chain.invoke({"context": ipc_context, "question": question})
    except Exception as e:
        print(f"[ERROR] Structuring failed: {e}")
        return {"agent_answer": agent_answer, "structured": ""}

    print("=" * 60)
    for line in structured.strip().split("\n"):
        line = line.strip()
        if line.startswith("LEGAL BASIS:"):
            print(f"  LEGAL BASIS    : {line[12:].strip()}")
        elif line.startswith("RISK LEVEL:"):
            print(f"  RISK LEVEL     : {line[11:].strip()}")
        elif line.startswith("RECOMMENDATION:"):
            print(f"  RECOMMENDATION : {line[15:].strip()}")
        elif line:
            print(f"  {line}")
    print("=" * 60)
    print("  DISCLAIMER: This is for informational purposes only.")
    print("  Consult a qualified lawyer for actual legal advice.")
    print("=" * 60 + "\n")

    return {"agent_answer": agent_answer, "structured": structured}


DEMO_QUESTIONS = [
    "What is the punishment for cheating under Indian law?",
    "Someone is threatening me with death. What IPC section applies?",
    "What are the legal consequences of defamation in India?",
]


def main():
    print("\n" + "#" * 60)
    print("  USE CASE 1: Legal Research Assistant (Indian IPC)")
    print("  IPC lookup + ReAct agent + structured output")
    print("#" * 60)
    time.sleep(2)

    for q in DEMO_QUESTIONS:
        ask_legal(q)


if __name__ == "__main__":
    main()
