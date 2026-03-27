"""
Hallucination Guard
===================
Takes agent output and source chunks, then checks whether each factual claim
in the output is supported by the sources.  Unsupported claims are flagged
with [UNVERIFIED].
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
    sys.exit("ERROR: OPENAI_API_KEY not set. Copy .env.example -> .env and add your key.")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# =========================================================================
#  1. Claim extractor
# =========================================================================
EXTRACT_CLAIMS_PROMPT = ChatPromptTemplate.from_template(
    "Extract every distinct factual claim from the text below.\n"
    "Return one claim per line, numbered. Only factual statements -- "
    "skip opinions, hedging, and meta-commentary.\n\n"
    "Text:\n{text}\n\n"
    "Claims:"
)

extract_chain = EXTRACT_CLAIMS_PROMPT | llm | StrOutputParser()

# =========================================================================
#  2. Claim verifier
# =========================================================================
VERIFY_CLAIM_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict fact-checker. Your job is to determine if a claim "
    "can be verified ENTIRELY from the source text below.\n\n"
    "RULES:\n"
    "- ONLY use the source text. Do NOT use your own knowledge.\n"
    "- If the claim contains a proper name (person, company) that does NOT "
    "appear in the source text, respond UNSUPPORTED.\n"
    "- If the claim contains a specific date or year that does NOT appear "
    "in the source text, respond UNSUPPORTED.\n"
    "- The claim must be fully supported, not partially.\n\n"
    "Source text:\n{sources}\n\n"
    "Claim: {claim}\n\n"
    "First, list the key entities/dates in the claim and check if each "
    "appears in the source. Then respond with FINAL VERDICT: SUPPORTED "
    "or UNSUPPORTED"
)

verify_chain = VERIFY_CLAIM_PROMPT | llm | StrOutputParser()


# =========================================================================
#  3. Main guard function
# =========================================================================
def check_hallucinations(agent_output: str, source_chunks: list[str]) -> str:
    sources_text = "\n\n".join(source_chunks)

    print("[GUARD] Extracting factual claims from agent output ...")
    time.sleep(2)
    try:
        raw_claims = extract_chain.invoke({"text": agent_output})
    except Exception as e:
        print(f"[ERROR] Claim extraction failed: {e}")
        return agent_output

    claims = [
        line.split(". ", 1)[-1].strip()
        for line in raw_claims.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    if not claims:
        print("[GUARD] No factual claims found to verify.")
        return agent_output

    print(f"[GUARD] Found {len(claims)} claim(s) to verify\n")
    time.sleep(2)

    results = []
    for i, claim in enumerate(claims, 1):
        try:
            verdict = verify_chain.invoke({"sources": sources_text, "claim": claim}).strip()
            verdict_upper = verdict.upper()
            if "FINAL VERDICT:" in verdict_upper:
                final = verdict_upper.split("FINAL VERDICT:")[-1].strip()
                supported = final.startswith("SUPPORTED")
            else:
                supported = verdict_upper.strip().endswith("SUPPORTED")
        except Exception as e:
            print(f"  [?] ERROR verifying claim {i}: {e}")
            supported = False

        tag = "SUPPORTED" if supported else "UNVERIFIED"
        symbol = "+" if supported else "!"
        results.append({"claim": claim, "supported": supported, "tag": tag})
        print(f"  [{symbol}] {tag}: {claim}")

    # Build annotated output
    annotated = agent_output
    unverified = [r for r in results if not r["supported"]]
    for item in unverified:
        annotated = annotated.replace(item["claim"], f"[UNVERIFIED] {item['claim']}")

    # Report
    total = len(results)
    verified = sum(1 for r in results if r["supported"])
    flagged = total - verified

    print(f"\n{'=' * 60}")
    print("  HALLUCINATION GUARD REPORT")
    print(f"{'=' * 60}")
    print(f"  Total claims  : {total}")
    print(f"  Supported     : {verified}")
    print(f"  Unverified    : {flagged}")
    print(f"  Trust score   : {verified}/{total} ({100 * verified // total if total else 0}%)")
    print(f"{'=' * 60}")

    if unverified:
        print("\n  Flagged claims:")
        for item in unverified:
            print(f"    [UNVERIFIED] {item['claim']}")
    else:
        print("\n  All claims verified against sources.")

    print(f"{'=' * 60}\n")

    return annotated


# =========================================================================
#  4. Standalone demo
# =========================================================================
def main():
    print("\n" + "#" * 60)
    print("  DEMO 3b: Hallucination Guard")
    print("  Extract claims -> Verify against sources -> Flag")
    print("#" * 60)
    time.sleep(2)

    agent_output = (
        "Retrieval-Augmented Generation (RAG) combines large language models with "
        "external knowledge retrieval to improve factual accuracy. "
        "RAG was invented by Facebook AI Research in 2020. "
        "It reduces hallucinations by grounding responses in retrieved documents. "
        "AI agents can perceive their environment, reason about goals, and execute "
        "actions using available tools. "
        "LangChain was created by Harrison Chase in October 2022. "
        "The transformer architecture was developed in 2017."
    )

    source_chunks = [
        "Retrieval-Augmented Generation (RAG) is an important pattern in modern AI "
        "applications. RAG combines the power of large language models with external "
        "knowledge retrieval, allowing systems to access up-to-date and domain-specific "
        "information beyond their training data. This approach significantly reduces "
        "hallucinations and improves the factual accuracy of AI-generated responses.",

        "AI agents represent the next frontier in artificial intelligence. An AI agent "
        "is an autonomous system that can perceive its environment, reason about goals, "
        "plan sequences of actions, and execute those actions using available tools.",

        "The development of transformer architectures in 2017 marked a pivotal moment, "
        "leading to large language models (LLMs) that can generate human-like text.",
    ]

    print("\n[INPUT] Agent output:")
    print(f"  {agent_output}\n")
    print("[INPUT] Source chunks: 3 chunks provided\n")
    time.sleep(2)

    annotated = check_hallucinations(agent_output, source_chunks)

    print("[OUTPUT] Annotated text:")
    print(f"  {annotated}\n")


if __name__ == "__main__":
    main()
