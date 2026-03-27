"""
Demo 4 -- Run All Use Case Demos (30-second showcase each)
===========================================================
Imports each use-case agent and runs a quick demo with 1-2 sample inputs.
"""

import sys
import time
import warnings
from pathlib import Path

if not sys.warnoptions:
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-W", "ignore"] + sys.argv))

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def divider(title: str):
    print("\n")
    print("*" * 60)
    print(f"  {title}")
    print("*" * 60)
    time.sleep(2)


def run_legal_demo():
    divider("USE CASE 1: Legal Research Assistant (Indian IPC)")
    from demo4_usecases.legal_agent import ask_legal

    ask_legal("What is the punishment for cheating under Indian law?")
    ask_legal("Someone is threatening me with death. What IPC section applies?")


def run_healthcare_demo():
    divider("USE CASE 2: Healthcare Symptom Triage Agent")
    from demo4_usecases.healthcare_triage import triage

    triage("I have chest pain and difficulty breathing")
    triage("I have had a mild fever for two days")


def run_saas_demo():
    divider("USE CASE 3: Finixy SaaS Customer Support Copilot")
    from demo4_usecases.saas_copilot import support_query

    support_query("How much does Finixy cost?")
    support_query("How do I create and send an invoice?")
    support_query("What was my first question?")


def main():
    print("\n" + "=" * 60)
    print("  AGENTIC NLP WORKSHOP -- USE CASE SHOWCASE")
    print("  Running 3 domain-specific agents with sample inputs")
    print("=" * 60)
    time.sleep(2)

    start = time.time()

    run_legal_demo()
    run_healthcare_demo()
    run_saas_demo()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  ALL DEMOS COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
