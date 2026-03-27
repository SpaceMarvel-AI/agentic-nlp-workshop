"""
Use Case 3 -- SaaS Customer Support Copilot for "Finixy" Accounting App
========================================================================
A support agent with FAQ lookup (15 entries), ConversationBufferWindowMemory(k=5),
and a friendly, professional tone.
"""

import os
import re
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
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferWindowMemory

# -- Finixy FAQ Database ---------------------------------------------------
FINIXY_FAQ = {
    "pricing": "Finixy offers three plans: Starter ($9/mo, 1 user), Pro ($29/mo, up to 5 users), and Enterprise ($99/mo, unlimited users). All plans include a 14-day free trial.",
    "free trial": "Yes! All Finixy plans come with a 14-day free trial. No credit card required to start. You can upgrade or cancel anytime.",
    "invoice": "To create an invoice: Go to Dashboard > Invoices > New Invoice. Fill in client details, line items, and click Send. Invoices can be sent via email or shareable link.",
    "export": "You can export data from Settings > Data Export. Supported formats: CSV, Excel, PDF. You can export invoices, transactions, reports, and client lists.",
    "integrations": "Finixy integrates with: Stripe, PayPal, Razorpay (payments), Gmail & Outlook (email), Slack (notifications), Google Sheets, and Zapier (1000+ apps).",
    "tax": "Finixy supports GST (India), VAT (EU), and Sales Tax (US). Configure your tax settings in Settings > Tax Configuration. Tax is auto-calculated on invoices.",
    "multi-currency": "Finixy supports 50+ currencies. Set your base currency in Settings > Currency. Invoices auto-convert using real-time exchange rates.",
    "reports": "Available reports: Profit & Loss, Balance Sheet, Cash Flow, Expense Breakdown, Tax Summary, and Accounts Receivable Aging. Find them under Reports tab.",
    "mobile app": "Finixy has mobile apps for iOS and Android. Download from App Store or Google Play. All features sync in real-time with the web app.",
    "security": "Finixy uses 256-bit AES encryption, SOC 2 Type II compliance, 2FA authentication, and daily backups. Your financial data is stored in AWS with 99.99% uptime SLA.",
    "cancel subscription": "To cancel: Go to Settings > Billing > Cancel Plan. Your data is retained for 30 days after cancellation. You can reactivate anytime within that period.",
    "refund": "Finixy offers a 30-day money-back guarantee on all paid plans. Contact support@finixy.com with your account email to request a refund.",
    "password reset": "Click 'Forgot Password' on the login page. Enter your email and we'll send a reset link. The link expires in 1 hour. Check spam folder if not received.",
    "api access": "API access is available on Pro and Enterprise plans. Find your API key in Settings > Developer > API Keys. Full API documentation at docs.finixy.com/api.",
    "support": "Support channels: In-app chat (all plans), Email support@finixy.com (all plans), Priority phone support (Enterprise only). Response time: <4 hours on business days.",
}


def lookup_faq(query: str) -> str:
    query_lower = query.lower().strip()

    # Exact topic match
    for topic, answer in FINIXY_FAQ.items():
        if topic in query_lower:
            return f"[FAQ: {topic.title()}] {answer}"

    # Keyword search -- score by number of matching words
    query_words = [w for w in re.findall(r"[a-z]+", query_lower) if len(w) > 3]
    scored = []
    for topic, answer in FINIXY_FAQ.items():
        combined = f"{topic} {answer}".lower()
        score = sum(1 for word in query_words if word in combined)
        if topic in query_lower:
            score += 3
        if score > 0:
            scored.append((score, f"[FAQ: {topic.title()}] {answer}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [entry for _, entry in scored]

    if matches:
        return "\n\n".join(matches[:3])
    return "No matching FAQ found. I'll do my best to help based on general product knowledge."


faq_tool = Tool(
    name="lookup_faq",
    func=lookup_faq,
    description="Searches the Finixy accounting app FAQ database. Input: question or keyword.",
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

SYSTEM_PROMPT = (
    "You are Finixy Support Copilot, a friendly and professional customer support "
    "agent for Finixy, a cloud-based accounting application. "
    "Always search the FAQ database first using lookup_faq. "
    "Be concise, helpful, and suggest next steps."
)

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[faq_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
)


def support_query(question: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"  CUSTOMER: {question}")
    print(f"{'=' * 60}\n")
    time.sleep(2)

    print("[FAQ SEARCH]")
    faq_result = lookup_faq(question)
    for line in faq_result.split("\n"):
        if line.strip():
            print(f"  {line.strip()[:100]}...")
    print()
    time.sleep(2)

    print("[COPILOT] Responding ...")
    try:
        response = agent.invoke({"input": question})
        answer = response.get("output", str(response))
    except Exception as e:
        print(f"[ERROR] Agent call failed: {e}")
        return "Sorry, something went wrong. Please try again."

    print(f"\n{'=' * 60}")
    print(f"  FINIXY COPILOT:")
    print(f"  {answer}")
    print(f"{'=' * 60}\n")
    time.sleep(2)

    return answer


DEMO_QUERIES = [
    "How much does Finixy cost?",
    "How do I create and send an invoice?",
    "Can I integrate Finixy with Stripe and Slack?",
    "I forgot my password, what should I do?",
    "What was the first question I asked?",
]


def main():
    print("\n" + "#" * 60)
    print("  USE CASE 3: Finixy SaaS Customer Support Copilot")
    print("  FAQ lookup + ReAct agent + sliding window memory")
    print("#" * 60)
    print(f"  FAQ entries : {len(FINIXY_FAQ)}")
    print(f"  Memory      : ConversationBufferWindowMemory(k=5)")
    print("#" * 60)
    time.sleep(2)

    for q in DEMO_QUERIES:
        support_query(q)

    print("=" * 60)
    print("  MEMORY WINDOW (last 5 exchanges)")
    print("=" * 60)
    for msg in memory.chat_memory.messages:
        role = msg.type.upper()
        print(f"  [{role}] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
    print()


if __name__ == "__main__":
    main()
