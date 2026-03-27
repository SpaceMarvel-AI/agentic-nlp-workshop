"""
Use Case 2 -- Healthcare Symptom Triage Agent
==============================================
A symptom triage agent with urgency mapping, guardrails that block
any response containing "diagnose" or "cure", and a mandatory
disclaimer to consult a doctor.
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
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferMemory

# -- Symptom-to-Urgency Mapping -------------------------------------------
SYMPTOM_URGENCY = {
    "chest pain": ("Emergency", "Call emergency services (112) immediately. Do not wait."),
    "difficulty breathing": ("Emergency", "Seek emergency care immediately. Call 112."),
    "severe bleeding": ("Emergency", "Apply pressure to wound. Call emergency services."),
    "unconsciousness": ("Emergency", "Call 112. Check airway and breathing. Do not move the person."),
    "stroke symptoms": ("Emergency", "Act FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 112."),
    "high fever": ("Urgent", "Visit a doctor within 24 hours. Stay hydrated and monitor temperature."),
    "fever": ("Non-urgent", "Rest, stay hydrated, and monitor. See a doctor if it persists beyond 3 days."),
    "persistent cough": ("Non-urgent", "See a doctor if it lasts more than 2 weeks. Stay hydrated."),
    "headache": ("Non-urgent", "Rest in a dark room. If severe or recurring, consult a doctor."),
    "mild nausea": ("Non-urgent", "Stay hydrated. Eat light foods. See a doctor if it persists."),
    "sprained ankle": ("Non-urgent", "RICE: Rest, Ice, Compression, Elevation. See a doctor if pain is severe."),
    "skin rash": ("Non-urgent", "Avoid scratching. See a dermatologist if it spreads or persists."),
    "abdominal pain": ("Urgent", "Visit a doctor within 24 hours. Note location and severity."),
    "allergic reaction": ("Urgent", "If swelling of face/throat: Emergency. Otherwise, take antihistamine and see a doctor."),
    "dizziness": ("Urgent", "Sit or lie down. If persistent or with other symptoms, see a doctor promptly."),
}


def check_symptoms(symptom_text: str) -> str:
    symptom_lower = symptom_text.lower().strip()
    matches = []
    for symptom, (urgency, advice) in SYMPTOM_URGENCY.items():
        if symptom in symptom_lower or any(word in symptom_lower for word in symptom.split()):
            matches.append(f"[{urgency.upper()}] {symptom.title()}: {advice}")
    if matches:
        return "\n".join(matches)
    return "Symptom not found in triage database. Please consult a healthcare professional."


symptom_tool = Tool(
    name="check_symptoms",
    func=check_symptoms,
    description="Checks symptom urgency level. Input: symptom description.",
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

SYSTEM_PROMPT = (
    "You are a medical triage assistant. Always recommend consulting a doctor. "
    "Never diagnose. Never claim to cure. "
    "Always use the check_symptoms tool to assess urgency before responding."
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[symptom_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
)

# -- Guardrail -------------------------------------------------------------
BLOCKED_WORDS = ["diagnose", "diagnosis", "cure", "cures", "curing", "diagnosing"]


def apply_guardrail(response: str) -> str:
    response_lower = response.lower()
    flagged = [w for w in BLOCKED_WORDS if w in response_lower]
    if flagged:
        print(f"  [GUARDRAIL] BLOCKED -- response contained: {', '.join(flagged)}")
        return (
            "I'm unable to provide a diagnosis or claim any cures. "
            "Based on your symptoms, I recommend consulting a qualified "
            "healthcare professional for proper evaluation and treatment."
        )
    return response


def triage(symptoms: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  SYMPTOM TRIAGE: {symptoms}")
    print(f"{'=' * 60}\n")
    time.sleep(2)

    print("[TRIAGE LOOKUP]")
    lookup = check_symptoms(symptoms)
    for line in lookup.split("\n"):
        print(f"  {line}")
    print()
    time.sleep(2)

    print("[AGENT] Reasoning ...")
    try:
        response = agent.invoke({"input": f"I am experiencing: {symptoms}"})
        raw_answer = response.get("output", str(response))
    except Exception as e:
        print(f"[ERROR] Agent call failed: {e}")
        return {"urgency": "UNKNOWN", "response": "Error", "raw": ""}
    time.sleep(2)

    print("\n[GUARDRAIL] Checking response ...")
    safe_answer = apply_guardrail(raw_answer)

    if "[EMERGENCY]" in lookup:
        urgency = "EMERGENCY"
    elif "[URGENT]" in lookup:
        urgency = "URGENT"
    else:
        urgency = "NON-URGENT"

    print(f"\n{'=' * 60}")
    print("  TRIAGE RESULT")
    print(f"{'=' * 60}")
    print(f"  Urgency : {urgency}")
    print(f"  Response: {safe_answer}")
    print(f"{'=' * 60}")
    print("  ** Please consult a qualified healthcare professional. **")
    print(f"{'=' * 60}\n")

    return {"urgency": urgency, "response": safe_answer, "raw": raw_answer}


DEMO_CASES = [
    "I have chest pain and difficulty breathing",
    "I have had a mild fever for two days",
    "I have a headache and some nausea",
]


def main():
    print("\n" + "#" * 60)
    print("  USE CASE 2: Healthcare Symptom Triage Agent")
    print("  Symptom lookup + urgency mapping + guardrails")
    print("#" * 60)
    time.sleep(2)

    for symptoms in DEMO_CASES:
        triage(symptoms)


if __name__ == "__main__":
    main()
