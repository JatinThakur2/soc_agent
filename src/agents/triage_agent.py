"""Triage Agent — severity classification & escalation decision."""
import json
from langchain.prompts import ChatPromptTemplate
from src.agents.llm_factory import make_llm
from src.agents.json_utils import safe_parse


TRIAGE_SYSTEM = """You are a Tier-1 SOC Triage Analyst. Given a structured alert packet,
determine the severity level and escalation status.

Consider:
- Confidence score (>0.9 = high trust, <0.6 = low confidence, needs investigation)
- Attack type severity hierarchy: DDoS/Infiltration > DoS/Web Attack > PortScan > Benign
- Flag anomalies (SYN-without-ACK = possible SYN flood)
- Targeted service criticality (SSH/RDP/DB = higher risk than HTTP)

Severity levels:
- Critical: Active exploitation, high confidence, critical service
- High:     Confirmed attack, notable service
- Medium:   Likely attack, lower impact service
- Low:      Suspicious but possibly benign
- Informational: No action required

Output ONLY valid JSON (no prose outside JSON):
{{
  "severity": "Critical|High|Medium|Low|Informational",
  "escalate": true|false,
  "reasoning": "2-3 sentences explaining your decision"
}}
"""

TRIAGE_USER = "Alert packet:\n{alert_json}"


def build_triage_chain(cfg: dict):
    llm = make_llm(cfg)
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM),
        ("human", TRIAGE_USER),
    ])
    return prompt | llm


def run_triage(chain, alert: dict) -> dict:
    result = chain.invoke({"alert_json": json.dumps(alert, indent=2)})
    text = result.content if hasattr(result, "content") else str(result)
    return safe_parse(text, {"severity": "Unknown", "escalate": False, "reasoning": text[:500]})
