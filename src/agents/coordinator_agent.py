"""Coordinator Agent — synthesizes final triage report."""
import json
from langchain.prompts import ChatPromptTemplate
from src.agents.llm_factory import make_llm
from src.agents.json_utils import safe_parse


COORDINATOR_SYSTEM = """You are the SOC Coordinator. Synthesize all agent outputs into
a concise triage report that a human analyst would review.

The report must cite the reasoning of each prior agent and produce a clear recommendation.
Keep the executive summary to 2-3 sentences.

Output ONLY valid JSON (no prose outside JSON):
{{
  "executive_summary": "2-3 sentence summary for the analyst",
  "final_severity": "Critical|High|Medium|Low|Informational",
  "attck_techniques": ["T1498", "..."],
  "recommended_actions": ["action 1", "action 2"],
  "reasoning_chain": {{
    "triage": "1 sentence on what triage decided",
    "investigation": "1 sentence on what investigation found",
    "response": "1 sentence on what response recommended"
  }},
  "analyst_action_required": true|false
}}
"""

COORDINATOR_USER = """Alert:
{alert_json}

Triage output:
{triage_json}

Investigation output:
{investigation_json}

Response output:
{response_json}
"""


def build_coordinator_chain(cfg: dict):
    llm = make_llm(cfg)
    prompt = ChatPromptTemplate.from_messages([
        ("system", COORDINATOR_SYSTEM),
        ("human", COORDINATOR_USER),
    ])
    return prompt | llm


def run_coordinator(chain, alert, triage_result, investigation_result, response_result) -> dict:
    result = chain.invoke({
        "alert_json": json.dumps(alert, indent=2),
        "triage_json": json.dumps(triage_result, indent=2),
        "investigation_json": json.dumps(investigation_result, indent=2),
        "response_json": json.dumps(response_result, indent=2),
    })
    text = result.content if hasattr(result, "content") else str(result)
    return safe_parse(text, {
        "executive_summary": text[:500],
        "final_severity": "Unknown",
        "attck_techniques": [],
        "recommended_actions": [],
        "reasoning_chain": {},
        "analyst_action_required": True,
    })
