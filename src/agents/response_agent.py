"""Response Agent — recommends containment & mitigation actions."""
import json
from langchain.prompts import ChatPromptTemplate
from src.agents.llm_factory import make_llm
from src.agents.json_utils import safe_parse


RESPONSE_SYSTEM = """You are a SOC Response Analyst. Given the triage result
and investigation findings, recommend concrete actions.

Consider:
- Severity (from triage)
- Confirmed ATT&CK techniques (from investigation)
- Service targeted (from alert)
- Whether evidence is strong enough to warrant blocking
- MITRE ATT&CK mitigation recommendations

Output ONLY valid JSON (no prose outside JSON):
{{
  "immediate_actions": ["action 1", "action 2"],
  "block_source": true|false,
  "block_justification": "1-sentence reason",
  "mitigation_steps": ["long-term step 1", "long-term step 2"],
  "analyst_action_required": true|false,
  "reasoning": "2-3 sentences on overall response strategy"
}}
"""

RESPONSE_USER = """Alert:
{alert_json}

Triage result:
{triage_json}

Investigation result:
{investigation_json}
"""


def build_response_chain(cfg: dict):
    llm = make_llm(cfg)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM),
        ("human", RESPONSE_USER),
    ])
    return prompt | llm


def run_response(chain, alert: dict, triage_result: dict, investigation_result: dict) -> dict:
    result = chain.invoke({
        "alert_json": json.dumps(alert, indent=2),
        "triage_json": json.dumps(triage_result, indent=2),
        "investigation_json": json.dumps(investigation_result, indent=2),
    })
    text = result.content if hasattr(result, "content") else str(result)
    return safe_parse(text, {
        "immediate_actions": [],
        "block_source": False,
        "block_justification": "Parse failed",
        "mitigation_steps": [],
        "analyst_action_required": True,
        "reasoning": text[:500],
    })
