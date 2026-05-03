"""Investigation Agent — queries MITRE ATT&CK RAG for evidence."""
import json
import numpy as np
from pathlib import Path
from langchain.prompts import ChatPromptTemplate

from src.utils.paths import get_path
from src.agents.llm_factory import make_llm
from src.agents.json_utils import safe_parse


INVESTIGATION_SYSTEM = """You are a SOC Investigation Analyst. Given an alert and
MITRE ATT&CK context, identify:
1. Which ATT&CK technique(s) match this alert
2. What evidence supports this mapping
3. What additional data would be needed to confirm

Output ONLY valid JSON (no prose outside JSON):
{{
  "techniques": [{{"id": "T1498", "name": "Network Denial of Service", "tactic": "impact"}}],
  "evidence": ["evidence item 1", "evidence item 2"],
  "gaps": ["missing data point 1", "..."],
  "confidence": "high|medium|low"
}}
"""

INVESTIGATION_USER = """Alert:
{alert_json}

Retrieved MITRE ATT&CK context:
{mitre_context}

Triage output:
{triage_json}
"""


class InvestigationAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = make_llm(cfg)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", INVESTIGATION_SYSTEM),
            ("human", INVESTIGATION_USER),
        ])
        self.chain = self.prompt | self.llm

        self._faiss_index = None
        self._techniques = None
        self._embedder = None

    def _get_vector_store(self):
        if self._faiss_index is None:
            import faiss
            from sentence_transformers import SentenceTransformer

            kb = get_path(self.cfg, "knowledge_base")
            index_path = kb / "faiss_index.bin"
            techniques_path = kb / "techniques.json"

            if not index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {index_path}. "
                    "Run: python -m src.knowledge_base.build"
                )

            self._faiss_index = faiss.read_index(str(index_path))
            self._techniques = json.loads(techniques_path.read_text())
            self._embedder = SentenceTransformer(self.cfg["rag"]["embedding_model"])

        return self._faiss_index, self._techniques, self._embedder

    def query_rag(self, query_text: str) -> list:
        index, techniques, embedder = self._get_vector_store()
        q_emb = embedder.encode([query_text], convert_to_numpy=True).astype("float32")
        import faiss
        faiss.normalize_L2(q_emb)

        k = self.cfg["rag"]["top_k_retrieval"]
        distances, indices = index.search(q_emb, k)

        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(techniques):
                continue
            t = techniques[idx]
            hits.append({
                "id": t["id"],
                "name": t["name"],
                "tactics": ", ".join(t.get("tactics", [])),
                "description_snippet": t.get("description", "")[:400],
                "score": float(dist),
            })
        return hits

    def run(self, alert: dict, triage_result: dict) -> dict:
        query = f"{alert.get('predicted_label', '')} {alert.get('service', '')} attack"
        if alert.get("syn_no_ack"):
            query += " SYN flood no ACK"

        mitre_hits = self.query_rag(query)

        result = self.chain.invoke({
            "alert_json": json.dumps(alert, indent=2),
            "mitre_context": json.dumps(mitre_hits, indent=2),
            "triage_json": json.dumps(triage_result, indent=2),
        })

        text = result.content if hasattr(result, "content") else str(result)
        parsed = safe_parse(text, {
            "techniques": [], "evidence": [], "gaps": [], "confidence": "unknown", "raw": text[:500]
        })
        parsed["_retrieved_mitre_hits"] = mitre_hits
        return parsed
