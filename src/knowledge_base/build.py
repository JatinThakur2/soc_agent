"""
Phase 4: Build MITRE ATT&CK RAG knowledge base.

Downloads STIX data, parses techniques, embeds with sentence-transformers,
stores in a FAISS index for similarity search.
"""
import sys
import json
import numpy as np
from pathlib import Path

import requests

from src.utils.logger import get_logger
from src.utils.paths import load_config, get_path

log = get_logger(__name__)


def download_mitre_attck(cfg) -> dict:
    kb = get_path(cfg, "knowledge_base")
    out = kb / "enterprise-attack.json"

    if out.exists():
        log.info(f"MITRE ATT&CK already downloaded at {out}")
        return json.loads(out.read_text())

    url = cfg["rag"]["mitre_attck_url"]
    log.info(f"Downloading MITRE ATT&CK from {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    out.write_text(json.dumps(data))
    log.info(f"Saved to {out} ({len(data.get('objects', []))} objects)")
    return data


def parse_techniques(attck_data: dict) -> list:
    techniques = []
    for obj in attck_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        refs = obj.get("external_references", [])
        tech_id = ""
        for r in refs:
            if r.get("source_name") == "mitre-attack":
                tech_id = r.get("external_id", "")
                break
        if not tech_id:
            continue
        techniques.append({
            "id": tech_id,
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
            "tactics": [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])],
        })
    log.info(f"Parsed {len(techniques)} active ATT&CK techniques")
    return techniques


def build_vector_store(techniques: list, cfg):
    import faiss
    from sentence_transformers import SentenceTransformer

    kb = get_path(cfg, "knowledge_base")
    kb.mkdir(parents=True, exist_ok=True)
    index_path = kb / "faiss_index.bin"
    techniques_path = kb / "techniques.json"

    # Checkpoint: skip if already built
    if index_path.exists() and techniques_path.exists():
        log.info("Checkpoint found — FAISS index already exists, skipping rebuild.")
        return

    log.info(f"Loading embedding model: {cfg['rag']['embedding_model']}")
    embedder = SentenceTransformer(cfg["rag"]["embedding_model"])

    docs = [f"{t['id']}: {t['name']}\n{t['description']}" for t in techniques]

    log.info(f"Embedding {len(techniques)} techniques...")
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch_embs = embedder.encode(docs[i:i+batch_size], convert_to_numpy=True)
        all_embeddings.append(batch_embs)
        log.info(f"  Embedded {min(i+batch_size, len(docs))}/{len(docs)}")

    embeddings = np.vstack(all_embeddings).astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(techniques_path, "w") as f:
        json.dump(techniques, f)

    log.info(f"FAISS index saved: {index.ntotal} vectors, dim={dim}")


def main():
    cfg = load_config()
    attck = download_mitre_attck(cfg)
    techniques = parse_techniques(attck)
    build_vector_store(techniques, cfg)
    log.info("Phase 4 (knowledge base) complete.")


if __name__ == "__main__":
    main()
