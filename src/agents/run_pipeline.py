"""
Full multi-agent pipeline runner.

Samples alerts from the test set, generates alert packets, and runs them
through Triage → Investigation → Response → Coordinator.
Saves complete reasoning chains to results/logs/.
"""
import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.paths import load_config, get_path
from src.models.alert_generator import AlertGenerator
from src.agents.triage_agent import build_triage_chain, run_triage
from src.agents.investigation_agent import InvestigationAgent
from src.agents.response_agent import build_response_chain, run_response
from src.agents.coordinator_agent import build_coordinator_chain, run_coordinator

log = get_logger(__name__)

# Checkpoint path alongside output file
_CKPT_SUFFIX = ".checkpoint.json"


def _ckpt_path(out: Path) -> Path:
    return out.with_suffix(_CKPT_SUFFIX)


def _load_checkpoint(out: Path) -> list:
    p = _ckpt_path(out)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        log.info(f"Checkpoint found — resuming from {len(data)} completed alerts.")
        return data
    return []


def _save_checkpoint(results: list, out: Path):
    with open(_ckpt_path(out), "w") as f:
        json.dump(results, f, default=str)


def check_api_key(cfg):
    backend = cfg["agents"].get("backend", "anthropic")
    if backend == "ollama":
        log.info("Backend: Ollama (local). Make sure 'ollama serve' is running.")
        return
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("=" * 70)
        log.error("ANTHROPIC_API_KEY not found!")
        log.error("")
        log.error("For Anthropic backend:")
        log.error("  Create a .env file in the project root:")
        log.error("  ANTHROPIC_API_KEY=sk-ant-xxxxx")
        log.error("")
        log.error("To use Ollama (free, local) instead:")
        log.error("  1. Install Ollama: https://ollama.com")
        log.error("  2. Run: ollama pull phi3.5")
        log.error("  3. Run: ollama serve")
        log.error("  4. Set agents.backend: ollama in configs/config.yaml")
        log.error("=" * 70)
        sys.exit(1)


def process_alert(alert_id, X_row, true_label, label_names, generator,
                  triage_chain, investigation_agent, response_chain, coordinator_chain):
    """Run a single alert through the full agent pipeline."""
    result = {
        "alert_id": alert_id,
        "ground_truth": str(label_names[true_label]),
    }

    try:
        alert = generator.generate(X_row, alert_id)
        result["alert"] = alert

        # Skip Benign predictions — no agent cost needed
        if alert["predicted_label"].upper() in ("BENIGN", "NORMAL"):
            result["skipped"] = True
            result["skip_reason"] = "Predicted Benign"
            return result

        triage = run_triage(triage_chain, alert)
        result["triage"] = triage

        investigation = investigation_agent.run(alert, triage)
        result["investigation"] = investigation

        response = run_response(response_chain, alert, triage, investigation)
        result["response"] = response

        coordinator = run_coordinator(coordinator_chain, alert, triage, investigation, response)
        result["coordinator"] = coordinator

    except Exception as e:
        log.error(f"Alert {alert_id} failed: {e}")
        result["error"] = str(e)

    return result


def main():
    cfg = load_config()
    check_api_key(cfg)

    # Load test data and label names
    splits = get_path(cfg, "data_splits")
    proc = get_path(cfg, "data_processed")
    X_test = pd.read_parquet(splits / "X_test.parquet")
    y_test = np.load(splits / "y_test.npy")
    le = joblib.load(proc / "label_encoder.pkl")

    log.info(f"Test set: {X_test.shape}, {len(le.classes_)} classes")
    log.info(f"LLM backend: {cfg['agents'].get('backend', 'anthropic')}")

    # Initialize agent chains (all share the same backend)
    generator = AlertGenerator(cfg)
    triage_chain = build_triage_chain(cfg)
    investigation_agent = InvestigationAgent(cfg)
    response_chain = build_response_chain(cfg)
    coordinator_chain = build_coordinator_chain(cfg)

    # Sample alerts (stratified — prefer attack samples over Benign)
    n_eval = min(cfg["agents"]["n_eval_alerts"], len(X_test))
    rng = np.random.default_rng(cfg["random_seed"])

    benign_labels = {i for i, name in enumerate(le.classes_)
                     if name.upper() in ("BENIGN", "NORMAL")}
    non_benign_idx = [i for i, y in enumerate(y_test) if y not in benign_labels]
    benign_idx = [i for i, y in enumerate(y_test) if y in benign_labels]

    n_attack = int(n_eval * 0.8)
    n_benign = n_eval - n_attack
    sample_idx = []
    if non_benign_idx:
        sample_idx.extend(rng.choice(non_benign_idx, min(n_attack, len(non_benign_idx)), replace=False).tolist())
    if benign_idx:
        sample_idx.extend(rng.choice(benign_idx, min(n_benign, len(benign_idx)), replace=False).tolist())

    log.info(f"Running agent pipeline on {len(sample_idx)} alerts "
             f"({n_attack} attacks, {n_benign} benign)...")

    # Checkpoint / resume
    logs = get_path(cfg, "results_logs")
    out = logs / "agent_pipeline_results.json"
    results = _load_checkpoint(out)
    completed_ids = {r["alert_id"] for r in results}

    for i, idx in enumerate(tqdm(sample_idx, desc="Processing alerts")):
        alert_id = f"alert-{i:04d}-row{idx}"
        if alert_id in completed_ids:
            continue

        result = process_alert(
            alert_id,
            X_test.iloc[idx],
            int(y_test[idx]),
            le.classes_,
            generator,
            triage_chain,
            investigation_agent,
            response_chain,
            coordinator_chain,
        )
        results.append(result)
        _save_checkpoint(results, out)  # survive power cuts

    # Write final output
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved {len(results)} agent runs to {out}")

    completed = [r for r in results if "coordinator" in r]
    skipped = [r for r in results if r.get("skipped")]
    errored = [r for r in results if r.get("error")]
    log.info(f"Summary: {len(completed)} completed, {len(skipped)} skipped (benign), {len(errored)} errors")

    if completed:
        severities = [r["coordinator"].get("final_severity", "Unknown") for r in completed]
        sev_counts = pd.Series(severities).value_counts().to_dict()
        log.info(f"Severity distribution: {sev_counts}")


if __name__ == "__main__":
    main()
