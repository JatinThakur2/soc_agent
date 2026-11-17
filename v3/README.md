# SOC-Agent v3: multi-LLM triage experiments

The v3 notebook was run on the JIIT HPC cluster (PBS Pro, NVIDIA A100 80 GB GPUs) with 10 open LLMs
from 3B to 72B parameters. Every model triaged the **same 508 alerts** under the **same conditions**, so
the models can be compared directly.

## Contents

| Path | What |
|---|---|
| `SOC_Agent_Local_v3.ipynb` | The notebook (detector → alerts → SHAP → LLM triage → evaluation) |
| `requirements.txt` | Python packages (`transformers<5` and `sentence-transformers<5` pinned; 5.x breaks the imports) |
| `cluster/` | Scripts used on the cluster: environment setup, PBS job, submission chains |
| `worker_code/` | `soc_agent_lib.py` / `agent_worker.py` as written by the notebook for the GPU worker |
| `notebooks_executed/` | One fully executed notebook per model, with every cell's output |
| `results/tables/` | All CSV tables. **`v3/models_same_alerts.csv`** is the main cross-model table |
| `results/figures/` | All figures (PNG + PDF); `figures/v3/` has per-model condition plots |
| `results/expert_eval/` | Blinded rating sheet for human SOC experts |
| `results/logs/` | Alerts, cohort, the exact prompts, and `run_manifest.json` (run settings) |
| `results/agent_runs.tar.gz` | Every raw LLM answer for every alert, model and condition (JSON) |

## Running on a PBS cluster

```bash
bash cluster/setup_env_cluster.sh          # builds ~/soc-agent-env from a Python 3.11 env, torch cu121
echo auto > gpu_choice                     # or a GPU index 0-7 to force one GPU
bash cluster/submit_all.sh                 # chains one PBS job per model (afterany dependencies)
bash cluster/submit_big.sh <last_job_id>   # Mistral-24B, Qwen2.5-72B, Llama-3.3-70B
```

Notes:
- PBS on this cluster does not isolate GPUs, so `soc_job.pbs` picks the GPU itself (least busy, or the
  index in `gpu_choice`). The notebook maps its worker onto the visible GPU.
- LLMs are loaded in 4-bit NF4 (bitsandbytes) on one GPU. A 72B model fits in about 45 GB.
- API keys (`HF_TOKEN`, Kaggle) are read from a private file sourced by `run_notebook.sh`
  (`~/.soc_secrets`, chmod 600). They are never stored in the repository.
- Each step is cached under `~/SOC_Agent/`, so a re-run resumes where it stopped.
- Run times on one A100: 1–3 h for 3–8B models, 3–9 h for 14–32B, 20–23 h for 70B.

## Results

### Detector (CIC-IDS2017, random split)

| Model | Accuracy | Weighted F1 | Macro F1 [95% CI] |
|---|---|---|---|
| XGBoost | 98.16 | 98.88 | 71.17 [65.56, 74.35] |
| Random Forest | 98.27 | 98.95 | 72.68 [67.30, 75.85] |
| DNN | 91.46 | 93.27 | 57.70 [54.24, 61.18] |

### LLM triage: F1 (%) per model and condition, prompt P1, same 508 alerts

| Model | Params (B) | raw | pred | pred_shap | pred_shap_explicit | pred_shuffled | flipped |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 3.1 | 69.0 | 60.3 | 58.1 | 58.1 | 58.1 | 18.8 |
| Phi-4-mini-instruct | 3.8 | 51.9 | 60.4 | 59.2 | 60.6 | 59.1 | 19.5 |
| Qwen2.5-7B-Instruct | 7.6 | 34.1 | 66.0 | 66.0 | 66.4 | 65.9 | 15.8 |
| Llama-3.1-8B-Instruct | 8.0 | 75.0 | 62.8 | 66.5 | 66.3 | 61.6 | 11.4 |
| Qwen2.5-14B-Instruct | 14.7 | 65.3 | 71.2 | 64.8 | 65.7 | 67.4 | 13.2 |
| phi-4 | 14.7 | 52.7 | 64.1 | 65.3 | 64.3 | 66.2 | 40.9 |
| Mistral-Small-24B-Instruct-2501 | 23.6 | 50.9 | 65.8 | 64.7 | 62.8 | 64.0 | 28.6 |
| Qwen2.5-32B-Instruct | 32.5 | 64.6 | 65.7 | 64.1 | 64.2 | 63.6 | 14.1 |
| Llama-3.3-70B-Instruct | 70.0 | 0.0* | 59.1 | 59.5 | 59.4 | 48.5 | 6.4 |
| Qwen2.5-72B-Instruct | 72.0 | 32.4 | 64.3 | 67.0 | 68.5 | 64.6 | 11.2 |

Conditions: `raw`, flow features only; `pred`, plus the detector's verdict; `pred_shap`, plus SHAP
explanations; `pred_shap_explicit`, plus SHAP with an explicit instruction to use it; `pred_shuffled`,
SHAP values shuffled across alerts (placebo); `flipped`, the detector's verdict deliberately inverted.

\* Llama-3.3-70B escalated **0 of 508** alerts from raw features in every run (2 prompts × 3 seeds,
all answers valid JSON). This is real model behaviour, not a parsing error: it rates every single flow
"Informational" or "Low".

### Mean over the 10 models (P1)

| Condition | Recall | FPR | F1 |
|---|---|---|---|
| raw | 45.8 | 18.6 | 49.6 |
| pred | 74.3 | 40.3 | 64.0 |
| pred_shap | 80.2 | 50.7 | 63.5 |
| pred_shap_explicit | 76.8 | 45.6 | 63.6 |
| pred_shuffled | 67.7 | 35.4 | 61.9 |
| flipped | 13.6 | 18.2 | 18.0 |

### Effect of adding SHAP (pred_shap − pred), paired bootstrap 95% CI, McNemar with Holm correction

| Model | ΔF1 [95% CI] | ΔFPR [95% CI] | p (Holm) |
|---|---|---|---|
| Qwen2.5-3B-Instruct | -2.17 [-2.2, -2.2] | +33.33 [33.3, 33.3] | 0.000 |
| Phi-4-mini-instruct | -1.21 [-2.2, -0.2] | +0.00 [0.0, 0.0] | 0.312 |
| Qwen2.5-7B-Instruct | +0.02 [-1.7, 1.9] | +3.33 [1.2, 5.7] | 0.719 |
| Llama-3.1-8B-Instruct | +3.75 [0.0, 7.6] | +18.67 [14.5, 22.7] | 0.501 |
| phi-4 | +1.15 [-0.8, 2.8] | +10.67 [7.7, 14.2] | 0.501 |
| Qwen2.5-14B-Instruct | -6.41 [-9.6, -3.6] | +17.00 [12.7, 21.3] | 0.000 |
| Mistral-Small-24B-Instruct-2501 | -1.15 [-4.1, 1.2] | -1.67 [-4.7, 2.0] | 0.901 |
| Qwen2.5-32B-Instruct | -1.61 [-4.1, 1.3] | +9.67 [5.7, 13.5] | 0.033 |
| Llama-3.3-70B-Instruct | +0.38 [-5.0, 5.2] | +25.33 [21.3, 30.0] | 0.000 |
| Qwen2.5-72B-Instruct | +2.69 [-0.3, 5.3] | -12.00 [-16.0, -8.7] | 0.002 |

### Main findings

1. **The detector's verdict drives the LLM.** Adding it raises mean F1 from 49.6 to 64.0, but roughly
   doubles the false-positive rate (18.6% → 40.3%).
2. **SHAP explanations do not improve F1** (63.5 vs 64.0 on average) and usually raise false positives.
   Qwen2.5-72B is the only model where SHAP significantly lowers false positives (−12 points).
3. **LLMs mostly follow the detector.** With an inverted verdict, F1 collapses to 18%. phi-4 (40.9) and
   Mistral-Small (28.6) resist a wrong verdict best.
4. **Size is not decisive.** Llama-3.1-8B alone (raw F1 75.0) beats both 70B-class models, and
   Qwen2.5-32B has the lowest raw false-positive rate (3.3%).
