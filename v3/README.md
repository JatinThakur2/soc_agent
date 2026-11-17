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

## Experiment design

### Pipeline
1. **Data.** CIC-IDS2017: 2,830,743 flows → 2,497,980 after removing NaN/Inf rows and exact duplicates.
   Leaky columns (IPs, ports, flow ID, timestamp) and always-null columns are dropped.
2. **Features.** Mutual-information top 30 + XGBoost top 25, plus engineered service, port-category and
   duration features. The correlation threshold is 0.95.
3. **Detector.** XGBoost, Random Forest and a DNN, trained on an 80/20 stratified random split. Benign rows
   are capped at 500k and SMOTE (k = 5) is used. XGBoost is the detector the LLMs see.
4. **Explanations.** TreeSHAP on XGBoost. Each alert carries its top-5 SHAP attributions.
5. **LLM triage.** Each LLM reads one alert and returns JSON:
   `{"severity": "Critical|High|Medium|Low|Informational", "escalate": true|false, "reasoning": "2-3 sentences"}`.
   `escalate = true` counts as "attack" for scoring.

### Alert cohort (508 alerts, identical for every model)

| Stratum | Alerts | Meaning |
|---|---|---|
| S1 attack flagged | 176 | Real attacks the detector caught (up to 15 per attack class) |
| S2 detector false positive | 200 | Benign flows the detector wrongly flagged |
| S3 hard benign | 100 | Benign flows with attack probability ≥ 0.05 |
| S4 detector miss | 32 | Every attack the detector missed in the test set |

Ground truth: 300 benign and 208 attacks across 14 classes: DoS Hulk 26, DDoS 22, Bot 22,
Web Brute Force 17, PortScan 16, slowloris 16, XSS 16, FTP-Patator 15, GoldenEye 15, Slowhttptest 15,
SSH-Patator 15, Infiltration 7, SQL Injection 4, Heartbleed 2.

The cohort is deliberately hard: 60% of it is benign traffic that looks suspicious. The scores therefore
measure whether an LLM can *correct* the detector, not raw detection accuracy.

### Conditions, prompts and seeds
- **6 conditions:** `raw`, `pred`, `pred_shap`, `pred_shap_explicit`, `pred_shuffled`, `flipped`
  (defined under the results table below).
- **2 prompt wordings:** P1 ("Tier-1 SOC Triage Analyst … the detector can be wrong") and P2 (a
  reworded version that treats the verdict "as a hint that may be mistaken"). P2 is run for `raw`,
  `pred` and `pred_shap`. Each model therefore has 9 condition×prompt settings. The exact prompt
  text is in `results/logs/prompts_v3.json` and `serialized_prompts_v3.md`.
- **3 seeds** (42, 43, 44) per setting. Temperature is 0.01 and top-p 0.95.
- **Statistics:** paired bootstrap 95% CIs, and McNemar tests with Holm correction across models.

### Metrics
- **Recall:** share of the 208 attacks escalated.
- **FPR:** share of the 300 benign alerts escalated.
- **F1:** on the escalate decision.
- **Same-as-detector %:** how often the LLM agrees with the detector.
- **Override rate / precision:** how often it disagrees with the detector, and how often it is right when it does.
- **Detector FPs closed %:** S2 alerts the LLM correctly closed.
- **Misses caught %:** S4 alerts the LLM correctly escalated.

## Running on a PBS cluster

The scripts expect to sit in the same folder as the notebook, so first copy them there:

```bash
mkdir -p ~/soc_agent_v3 && cp SOC_Agent_Local_v3.ipynb requirements.txt cluster/* ~/soc_agent_v3/
cd ~/soc_agent_v3 && mkdir -p logs
bash setup_env_cluster.sh            # builds ~/soc-agent-env from a Python 3.11 env, torch cu121
echo auto > gpu_choice               # or a GPU index 0-7 to force one GPU
bash submit_all.sh                   # chains one PBS job per model (afterany dependencies)
bash submit_big.sh <last_job_id>     # Mistral-24B, Qwen2.5-72B, Llama-3.3-70B
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

### Detector robustness

| Test | Result |
|---|---|
| Temporal split (train on earlier days, test on later ones) | Macro F1 73.87 (random split: 71.17) |
| Cross-dataset (train CIC-IDS2017 → test CIC-IDS2018) | Accuracy 66.5% (XGB) / 69.1% (RF), macro F1 ≈ 21–23. It does not transfer. |
| Leave-one-day-out (attack classes never seen in training) | Attack recall 0.01–4.2%. Unseen attack types are almost never detected. |

Top SHAP features: `Init_Win_bytes_backward`, `service_HTTP`, `Flow Bytes/s`, `Total Fwd Packets`,
`Fwd IAT Std`.

These results are why an LLM second opinion is studied at all: the detector looks excellent on
in-distribution data (98% accuracy) but fails on shifted or new traffic.

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

### How models override the detector (condition `pred_shap`, P1)

| Model | Same as detector % | Override rate % | Override precision % | Detector FPs closed % | Misses caught % |
|---|---|---|---|---|---|
| Qwen2.5-3B | 73.6 | 26.4 | 24.6 | 0.0 | 100.0 |
| Phi-4-mini | 98.4 | 1.6 | 25.0 | 0.5 | 0.0 |
| Qwen2.5-7B | 81.7 | 18.3 | 84.9 | 39.5 | 0.0 |
| Llama-3.1-8B | 82.7 | 17.3 | 81.8 | 31.0 | 28.1 |
| Qwen2.5-14B | 76.4 | 23.6 | 78.3 | 47.0 | 3.1 |
| phi-4 | 76.6 | 23.4 | 63.0 | 26.0 | 68.8 |
| Mistral-Small-24B | 76.0 | 24.0 | 68.0 | 34.0 | 50.0 |
| Qwen2.5-32B | 85.6 | 14.4 | 82.2 | 29.5 | 3.1 |
| Llama-3.3-70B | 66.1 | 33.9 | 69.2 | 60.0 | 0.0 |
| Qwen2.5-72B | 74.2 | 25.8 | 80.9 | 52.5 | 0.0 |

Larger models close more of the detector's false alarms (Llama-3.3-70B 60%, Qwen2.5-72B 52.5%) but
catch almost none of its misses. phi-4 and Mistral-Small are the only models that do some of both.
Qwen2.5-3B "catches" all misses only because it escalates nearly everything.

### Main findings

1. **The detector's verdict drives the LLM.** Adding it raises mean F1 from 49.6 to 64.0, but roughly
   doubles the false-positive rate (18.6% → 40.3%).
2. **SHAP explanations do not improve F1** (63.5 vs 64.0 on average) and usually raise false positives.
   Qwen2.5-72B is the only model where SHAP significantly lowers false positives (−12 points).
3. **LLMs mostly follow the detector.** With an inverted verdict, F1 collapses to 18%. phi-4 (40.9) and
   Mistral-Small (28.6) resist a wrong verdict best.
4. **Size is not decisive.** Llama-3.1-8B alone (raw F1 75.0) beats both 70B-class models, and
   Qwen2.5-32B has the lowest raw false-positive rate (3.3%).
5. **The detector itself does not generalise** to unseen attack types or to CIC-IDS2018, and the LLMs
   catch almost none of its misses. An LLM layer reduces alert fatigue (closing false positives) more
   than it improves detection.

## Compute log

| Model | Hugging Face ID | Run time (1× A100 80 GB, 4-bit) |
|---|---|---|
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` | 1.1 h |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 2.6 h |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 1.2 h |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 4.6 h |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B-Instruct` | 8.6 h (also built the detector and cohort) |
| phi-4 | `microsoft/phi-4` | 2.9 h |
| Mistral-Small-24B | `mistralai/Mistral-Small-24B-Instruct-2501` | 5.9 h |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | ~8 h (over 3 attempts, resumed from cache) |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 22.6 h |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | 20 h |

Llama models are gated: accept the licence on Hugging Face and set `HF_TOKEN` before running.

Software: Python 3.11, PyTorch 2.5.1+cu121 (NVIDIA driver 525), transformers 4.57.6,
sentence-transformers 4.1.0, bitsandbytes (NF4 4-bit), papermill 2.7.0.

## Known issues and fixes

| Problem | Fix |
|---|---|
| `NameError: name 'nn' is not defined` while importing sentence-transformers | transformers 5.x is incompatible: pin `transformers<5`, `sentence-transformers<5` |
| The LLM worker landed on another user's GPU | PBS does not set `CUDA_VISIBLE_DEVICES`. `soc_job.pbs` picks a GPU, and the notebook maps the worker to it |
| `#PBS` directive error | PBS does not allow comments at the end of `#PBS` lines |
| CUDA out of memory on a shared GPU | Set `gpu_choice` to `auto` (picks the least-used GPU) and resubmit. Finished conditions are cached |
| `OSError: [Errno 70] Communication error on send` | Transient network-filesystem error. Resubmit: the run resumes from the cache |
| Llama-3.3-70B raw F1 = 0 | Real behaviour: it never escalates from raw features (see results) |

## Reproducing one model

```bash
cd ~/soc_agent_v3                    # the folder prepared above
qsub -N soc_qwen7b -v MODEL=Qwen/Qwen2.5-7B-Instruct,V3ONLY=1,BATCH=16 soc_job.pbs
# or interactively on a GPU node:
SOC_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct SOC_V3_ONLY=1 bash run_notebook.sh
```
`V3ONLY=1` skips detector training when it is already cached, so run one model with `V3ONLY=0` first.
Progress: `tail -f ~/SOC_Agent/workers/worker_0.log`.

## Datasets

- CIC-IDS2017 and CSE-CIC-IDS2018, Canadian Institute for Cybersecurity, University of New Brunswick
  (https://www.unb.ca/cic/datasets/).
- MITRE ATT&CK Enterprise v17.1 (used for retrieval in the full agent pipeline).
