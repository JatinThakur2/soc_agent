# SOC-Agent — Complete Setup Guide

Autonomous multi-agent LLM framework for SOC alert triage using CIC-IDS2017 network intrusion data.  
Supports **local machines**, **Linux servers**, and **HPC clusters with PBS + A100 GPUs**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Kaggle Credentials](#4-kaggle-credentials)
5. [LLM Backend — Choose One](#5-llm-backend--choose-one)
6. [Run the Pipeline (Local / Interactive)](#6-run-the-pipeline-local--interactive)
7. [Run on HPC Cluster (PBS + A100 GPU)](#7-run-on-hpc-cluster-pbs--a100-gpu)
8. [Configuration Reference](#8-configuration-reference)
9. [Project Structure](#9-project-structure)
10. [Output Files](#10-output-files)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.10 or 3.11 |
| RAM | 16 GB | 64 GB |
| Disk | 50 GB free | 100 GB free |
| GPU | — | NVIDIA A100 / RTX 3090+ (CUDA 12.x) |

### Accounts Required

- **Kaggle account** — free at [kaggle.com](https://www.kaggle.com). Needed to download the dataset.
- **LLM backend** — either:
  - **Ollama** (free, runs locally) — recommended for HPC/offline use
  - **Anthropic API** — paid, requires key from [console.anthropic.com](https://console.anthropic.com)

---

## 2. Clone the Repository

```bash
git clone https://github.com/<your-repo>/soc_agent.git
cd soc_agent
```

---

## 3. Python Environment Setup

### Option A — Standard Linux / Local Machine

```bash
# Check Python version (need 3.9+)
python3 --version

# Create virtual environment
python3 -m venv .venv

# If venv creation fails (ensurepip missing), bootstrap pip manually:
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3

# Install all dependencies
.venv/bin/pip install -r requirements.txt

# Install pyarrow (required for parquet files — sometimes missing from requirements)
.venv/bin/pip install pyarrow
```

### Option B — HPC Cluster (module-based Python)

On clusters where system Python is too old (e.g., 3.6), load a newer version via modules:

```bash
# Check what's available
module avail python

# Load Python 3.9 (adjust version to what your cluster has)
module load python39

# Create the venv
python3.9 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pyarrow
```

> **Broken symlink fix:** If you set up the venv on one node and move to another,
> the `python` symlink may point to the wrong binary. Fix it with:
> ```bash
> ln -sf python3.9 .venv/bin/python
> ln -sf python3.9 .venv/bin/python3
> ```

### Option C — Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyarrow
```

### Verify GPU is detected

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
.venv/bin/python -c "import xgboost as xgb; print('XGB GPU OK')"
```

If PyTorch shows `CUDA: False` but you have a GPU, reinstall with the correct CUDA wheel:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Kaggle Credentials

1. Go to [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Click **Create New API Token** — downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp /path/to/downloaded/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Or set environment variables instead of the file:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

> **Note:** The dataset used is `kk0105/cicids2017` (Apache 2.0 license, no browser acceptance required).
> This is a mirror of the original CIC-IDS2017 dataset.

---

## 5. LLM Backend — Choose One

### Option A — Ollama (Free, Local — Recommended)

Ollama runs LLMs entirely on your machine with no API cost.

**Step 1 — Install Ollama:**

```bash
# Linux (standard install)
curl -fsSL https://ollama.com/install.sh | sh

# If the above fails (TLS/proxy issues on HPC), download manually:
wget https://ollama.com/download/ollama-linux-amd64 -O ~/.local/bin/ollama
chmod +x ~/.local/bin/ollama
```

**Step 2 — Pull the model:**

```bash
ollama pull phi3.5
```

> If `ollama pull` fails due to corporate proxy/TLS, download the GGUF manually:
> ```bash
> # Download phi3.5 GGUF from HuggingFace
> python3 -c "
> from huggingface_hub import hf_hub_download
> import urllib3; urllib3.disable_warnings()
> hf_hub_download(
>     repo_id='bartowski/Phi-3.5-mini-instruct-GGUF',
>     filename='Phi-3.5-mini-instruct-Q4_K_M.gguf',
>     local_dir='~/ollama_models',
>     endpoint='https://huggingface.co'
> )
> "
> # Import it into Ollama via Modelfile
> cat > /tmp/Modelfile.phi35 << 'EOF'
> FROM ~/ollama_models/Phi-3.5-mini-instruct-Q4_K_M.gguf
> PARAMETER temperature 0.0
> EOF
> ollama create phi3.5 -f /tmp/Modelfile.phi35
> ```

**Step 3 — Start the server and configure `.env`:**

```bash
# Default port
ollama serve &

# Or on a custom port (e.g., if default 11434 is taken on HPC)
OLLAMA_HOST=0.0.0.0:11435 ollama serve &
```

**Step 4 — Set in `.env`:**

```bash
cp .env.example .env
# Edit .env:
#   OLLAMA_HOST=http://localhost:11434    # or 11435 if using custom port
```

**Step 5 — Set backend in `configs/config.yaml`:**

```yaml
agents:
  backend: ollama
  ollama_model: phi3.5
```

---

### Option B — Anthropic API (Claude)

```bash
cp .env.example .env
# Edit .env and add:
#   ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

Set in `configs/config.yaml`:

```yaml
agents:
  backend: anthropic
  llm_model: claude-haiku-4-5-20251001   # cheapest; or claude-sonnet-4-6 for better quality
```

> Cost estimate: ~$0.03–0.10 per alert with Haiku, ~$0.30–1.00 with Sonnet.
> Set `n_eval_alerts: 10` in config for a cheap smoke test.

---

## 6. Run the Pipeline (Local / Interactive)

Run each step in order. Each step depends on the previous one completing successfully.

```bash
cd /path/to/soc_agent

# Step 1 — Download dataset (~500 MB, ~3 min)
.venv/bin/python -m src.data.download

# Step 2 — Preprocess data (~5 min)
.venv/bin/python -m src.data.preprocess

# Step 3 — Feature engineering (~10-15 min, CPU-heavy)
.venv/bin/python -m src.features.engineer

# Step 4 — Train models (~15-30 min, GPU accelerated if available)
.venv/bin/python -m src.models.train

# Step 5 — Build MITRE ATT&CK knowledge base (one-time, ~2 min)
.venv/bin/python -m src.knowledge_base.build

# Step 6 — Start Ollama server (keep running in a separate terminal)
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Step 7 — Run agent triage pipeline (~20-40 min)
.venv/bin/python -m src.agents.run_pipeline

# Step 8 — Evaluate and generate plots
.venv/bin/python -m src.evaluation.evaluate
```

> Steps 1 and 2 only need to run once. If you already have
> `data/cleaned/combined_cleaned.parquet`, skip directly to Step 3.

---

## 7. Run on HPC Cluster (PBS + A100 GPU)

PBS job scripts are provided in `jobs/`. They handle CUDA environment setup,
Ollama lifecycle, and job chaining automatically.

### Cluster Overview

| Queue | Nodes | GPUs | CPUs | RAM |
|-------|-------|------|------|-----|
| `gpu` | jiit-gpu01 | 8× A100 SXM 80GB | 96 | ~2 TB |
| `workq` | jiit-cn01–06 | — | 64 each | ~256 GB each |

### One-Command Submit (All Steps, Auto-Chained)

```bash
bash jobs/submit_all.sh
```

This submits all 5 jobs with PBS dependencies — each job waits for the previous
to succeed before starting.

### Monitor Jobs

```bash
# See status of all your jobs
qstat -u $USER

# Watch live output of a running job
tail -f jobs/logs/01_engineer.out
tail -f jobs/logs/02_train.out
tail -f jobs/logs/04_pipeline.out
```

### Submit Jobs Individually

```bash
# Feature engineering (CPU, workq)
qsub jobs/01_engineer.pbs

# Model training (GPU, A100)
qsub jobs/02_train.pbs

# Build knowledge base (CPU)
qsub jobs/03_build_kb.pbs

# Agent pipeline + Ollama (GPU, A100)
qsub jobs/04_pipeline.pbs

# Evaluation (CPU)
qsub jobs/05_evaluate.pbs
```

### PBS Job Summary

| Script | Queue | Resources | Walltime | What it runs |
|--------|-------|-----------|----------|--------------|
| `01_engineer.pbs` | workq | 32 CPU, 64 GB | 1h | `src.features.engineer` |
| `02_train.pbs` | gpu | 1× A100, 16 CPU | 2h | `src.models.train` |
| `03_build_kb.pbs` | workq | 4 CPU, 16 GB | 30min | `src.knowledge_base.build` |
| `04_pipeline.pbs` | gpu | 1× A100, 8 CPU | 3h | Ollama + `src.agents.run_pipeline` |
| `05_evaluate.pbs` | workq | 4 CPU, 16 GB | 30min | `src.evaluation.evaluate` |

### Re-run a Failed Job

```bash
# Check what went wrong
cat jobs/logs/02_train.out

# Resubmit just that step
qsub jobs/02_train.pbs
```

---

## 8. Configuration Reference

All settings are in `configs/config.yaml`. Key options:

```yaml
quick_mode: false          # true = use 5% data, 5 epochs, 10 alerts (smoke test)

datasets:
  use: ["cicids2017"]      # add "cicids2018" for both datasets
  sample_fraction: 1.0     # 0.1 = use 10% of data

agents:
  backend: ollama           # "ollama" (free local) or "anthropic" (paid API)
  ollama_model: phi3.5      # any model pulled via `ollama pull`
  llm_model: claude-haiku-4-5-20251001
  n_eval_alerts: 100        # number of alerts to process (reduce to cut cost/time)

models:
  train_xgboost: true
  train_random_forest: true
  train_dnn: true
  xgboost:
    use_gpu: true           # set false if no NVIDIA GPU
  dnn:
    use_gpu: true

rag:
  embedding_model: all-MiniLM-L6-v2
  top_k_retrieval: 5
```

### Quick Mode (Fast Smoke Test)

```bash
# Edit configs/config.yaml: set quick_mode: true
# Then run normally — completes in ~10 min
.venv/bin/python -m src.features.engineer
.venv/bin/python -m src.models.train
.venv/bin/python -m src.agents.run_pipeline
.venv/bin/python -m src.evaluation.evaluate
```

---

## 9. Project Structure

```
soc_agent/
├── configs/
│   └── config.yaml              # All settings and hyperparameters
├── src/
│   ├── data/
│   │   ├── download.py          # Download CIC-IDS2017 from Kaggle
│   │   └── preprocess.py        # Clean raw CSVs → parquet
│   ├── features/
│   │   └── engineer.py          # Feature selection (MI + XGBoost) + engineering
│   ├── models/
│   │   ├── train.py             # Train XGBoost, Random Forest, DNN
│   │   └── alert_generator.py   # Convert test rows into alert packets
│   ├── knowledge_base/
│   │   └── build.py             # Download MITRE ATT&CK + build FAISS index
│   ├── agents/
│   │   ├── triage_agent.py      # Severity + urgency classification
│   │   ├── investigation_agent.py  # MITRE ATT&CK RAG mapping
│   │   ├── response_agent.py    # Remediation recommendations
│   │   ├── coordinator_agent.py # Final synthesis + priority
│   │   ├── run_pipeline.py      # Pipeline orchestrator
│   │   ├── llm_factory.py       # LLM backend abstraction (Ollama / Anthropic)
│   │   └── json_utils.py        # Robust JSON parsing for LLM output
│   ├── evaluation/
│   │   └── evaluate.py          # Metrics, confusion matrices, SHAP plots
│   └── utils/
│       ├── logger.py
│       └── paths.py
├── jobs/                        # PBS job scripts (HPC only)
│   ├── 01_engineer.pbs
│   ├── 02_train.pbs
│   ├── 03_build_kb.pbs
│   ├── 04_pipeline.pbs
│   ├── 05_evaluate.pbs
│   ├── submit_all.sh            # Submit all jobs with dependencies
│   └── logs/                    # PBS stdout/stderr logs
├── data/                        # Created at runtime
│   ├── raw/cic-ids-2017/        # Downloaded CSVs
│   ├── cleaned/                 # combined_cleaned.parquet
│   ├── processed/               # scaler.pkl, label_encoder.pkl, feature_list.json
│   └── splits/                  # X_train/X_test parquet, y_train/y_test npy
├── models/                      # Saved model files (.pkl, .pt)
├── knowledge_base/              # FAISS index + techniques.json
├── results/
│   ├── figures/                 # ROC curves, confusion matrices, SHAP plots
│   ├── tables/                  # CSVs: classification report, feature rankings
│   └── logs/                    # agent_pipeline_results.json
├── scripts/
│   ├── setup_and_run.sh         # One-command run (Linux/Mac)
│   ├── setup_and_run.ps1        # One-command run (Windows)
│   └── verify_setup.py          # Pre-flight dependency check
├── requirements.txt
├── .env.example
├── .env                         # Your API keys (never commit this)
└── README.md
```

---

## 10. Output Files

| Path | Contents |
|------|----------|
| `models/` | Trained XGBoost (`.pkl`), Random Forest (`.pkl`), DNN (`.pt`) |
| `results/figures/` | ROC curves, confusion matrices, SHAP feature importance plots |
| `results/tables/` | Classification reports, precision/recall/F1, feature rankings |
| `results/logs/agent_pipeline_results.json` | Full reasoning chains from all 4 agents per alert |
| `knowledge_base/faiss_index.bin` | MITRE ATT&CK FAISS vector index |
| `knowledge_base/techniques.json` | Parsed ATT&CK technique metadata |

---

## 11. Troubleshooting

### `.venv/bin/python: No such file or directory`

The venv's `python` symlink is broken (created on a different node/machine).

```bash
ln -sf python3.9 .venv/bin/python
ln -sf python3.9 .venv/bin/python3
```

### `ImportError: No module named 'pyarrow'`

```bash
.venv/bin/pip install pyarrow
```

### `403 Forbidden` on Kaggle download

The original `cicdataset/cicids2017` dataset requires accepting license terms in a browser.
The code uses `kk0105/cicids2017` (Apache 2.0, no browser step needed).
If you still get 403, verify your `kaggle.json` is valid:

```bash
cat ~/.kaggle/kaggle.json   # should show {"username":"...","key":"..."}
.venv/bin/python -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); print('OK')"
```

### `ollama pull` fails (TLS / proxy error)

On corporate networks or HPC clusters, the TLS certificate is often intercepted by a proxy.
Download the model directly from HuggingFace:

```bash
.venv/bin/pip install huggingface_hub
.venv/bin/python -c "
import ssl, urllib3
urllib3.disable_warnings()
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Phi-3.5-mini-instruct-GGUF',
    filename='Phi-3.5-mini-instruct-Q4_K_M.gguf',
    local_dir='ollama_models'
)
"
cat > /tmp/Modelfile <<EOF
FROM ./ollama_models/Phi-3.5-mini-instruct-Q4_K_M.gguf
EOF
ollama create phi3.5 -f /tmp/Modelfile
```

### `ANTHROPIC_API_KEY not found`

Create or edit the `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
```

Or switch to Ollama (free):

```yaml
# configs/config.yaml
agents:
  backend: ollama
```

### `CUDA out of memory` during training

Reduce batch size or use quick mode:

```yaml
# configs/config.yaml
quick_mode: true   # uses 5% of data
```

Or disable GPU for the model that fails:

```yaml
models:
  xgboost:
    use_gpu: false
  dnn:
    use_gpu: false
```

### PBS job stuck in queue (`qstat` shows `Q` state)

```bash
qstat -f <job_id> | grep comment   # see why it's waiting
```

Common causes: requested more GPUs/CPUs than available, walltime too long for queue policy.

### PBS job fails immediately

```bash
cat jobs/logs/02_train.out   # check the error output
```

Common causes: wrong Python path, missing data files from a failed earlier step.

### `FileNotFoundError: FAISS index not found`

The knowledge base has not been built yet. Run:

```bash
.venv/bin/python -m src.knowledge_base.build
# or on HPC:
qsub jobs/03_build_kb.pbs
```

---

## Cost Estimates

| Resource | Cost |
|----------|------|
| Dataset download | Free (Kaggle) |
| Ollama LLM (local) | Free (GPU compute only) |
| Anthropic Claude Haiku (100 alerts) | ~$3–5 |
| Anthropic Claude Sonnet (100 alerts) | ~$30–50 |
| HPC GPU hours (A100, ~6h total) | Depends on your allocation |

To minimize API cost, set `n_eval_alerts: 10` in `configs/config.yaml` for initial testing.
