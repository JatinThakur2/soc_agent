# SOC-Agent: Autonomous Multi-Agent LLM Framework for SOC Alert Triage

End-to-end implementation for intrusion detection and LLM-powered alert triage using CIC-IDS2017 and CIC-IDS2018 datasets.

---

## Prerequisites (One-time manual setup)

Before running anything, complete these two steps:

### 1. Kaggle API credentials (for dataset download)

```powershell
# Step 1: Go to https://www.kaggle.com/settings/account
# Step 2: Click "Create New API Token" — it downloads kaggle.json
# Step 3: Place it here:
mkdir "$env:USERPROFILE\.kaggle" -Force
# Copy kaggle.json to C:\Users\<YOU>\.kaggle\kaggle.json
```

### 2. Anthropic API key (for LLM agents)

```powershell
# In the project root, create a .env file:
Copy-Item .env.example .env
# Then edit .env and set your key:
#   ANTHROPIC_API_KEY=sk-ant-xxxxx
notepad .env
```

---

## Quick Start (Recommended — Windows PowerShell)

Open PowerShell **in the project root** and run:

```powershell
# Allow script execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run the full pipeline (setup + download + train + agents + eval)
.\scripts\setup_and_run.ps1
```

That's it. The script will:

1. Create a Python virtual environment (`.venv`)
2. Install all dependencies from `requirements.txt`
3. Download CIC-IDS2017 dataset (via Kaggle API)
4. Preprocess and feature-engineer the data
5. Train XGBoost, Random Forest, and DNN classifiers
6. Build the MITRE ATT&CK knowledge base (RAG)
7. Run the multi-agent triage pipeline
8. Evaluate and save results

**First-time setup time:** ~30-45 minutes (mostly dataset downloads)
**Subsequent runs:** ~15-30 minutes (depending on dataset size)

### Quick Start (Linux/macOS)

```bash
bash scripts/setup_and_run.sh
```

---

## Step-by-Step Manual Setup

Use this if you want full control over each step.

### Step 1 — Create and activate virtual environment

```powershell
# Create venv
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (CMD)
.\.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 2 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU support (optional):** For CUDA-accelerated PyTorch, replace the torch line:
>
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 3 — Verify setup (optional but recommended)

```powershell
python -m scripts.verify_setup
```

This checks Python version, all dependencies, Kaggle credentials, Anthropic API key, GPU availability, and disk space.

### Step 4 — Download datasets

```powershell
python -m src.data.download
```

Downloads CIC-IDS2017 from Kaggle to `data/raw/cic-ids-2017/`.

### Step 5 — Preprocess data (Phase 1)

```powershell
python -m src.data.preprocess
```

Cleans raw CSVs, drops null/leaky features, removes infinities, outputs to `data/cleaned/`.

### Step 6 — Feature engineering (Phase 2)

```powershell
python -m src.features.engineer
```

Mutual Information + XGBoost feature selection, engineers port/protocol/ratio features, outputs to `data/processed/`.

### Step 7 — Train classifiers (Phase 3)

```powershell
python -m src.models.train
```

Trains XGBoost, Random Forest, and DNN with SMOTE resampling. Saves models to `models/`.

### Step 8 — Build MITRE ATT&CK knowledge base (Phase 4a)

```powershell
python -m src.knowledge_base.build
```

Downloads MITRE ATT&CK enterprise JSON, chunks it, creates ChromaDB vector store in `knowledge_base/`.

### Step 9 — Run agent pipeline (Phase 4b)

```powershell
python -m src.agents.run_pipeline
```

Runs Triage → Investigation → Response → Coordinator agents on 100 alerts (configurable). Requires `ANTHROPIC_API_KEY` in `.env`.

### Step 10 — Evaluate results (Phase 5)

```powershell
python -m src.evaluation.evaluate
```

Generates confusion matrices, ROC curves, SHAP plots, and summary tables in `results/`.

---

## Partial / Selective Runs

### Skip expensive stages using PowerShell flags

```powershell
# Skip data download (if already downloaded)
.\scripts\setup_and_run.ps1 -SkipData

# Skip LLM agents (no Anthropic API cost)
.\scripts\setup_and_run.ps1 -SkipAgents

# Skip MITRE KB build (if already built)
.\scripts\setup_and_run.ps1 -SkipKB

# Skip venv setup (if already installed)
.\scripts\setup_and_run.ps1 -SkipSetup

# Run only one stage
.\scripts\setup_and_run.ps1 -Only train
.\scripts\setup_and_run.ps1 -Only eval
.\scripts\setup_and_run.ps1 -Only agents

# Combine flags
.\scripts\setup_and_run.ps1 -SkipSetup -SkipData -SkipKB
```

### Available `-Only` stage names

| Flag         | Stage                        |
| ------------ | ---------------------------- |
| `data`       | Download datasets            |
| `preprocess` | Phase 1: data cleaning       |
| `features`   | Phase 2: feature engineering |
| `train`      | Phase 3: model training      |
| `kb`         | Phase 4a: MITRE ATT&CK RAG   |
| `agents`     | Phase 4b: LLM agent pipeline |
| `eval`       | Phase 5: evaluation & plots  |

---

## Quick Mode (Fast Test Run)

To run a fast smoke test using 5% of the data and fewer epochs:

```yaml
# Edit configs/config.yaml — set:
quick_mode: true
```

Then run normally. Quick mode overrides:

- `sample_fraction: 0.05` (5% of data)
- `dnn_epochs: 5`
- `n_eval_alerts: 10` (only 10 agent runs, ~$0.30 cost)
- `benign_cap: 50000`

---

## Configuration

All hyperparameters, paths, and pipeline settings live in [configs/config.yaml](configs/config.yaml).

Key settings:

```yaml
quick_mode: false            # true = fast test run

datasets:
  use: ["cicids2017"]        # add "cicids2018" for both datasets
  sample_fraction: 1.0       # 0.1 = use 10% sample

agents:
  llm_model: "claude-sonnet-4-5-20250929"
  n_eval_alerts: 100         # reduce to cut API cost (~$0.03/alert)

models:
  train_xgboost: true
  train_random_forest: true
  train_dnn: true
  xgboost:
    use_gpu: true             # false if no NVIDIA GPU
  dnn:
    use_gpu: true
```

---

## Project Structure

```
soc_agent_project/
├── scripts/
│   ├── setup_and_run.sh        # One-command setup + run (Linux/Mac)
│   ├── setup_and_run.ps1       # One-command setup + run (Windows)
│   └── verify_setup.py         # Pre-flight check script
├── configs/
│   └── config.yaml             # All hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── download.py         # Dataset download (Kaggle)
│   │   └── preprocess.py       # Phase 1: cleaning
│   ├── features/
│   │   └── engineer.py         # Phase 2: feature selection + engineering
│   ├── models/
│   │   ├── train.py            # Phase 3: train all classifiers
│   │   └── alert_generator.py  # Alert packet generation
│   ├── knowledge_base/
│   │   └── build.py            # Phase 4a: MITRE ATT&CK ChromaDB RAG
│   ├── agents/
│   │   ├── triage_agent.py
│   │   ├── investigation_agent.py
│   │   ├── response_agent.py
│   │   ├── coordinator_agent.py
│   │   └── run_pipeline.py     # Full agent pipeline orchestrator
│   ├── evaluation/
│   │   └── evaluate.py         # Phase 5: metrics + plots
│   └── utils/
│       ├── logger.py
│       └── paths.py
├── data/                       # Created automatically
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── models/                     # Saved model files
├── knowledge_base/             # ChromaDB vector store
├── results/                    # Output plots and tables
│   ├── figures/
│   ├── tables/
│   └── logs/
├── requirements.txt
├── .env.example
└── README.md
```

---

## System Requirements

**Minimum:**

- Python 3.10+
- 16 GB RAM
- 50 GB free disk space

**Recommended:**

- 32 GB RAM
- NVIDIA GPU with 8+ GB VRAM (for DNN training)
- 100 GB free disk space
- CUDA 12.x + cuDNN

---

## Output Files

After a successful run, results are saved to:

| Path               | Contents                                         |
| ------------------ | ------------------------------------------------ |
| `models/`          | Trained XGBoost, RF, DNN model files             |
| `results/figures/` | ROC curves, confusion matrices, SHAP plots       |
| `results/tables/`  | Classification reports, F1/precision/recall CSVs |
| `results/logs/`    | Agent triage logs, pipeline run log              |
| `knowledge_base/`  | ChromaDB MITRE ATT&CK vector store               |

---

## Cost Estimate

| Resource                              | Cost                   |
| ------------------------------------- | ---------------------- |
| Dataset download                      | Free (Kaggle)          |
| Anthropic API (100 alerts)            | ~$3–10 (Claude Sonnet) |
| Anthropic API (10 alerts, quick mode) | ~$0.30                 |

Reduce `n_eval_alerts` in `configs/config.yaml` to control API spend.

---

## Troubleshooting

**"cannot be loaded because running scripts is disabled"**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"kaggle: command not found" or ImportError**

```powershell
pip install kaggle
```

**"OSError: Could not find kaggle.json"**
→ Place `kaggle.json` at `C:\Users\<YOU>\.kaggle\kaggle.json` (see Prerequisites step 1).

**CUDA out of memory**
→ Set `quick_mode: true` in `configs/config.yaml`, or set `use_gpu: false` under `xgboost` and `dnn`.

**"ANTHROPIC_API_KEY not set" / agent pipeline fails**
→ Edit `.env` and add `ANTHROPIC_API_KEY=sk-ant-xxxxx`. Then re-run with `-SkipSetup -SkipData -SkipKB`.

**API rate limit on Anthropic**
→ Reduce `n_eval_alerts` in config (default: 100). Set to 10–20 for testing.

**"403 Forbidden" on Kaggle download**
→ Kaggle requires accepting dataset terms before API access works:

1. Open [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/cicdataset/cicids2017) in your browser
2. Sign in and click **I Agree** / **Download** to accept the rules
3. Re-run: `python -m src.data.download`

**Download fails / slow internet**
→ Download CSVs manually from [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and place all CSV files into `data/raw/cic-ids-2017/`.
