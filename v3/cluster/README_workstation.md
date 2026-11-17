# SOC-Agent v3 — local workstation package (review experiments)

Files
- `SOC_Agent_Local_v3.ipynb` — the notebook (all instructions are in its first cell)
- `setup_env.sh` — one-time setup: checks the NVIDIA driver, creates `~/soc-agent-env`, installs CUDA PyTorch and the packages, registers the Jupyter kernel "SOC-Agent (local)"
- `run_notebook.sh` — runs the whole notebook headless for one model (use inside `tmux`); safe to re-run, it resumes
- `run_all_models.sh` — runs the notebook once per LLM (same alerts, same conditions); the review experiments need this
- `requirements.txt` — Python packages (PyTorch is installed by `setup_env.sh`)

Quick start
```bash
unzip soc_agent_v3.zip -d ~ && cd ~/soc_agent_v3
bash setup_env.sh          # only if ~/soc-agent-env does not exist yet (you already did this for v2)
tmux new -s soc
bash run_all_models.sh
```
Your existing `~/SOC_Agent` folder is reused: the detector, the cohort and the finished v2 runs are kept; the cohort is extended and only new work is run.
Results: `~/SOC_Agent/results/`. Worker progress: `tail -f ~/SOC_Agent/workers/worker_0.log`.
