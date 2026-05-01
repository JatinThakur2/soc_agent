#!/usr/bin/env bash
# =============================================================================
# SOC-Agent — one-command setup and full pipeline runner (Linux/Mac)
# =============================================================================
# Usage:  bash scripts/setup_and_run.sh
#
# Optional flags:
#   --skip-setup    Skip venv creation & pip install (if already done)
#   --skip-data     Skip dataset download
#   --skip-kb       Skip MITRE ATT&CK RAG build
#   --skip-agents   Skip the LLM agent pipeline (no Anthropic API cost)
#   --only <stage>  Run only one stage: data|preprocess|features|train|kb|agents|eval
# =============================================================================

set -euo pipefail

# ---- Colors ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }

# ---- Parse args ----
SKIP_SETUP=0
SKIP_DATA=0
SKIP_KB=0
SKIP_AGENTS=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup)  SKIP_SETUP=1; shift ;;
        --skip-data)   SKIP_DATA=1; shift ;;
        --skip-kb)     SKIP_KB=1; shift ;;
        --skip-agents) SKIP_AGENTS=1; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Navigate to project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# ---- Check Python ----
if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Please install Python 3.10+."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python version: $PY_VERSION"

# ---- Setup: virtualenv + dependencies ----
if [[ $SKIP_SETUP -eq 0 ]]; then
    if [[ ! -d ".venv" ]]; then
        log "Creating virtual environment..."
        python3 -m venv .venv
        ok "Virtual environment created at .venv"
    else
        log "Virtual environment already exists."
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    log "Upgrading pip..."
    pip install --upgrade pip --quiet

    log "Installing requirements (this can take a few minutes)..."
    pip install -r requirements.txt --quiet
    ok "Dependencies installed."
else
    # shellcheck disable=SC1091
    source .venv/bin/activate
    log "Skipped setup (reusing existing .venv)"
fi

# ---- Check .env ----
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        warn ".env not found. Copying .env.example — please edit it with your API key."
        cp .env.example .env
    fi
fi

# Warn about missing API key (but only block if we're actually running agents)
if [[ $SKIP_AGENTS -eq 0 ]] && [[ -z "$ONLY" || "$ONLY" == "agents" ]]; then
    if ! grep -q 'sk-ant-' .env 2>/dev/null; then
        warn "ANTHROPIC_API_KEY does not appear set in .env"
        warn "Agent pipeline will fail unless you add it. Edit .env and re-run with --skip-setup --skip-data --skip-kb"
    fi
fi

# ---- Stage runner helper ----
run_stage() {
    local name="$1"; local cmd="$2"
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
        return
    fi
    log ""
    log "============================================================"
    log "STAGE: $name"
    log "============================================================"
    if eval "$cmd"; then
        ok "Stage '$name' complete."
    else
        err "Stage '$name' failed."
        exit 1
    fi
}

# ---- Run pipeline stages ----

if [[ $SKIP_DATA -eq 0 ]]; then
    run_stage "data"       "python -m src.data.download"
fi

run_stage "preprocess" "python -m src.data.preprocess"
run_stage "features"   "python -m src.features.engineer"
run_stage "train"      "python -m src.models.train"

if [[ $SKIP_KB -eq 0 ]]; then
    run_stage "kb"         "python -m src.knowledge_base.build"
fi

if [[ $SKIP_AGENTS -eq 0 ]]; then
    run_stage "agents"     "python -m src.agents.run_pipeline"
fi

run_stage "eval"       "python -m src.evaluation.evaluate"

# ---- Done ----
log ""
ok "============================================================"
ok " SOC-Agent pipeline complete!"
ok "============================================================"
echo ""
echo " Results saved to:"
echo "   - Figures:  results/figures/"
echo "   - Tables:   results/tables/"
echo "   - Logs:     results/logs/"
echo "   - Models:   models/"
echo ""
