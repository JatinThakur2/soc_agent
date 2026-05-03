# =============================================================================
# SOC-Agent - one-command setup and full pipeline runner (Windows PowerShell)
# =============================================================================
# Usage:  .\scripts\setup_and_run.ps1
#
# Optional flags:
#   -SkipSetup    Skip venv creation and pip install
#   -SkipData     Skip dataset download
#   -SkipKB       Skip MITRE ATT&CK RAG build
#   -SkipAgents   Skip the LLM agent pipeline (no Anthropic API cost)
#   -Only <name>  Run only one stage: data|preprocess|features|train|kb|agents|eval
# =============================================================================

param(
    [switch]$SkipSetup,
    [switch]$SkipData,
    [switch]$SkipKB,
    [switch]$SkipAgents,
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"

function Log  { param([string]$msg) Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Cyan }
function Ok   { param([string]$msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn { param([string]$msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Err  { param([string]$msg) Write-Host "[ERR] $msg" -ForegroundColor Red }

# Navigate to project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot
Log "Project root: $ProjectRoot"

# Check Python
try {
    $pyVersion = python --version 2>&1
    Log "Python version: $pyVersion"
} catch {
    Err "Python not found. Install Python 3.10+ from python.org"
    exit 1
}

# Setup
if (-not $SkipSetup) {
    if (-not (Test-Path ".venv")) {
        Log "Creating virtual environment..."
        python -m venv .venv
        Ok "Virtual environment created."
    } else {
        Log "Virtual environment already exists."
    }

    & ".\.venv\Scripts\Activate.ps1"

    Log "Upgrading pip..."
    python -m pip install --upgrade pip --quiet

    Log "Installing requirements (this can take a few minutes)..."
    pip install -r requirements.txt --quiet
    Ok "Dependencies installed."
} else {
    & ".\.venv\Scripts\Activate.ps1"
    Log "Skipped setup (reusing existing .venv)"
}

# Check .env
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Warn ".env not found. Copying .env.example - please edit it with your API key."
        Copy-Item ".env.example" ".env"
    }
}

# API key check
if (-not $SkipAgents -and ($Only -eq "" -or $Only -eq "agents")) {
    $envContent = if (Test-Path ".env") { Get-Content ".env" -Raw } else { "" }
    if ($envContent -notmatch 'sk-ant-') {
        Warn "ANTHROPIC_API_KEY does not appear set in .env"
        Warn "Agent pipeline will fail. Edit .env and re-run with: -SkipSetup -SkipData -SkipKB"
    }
}

# Stage runner
function Run-Stage {
    param([string]$Name, [string]$Cmd)
    if ($Only -ne "" -and $Only -ne $Name) { return }
    Log ""
    Log "============================================================"
    Log "STAGE: $Name"
    Log "============================================================"
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        Err "Stage '$Name' failed."
        exit 1
    }
    Ok "Stage '$Name' complete."
}

# Run pipeline
if (-not $SkipData) { Run-Stage "data" "python -m src.data.download" }
Run-Stage "preprocess" "python -m src.data.preprocess"
Run-Stage "features"   "python -m src.features.engineer"
Run-Stage "train"      "python -m src.models.train"
if (-not $SkipKB)     { Run-Stage "kb"     "python -m src.knowledge_base.build" }
if (-not $SkipAgents) { Run-Stage "agents" "python -m src.agents.run_pipeline" }
Run-Stage "eval"       "python -m src.evaluation.evaluate"

# Done
Log ""
Ok "============================================================"
Ok " SOC-Agent pipeline complete!"
Ok "============================================================"
Write-Host ""
Write-Host " Results saved to:"
Write-Host "   - Figures:  results/figures/"
Write-Host "   - Tables:   results/tables/"
Write-Host "   - Logs:     results/logs/"
Write-Host "   - Models:   models/"
Write-Host ""
