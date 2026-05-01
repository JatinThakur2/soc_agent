"""
Pre-flight check — verify setup is correct before running the full pipeline.

Usage:
    python -m scripts.verify_setup
"""
import os
import sys
from pathlib import Path


def check(name, passed, detail=""):
    mark = "\033[92m[OK]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    print(f"  {mark}  {name}" + (f" — {detail}" if detail else ""))
    return passed


def main():
    print("\n" + "=" * 60)
    print("SOC-Agent Setup Verification")
    print("=" * 60 + "\n")

    all_ok = True

    # ---- Python version ----
    print("Python:")
    py_ok = sys.version_info >= (3, 10)
    all_ok &= check(
        f"Python version: {sys.version_info.major}.{sys.version_info.minor}",
        py_ok,
        "needs 3.10+" if not py_ok else "",
    )

    # ---- Core dependencies ----
    print("\nDependencies:")
    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "shap",
                "langchain", "langchain_anthropic", "chromadb",
                "sentence_transformers", "kaggle", "imblearn", "torch"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            all_ok &= check(f"{pkg}", True, f"v{ver}")
        except ImportError:
            all_ok &= check(f"{pkg}", False, "not installed")

    # ---- Kaggle credentials ----
    print("\nKaggle credentials:")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    all_ok &= check(
        f"kaggle.json at {kaggle_json}",
        kaggle_json.exists(),
        "see README step 1" if not kaggle_json.exists() else "",
    )

    # ---- Anthropic API key ----
    print("\nAnthropic API key:")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    all_ok &= check(
        "ANTHROPIC_API_KEY set",
        api_key.startswith("sk-ant-"),
        "edit .env file" if not api_key.startswith("sk-ant-") else "(key hidden)",
    )

    # ---- GPU (optional) ----
    print("\nGPU (optional):")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            check(f"CUDA available", True, f"{gpu}, {vram:.1f} GB")
        else:
            check("CUDA available", False, "CPU training will be slower but works")
    except ImportError:
        pass

    # ---- Disk space ----
    print("\nDisk space:")
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    all_ok &= check(
        f"Free disk space: {free_gb:.1f} GB",
        free_gb >= 10,
        "needs at least 10 GB",
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    if all_ok:
        print("\033[92mAll checks passed! Ready to run the pipeline.\033[0m")
        print("\nRun:  bash scripts/setup_and_run.sh  (Linux/Mac)")
        print("       .\\scripts\\setup_and_run.ps1   (Windows)")
    else:
        print("\033[91mSome checks failed. Fix the issues above, then re-run.\033[0m")
    print("=" * 60 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
