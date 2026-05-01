"""Configuration loader — reads config.yaml and applies quick_mode overrides."""
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load config.yaml and apply quick_mode overrides if set."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("quick_mode", False):
        overrides = cfg.get("quick_mode_overrides", {})
        cfg["datasets"]["sample_fraction"] = overrides.get("sample_fraction", 0.05)
        cfg["models"]["dnn"]["epochs"] = overrides.get("dnn_epochs", 5)
        cfg["agents"]["n_eval_alerts"] = overrides.get("n_eval_alerts", 10)
        cfg["resampling"]["benign_cap"] = overrides.get("benign_cap", 50000)
        cfg["shap"]["sample_size"] = overrides.get("shap_sample_size", 500)

    # Resolve paths relative to project root
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / val)

    cfg["project_root"] = str(PROJECT_ROOT)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Get a Path object for a config path key (creates parent dirs)."""
    p = Path(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p
