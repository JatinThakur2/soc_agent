"""
Phase 5: Evaluation & metrics aggregation.

Compiles classification reports from all models, generates comparison plots,
and computes SOC-level metrics from the agent pipeline results.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger
from src.utils.paths import load_config, get_path

log = get_logger(__name__)


def set_plot_style(cfg):
    plt.rcParams.update({
        "figure.dpi": cfg["evaluation"]["dpi"],
        "savefig.dpi": cfg["evaluation"]["dpi"],
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
    })


def compile_model_reports(cfg):
    """Consolidate per-model classification reports into one comparison table."""
    tables = get_path(cfg, "results_tables")
    figures = get_path(cfg, "results_figures")

    report_files = {
        "XGBoost":       tables / "xgboost_report.csv",
        "Random Forest": tables / "random_forest_report.csv",
        "DNN":           tables / "dnn_report.csv",
    }

    available = {name: p for name, p in report_files.items() if p.exists()}
    if not available:
        log.warning("No model reports found — skipping model comparison.")
        return

    # Extract weighted avg + macro avg F1/precision/recall for each model
    summary = []
    for name, p in available.items():
        df = pd.read_csv(p, index_col=0)
        if "weighted avg" in df.index:
            wavg = df.loc["weighted avg"]
            summary.append({
                "Model": name,
                "Precision": wavg.get("precision", 0),
                "Recall": wavg.get("recall", 0),
                "F1-Score": wavg.get("f1-score", 0),
                "Accuracy": df.loc["accuracy", "precision"] if "accuracy" in df.index else np.nan,
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        out = tables / "model_comparison.csv"
        summary_df.to_csv(out, index=False)
        log.info(f"Model comparison:\n{summary_df.to_string(index=False)}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        summary_df.set_index("Model")[["Precision", "Recall", "F1-Score"]].plot.bar(ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Classifier Comparison — Weighted Metrics")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig_path = figures / f"model_comparison.{cfg['evaluation']['figure_format']}"
        plt.savefig(fig_path)
        plt.close()
        log.info(f"Saved model comparison chart to {fig_path}")


def per_class_f1_chart(cfg):
    """Per-class F1 bar chart for the primary (XGBoost) model."""
    tables = get_path(cfg, "results_tables")
    figures = get_path(cfg, "results_figures")
    report_path = tables / "xgboost_report.csv"
    if not report_path.exists():
        return

    df = pd.read_csv(report_path, index_col=0)
    # Per-class rows: anything that's not an aggregate
    aggregate_rows = {"accuracy", "macro avg", "weighted avg"}
    class_df = df[~df.index.isin(aggregate_rows)].sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(class_df))))
    class_df["f1-score"].plot.barh(ax=ax, color="#378ADD")
    ax.set_xlabel("F1-Score")
    ax.set_title("Per-Class F1-Score — XGBoost")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = figures / f"per_class_f1_xgboost.{cfg['evaluation']['figure_format']}"
    plt.savefig(fig_path)
    plt.close()
    log.info(f"Saved per-class F1 chart to {fig_path}")


def class_distribution_plots(cfg):
    """Visualize label distribution in the processed training set."""
    splits = get_path(cfg, "data_splits")
    proc = get_path(cfg, "data_processed")
    figures = get_path(cfg, "results_figures")

    import joblib
    try:
        y_train = np.load(splits / "y_train.npy")
        y_test = np.load(splits / "y_test.npy")
        le = joblib.load(proc / "label_encoder.pkl")
    except Exception:
        log.warning("Splits not found for class distribution plot.")
        return

    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()

    classes = list(le.classes_)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    train_counts.index = [classes[i] for i in train_counts.index]
    test_counts.index = [classes[i] for i in test_counts.index]

    train_counts.plot.bar(ax=axes[0], color="#0F6E56")
    axes[0].set_title("Training Set (after resampling)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    test_counts.plot.bar(ax=axes[1], color="#993C1D")
    axes[1].set_title("Test Set (original distribution)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig_path = figures / f"class_distribution.{cfg['evaluation']['figure_format']}"
    plt.savefig(fig_path)
    plt.close()
    log.info(f"Saved class distribution to {fig_path}")


def analyze_agent_results(cfg):
    """SOC-level metrics from agent pipeline output."""
    logs = get_path(cfg, "results_logs")
    tables = get_path(cfg, "results_tables")
    results_path = logs / "agent_pipeline_results.json"
    if not results_path.exists():
        log.warning("Agent pipeline results not found — skipping SOC metrics.")
        return

    with open(results_path) as f:
        results = json.load(f)

    completed = [r for r in results if "coordinator" in r]
    if not completed:
        log.warning("No completed agent runs to analyze.")
        return

    # Severity distribution
    severities = [r["coordinator"].get("final_severity", "Unknown") for r in completed]
    sev_counts = pd.Series(severities).value_counts()

    # Agreement between ML prediction and ground truth
    ml_correct = 0
    for r in completed:
        alert = r.get("alert", {})
        if alert.get("predicted_label", "").upper() == r.get("ground_truth", "").upper():
            ml_correct += 1
    ml_accuracy = ml_correct / len(completed) if completed else 0

    # Escalation decisions
    escalated = sum(1 for r in completed if r.get("triage", {}).get("escalate"))

    # Action requirements
    action_required = sum(
        1 for r in completed
        if r.get("coordinator", {}).get("analyst_action_required")
    )

    # MITRE techniques identified
    all_techniques = []
    for r in completed:
        techs = r.get("coordinator", {}).get("attck_techniques", [])
        all_techniques.extend(techs)
    top_techniques = pd.Series(all_techniques).value_counts().head(10)

    summary = {
        "total_alerts_processed": len(results),
        "completed_agent_runs": len(completed),
        "skipped_benign": sum(1 for r in results if r.get("skipped")),
        "errors": sum(1 for r in results if r.get("error")),
        "ml_accuracy_on_sample": round(ml_accuracy, 4),
        "escalated_count": escalated,
        "action_required_count": action_required,
        "severity_distribution": sev_counts.to_dict(),
        "top_mitre_techniques": top_techniques.to_dict(),
    }

    out = tables / "agent_pipeline_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Agent pipeline summary:\n{json.dumps(summary, indent=2)}")

    # Plot severity distribution
    if len(sev_counts) > 0:
        figures = get_path(cfg, "results_figures")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {
            "Critical": "#A32D2D",
            "High": "#D85A30",
            "Medium": "#EF9F27",
            "Low": "#378ADD",
            "Informational": "#888780",
            "Unknown": "#B4B2A9",
        }
        bar_colors = [colors.get(s, "#888780") for s in sev_counts.index]
        sev_counts.plot.bar(ax=ax, color=bar_colors)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Alert Count")
        ax.set_title("Triage Severity Distribution — Agent Pipeline")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig_path = figures / f"severity_distribution.{cfg['evaluation']['figure_format']}"
        plt.savefig(fig_path)
        plt.close()
        log.info(f"Saved severity distribution to {fig_path}")


def feature_importance_chart(cfg):
    """Plot top-N features from XGBoost importance."""
    tables = get_path(cfg, "results_tables")
    figures = get_path(cfg, "results_figures")
    xgb_path = tables / "xgb_importance.csv"
    if not xgb_path.exists():
        return

    imp = pd.read_csv(xgb_path, index_col=0).squeeze()
    top20 = imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    top20.plot.barh(ax=ax, color="#534AB7")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances — XGBoost")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = figures / f"feature_importance_xgb.{cfg['evaluation']['figure_format']}"
    plt.savefig(fig_path)
    plt.close()
    log.info(f"Saved feature importance chart to {fig_path}")


def main():
    cfg = load_config()
    set_plot_style(cfg)

    log.info("=" * 60)
    log.info("Phase 5: Evaluation")
    log.info("=" * 60)

    compile_model_reports(cfg)
    per_class_f1_chart(cfg)
    class_distribution_plots(cfg)
    feature_importance_chart(cfg)
    analyze_agent_results(cfg)

    log.info("Evaluation complete. Check results/ for figures and tables.")


if __name__ == "__main__":
    main()
