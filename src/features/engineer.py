"""
Phase 2: Feature engineering & selection.

1. Consensus-based feature selection (MI + XGBoost)
2. SOC-context feature engineering (6 new features)
3. Encoding + scaling
4. Train/test split + resampling (train only)
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

from src.utils.logger import get_logger
from src.utils.paths import load_config, get_path

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------

def duration_bucket(val_us):
    """Microseconds → duration category."""
    if val_us < 1e6:      return "micro"
    elif val_us < 1e7:    return "short"
    elif val_us < 6e7:    return "medium"
    else:                 return "long"


def payload_category(val):
    if val == 0:       return "empty"
    elif val < 100:    return "tiny"
    elif val < 500:    return "small"
    elif val < 1500:   return "medium"
    else:              return "large"


def port_bin(p):
    if p <= 1023:    return "well_known"
    elif p <= 49151: return "registered"
    else:            return "ephemeral"


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add 6 SOC-context features."""
    df = df.copy()

    # 1. Fwd/Bwd ratio
    fwd_col = next((c for c in ["Total Fwd Packets", "Tot Fwd Pkts"] if c in df.columns), None)
    bwd_col = next((c for c in ["Total Backward Packets", "Tot Bwd Pkts"] if c in df.columns), None)
    if fwd_col and bwd_col:
        df["fwd_bwd_ratio"] = df[fwd_col] / (df[bwd_col] + 1e-8)
        log.info("Engineered: fwd_bwd_ratio")

    # 2. Flow duration bucket
    dur_col = next((c for c in ["Flow Duration"] if c in df.columns), None)
    if dur_col:
        df["duration_bucket"] = df[dur_col].apply(duration_bucket)
        log.info("Engineered: duration_bucket")

    # 3. Payload size category
    pkt_col = next((c for c in ["Average Packet Size", "Pkt Size Avg", "Avg Fwd Segment Size"]
                    if c in df.columns), None)
    if pkt_col:
        df["payload_category"] = df[pkt_col].apply(payload_category)
        log.info("Engineered: payload_category")

    # 4. Port bin + 5. Port→service
    dport_col = next((c for c in ["Destination Port", "Dst Port"] if c in df.columns), None)
    if dport_col:
        df["port_category"] = df[dport_col].apply(port_bin)
        port_map = cfg["features"]["port_service_map"]
        df["service"] = df[dport_col].map(port_map).fillna("Other")
        log.info("Engineered: port_category, service")

    # 6. SYN-without-ACK (flag anomaly)
    syn_col = next((c for c in ["SYN Flag Count", "SYN Flag Cnt"] if c in df.columns), None)
    ack_col = next((c for c in ["ACK Flag Count", "ACK Flag Cnt"] if c in df.columns), None)
    if syn_col and ack_col:
        df["syn_no_ack"] = ((df[syn_col] >= 1) & (df[ack_col] == 0)).astype(int)
        log.info("Engineered: syn_no_ack")

    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def consensus_feature_selection(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> list:
    """Three-stage: MI → XGBoost → intersection."""
    numeric_X = X.select_dtypes(include=[np.number])
    log.info(f"Running consensus selection on {numeric_X.shape[1]} numeric features...")

    # Stage 1: Mutual Information
    log.info("Stage 1: Mutual Information...")
    mi_scores = mutual_info_classif(numeric_X, y, random_state=42, n_neighbors=3)
    mi_ranking = pd.Series(mi_scores, index=numeric_X.columns).sort_values(ascending=False)
    mi_top = mi_ranking.head(cfg["features"]["mi_top_k"]).index.tolist()
    log.info(f"  MI top {len(mi_top)}: {mi_top[:5]}...")

    # Stage 2: XGBoost importance
    log.info("Stage 2: XGBoost importance...")
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    # Try GPU, fall back to CPU
    try:
        model = xgb.XGBClassifier(tree_method="gpu_hist", **xgb_params)
        model.fit(numeric_X, y)
    except Exception:
        model = xgb.XGBClassifier(tree_method="hist", **xgb_params)
        model.fit(numeric_X, y)

    xgb_imp = pd.Series(model.feature_importances_, index=numeric_X.columns).sort_values(ascending=False)
    xgb_top = xgb_imp.head(cfg["features"]["xgb_top_k"]).index.tolist()
    log.info(f"  XGBoost top {len(xgb_top)}: {xgb_top[:5]}...")

    # Stage 3: Consensus
    consensus = sorted(set(mi_top) & set(xgb_top))
    log.info(f"Consensus (intersection): {len(consensus)} features")

    return consensus, mi_ranking, xgb_imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Load cleaned data
    cleaned_path = get_path(cfg, "data_cleaned") / "combined_cleaned.parquet"
    if not cleaned_path.exists():
        log.error(f"Cleaned data not found. Run preprocess.py first.")
        sys.exit(1)

    log.info(f"Loading {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    log.info(f"Loaded: {df.shape}")

    # Drop source dataset tag
    if "__source_dataset__" in df.columns:
        df = df.drop(columns=["__source_dataset__"])

    # Separate label
    y_raw = df["Label"].values
    X = df.drop(columns=["Label"])

    # Engineer new features BEFORE encoding
    X = engineer_features(X, cfg)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    log.info(f"Label classes: {list(le.classes_)}")

    # One-hot encode new categorical features
    cat_cols = [c for c in ["duration_bucket", "payload_category", "port_category", "service"]
                if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
        log.info(f"One-hot encoded: {cat_cols}")

    # Feature selection (on numeric features only, using full data for ranking)
    consensus, mi_ranking, xgb_imp = consensus_feature_selection(X, y, cfg)

    # Keep consensus features + all engineered ones
    engineered = [c for c in X.columns if any(
        c.startswith(p) for p in ["duration_bucket_", "payload_category_", "port_category_", "service_"]
    )] + [c for c in ["fwd_bwd_ratio", "syn_no_ack"] if c in X.columns]

    final_features = sorted(set(consensus) | set(engineered))
    final_features = [c for c in final_features if c in X.columns]
    log.info(f"Final feature set: {len(final_features)} features")
    X_final = X[final_features]

    # Save rankings & feature list
    tables_dir = get_path(cfg, "results_tables")
    mi_ranking.to_csv(tables_dir / "mi_ranking.csv", header=["mutual_info"])
    xgb_imp.to_csv(tables_dir / "xgb_importance.csv", header=["importance"])
    pd.Series(final_features).to_csv(tables_dir / "selected_features.csv", index=False, header=["feature"])

    # Train/test split
    split_cfg = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=split_cfg["test_size"],
        stratify=y if split_cfg["stratify"] else None,
        random_state=cfg["random_seed"],
    )
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Scale
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Resample (train only!)
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    benign_idx = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else (
        list(le.classes_).index("Benign") if "Benign" in le.classes_ else 0
    )
    benign_count = int((y_train == benign_idx).sum())
    benign_cap = min(cfg["resampling"]["benign_cap"], benign_count)

    # Count min class — SMOTE k_neighbors must be less than smallest class size
    class_counts = pd.Series(y_train).value_counts()
    min_class = class_counts.min()
    k_neighbors = min(cfg["resampling"]["smote_k_neighbors"], max(1, min_class - 1))
    log.info(f"Resampling: benign cap={benign_cap}, SMOTE k_neighbors={k_neighbors}")

    try:
        pipeline = ImbPipeline([
            ("under", RandomUnderSampler(
                sampling_strategy={benign_idx: benign_cap},
                random_state=42
            )),
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ])
        X_train_rs, y_train_rs = pipeline.fit_resample(X_train_s, y_train)
        log.info(f"After resampling: {X_train_rs.shape}")
        log.info(f"Class distribution after resampling:\n{pd.Series(y_train_rs).value_counts().to_string()}")
    except Exception as e:
        log.warning(f"Resampling failed ({e}), using original training set")
        X_train_rs, y_train_rs = X_train_s, y_train

    # Save everything
    proc = get_path(cfg, "data_processed")
    splits = get_path(cfg, "data_splits")

    X_train_rs.to_parquet(splits / "X_train.parquet", index=False)
    X_test_s.to_parquet(splits / "X_test.parquet", index=False)
    np.save(splits / "y_train.npy", y_train_rs)
    np.save(splits / "y_test.npy", y_test)

    joblib.dump(scaler, proc / "scaler.pkl")
    joblib.dump(le, proc / "label_encoder.pkl")
    with open(proc / "feature_list.json", "w") as f:
        json.dump(final_features, f)

    log.info(f"Saved processed data to {splits} and {proc}")


if __name__ == "__main__":
    main()
