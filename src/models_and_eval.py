"""
Two fingerprinting approaches, trained on BENIGN traffic only (this is
the whole point of anomaly-based detection: you never need labeled
attacks to build the fingerprint, only a sample of normal behavior):

  1. StatBaseline      -- per-feature mean/std profile of benign traffic;
                           flags anything more than `z_thresh` std devs
                           from the benign mean on ANY feature.
  2. IsolationForest   -- sklearn's implementation, same benign-only fit.

Then: within-environment evaluation (train/test split on the SAME
dataset) and cross-environment evaluation (train on one dataset, test
on the other, unmodified) using precision/recall/F1/FPR.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size",
            "is_tcp", "is_udp", "is_icmp"]


class StatBaseline:
    """Flags a flow as anomalous if any feature is > z_thresh std devs
    from the mean of the benign training data."""
    def __init__(self, z_thresh=3.0):
        self.z_thresh = z_thresh

    def fit(self, X_benign):
        self.mean_ = X_benign.mean(axis=0)
        self.std_ = X_benign.std(axis=0).replace(0, 1e-6)
        return self

    def predict(self, X):
        z = ((X - self.mean_) / self.std_).abs()
        flagged = (z > self.z_thresh).any(axis=1)
        return flagged.astype(int).values  # 1 = anomalous, 0 = normal


def evaluate(y_true, y_pred, label=""):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"  [{label}] precision={prec:.3f} recall={rec:.3f} "
          f"f1={f1:.3f} FPR={fpr:.3f}  (n={len(y_true)}, attacks={y_true.sum()})")
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def run_within_environment(df, name):
    print(f"\n=== Within-environment: {name} ===")
    benign = df[df.label == 0]
    train_benign, holdout_benign = train_test_split(
        benign, test_size=0.3, random_state=1)
    attacks = df[df.label == 1]
    test_df = pd.concat([holdout_benign, attacks])

    stat = StatBaseline(z_thresh=3.0).fit(train_benign[FEATURES])
    iso = IsolationForest(contamination=0.1, random_state=1).fit(train_benign[FEATURES])

    pred_stat = stat.predict(test_df[FEATURES])
    pred_iso = (iso.predict(test_df[FEATURES]) == -1).astype(int)

    evaluate(test_df.label.values, pred_stat, "StatBaseline")
    evaluate(test_df.label.values, pred_iso, "IsolationForest")
    return stat, iso, train_benign


def run_cross_environment(train_df, test_df, train_name, test_name):
    print(f"\n=== Cross-environment: trained on {train_name}, tested on {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    stat = StatBaseline(z_thresh=3.0).fit(train_benign[FEATURES])
    iso = IsolationForest(contamination=0.1, random_state=1).fit(train_benign[FEATURES])

    pred_stat = stat.predict(test_df[FEATURES])
    pred_iso = (iso.predict(test_df[FEATURES]) == -1).astype(int)

    evaluate(test_df.label.values, pred_stat, "StatBaseline")
    evaluate(test_df.label.values, pred_iso, "IsolationForest")


if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")

    run_within_environment(iot23, "IoT-23")
    run_within_environment(ciciot, "CICIoT2023")

    run_cross_environment(iot23, ciciot, "IoT-23", "CICIoT2023")
    run_cross_environment(ciciot, iot23, "CICIoT2023", "IoT-23")
