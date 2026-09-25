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
CONTINUOUS_FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size"]

def standardize_by_reference(reference_benign_df, target_df):
    """
    Z-scores target_df's features using ONLY reference_benign_df's
    mean/std -- reference_benign_df must contain BENIGN rows only.
    Never compute reference stats from a mixed benign+attack set --
    that lets attacks skew the very yardstick meant to detect them.
    """
    mean = reference_benign_df[FEATURES].mean()
    std = reference_benign_df[FEATURES].std().replace(0, 1e-6)
    out = target_df.copy()
    out[FEATURES] = (target_df[FEATURES] - mean) / std
    return out


class StatBaseline:
    def __init__(self, z_thresh=3.0):
        self.z_thresh = z_thresh

    def fit(self, X_benign):
        self.mean_ = X_benign[CONTINUOUS_FEATURES].mean(axis=0)
        self.std_ = X_benign[CONTINUOUS_FEATURES].std(axis=0).clip(lower=0.05)
        return self

    def predict(self, X):
        z = ((X[CONTINUOUS_FEATURES] - self.mean_) / self.std_).abs()
        flagged = (z > self.z_thresh).any(axis=1)
        return flagged.astype(int).values


def evaluate(y_true, y_pred, label=""):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"  [{label}] precision={prec:.3f} recall={rec:.3f} "
          f"f1={f1:.3f} FPR={fpr:.3f}  (n={len(y_true)}, attacks={y_true.sum()})")
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr}

def balanced_test_set(benign_df, attack_df, ratio=1.0, random_state=1):
    """
    Caps the attack rows in a test set so precision/recall aren't
    trivially inflated by one class dominating. `ratio` = attacks per
    benign row (1.0 = balanced 50/50 test set). If attack_df already
    has fewer rows than the cap, we just keep all of them.
    """
    n_attacks_wanted = int(len(benign_df) * ratio)
    if len(attack_df) > n_attacks_wanted:
        attack_df = attack_df.sample(n=n_attacks_wanted, random_state=random_state)
    return pd.concat([benign_df, attack_df])

def run_within_environment(df, name):
    print(f"\n=== Within-environment: {name} ===")
    benign = df[df.label == 0]
    train_benign, holdout_benign = train_test_split(
    benign, test_size=0.3, random_state=1)
    attacks = df[df.label == 1]
    test_df = balanced_test_set(holdout_benign, attacks, ratio=1.0)
    test_df = standardize_by_reference(train_benign, test_df)
    train_benign = standardize_by_reference(train_benign, train_benign)

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
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_df = balanced_test_set(test_benign, test_attacks, ratio=1.0)
    test_df = standardize_by_reference(test_benign, test_df)
    train_benign = standardize_by_reference(train_benign, train_benign)
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
