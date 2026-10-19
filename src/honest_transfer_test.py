"""
honest_transfer_test.py
========================
Diagnoses and corrects a methodology flaw: run_cross_environment() in
models_and_eval.py standardizes the TEST set against its OWN benign
reference, not the TRAINING environment's. Because standardizing any
dataset against its own mean/std always yields mean=0, std=1 exactly,
StatBaseline's "trained" model carries zero actual information from
the training dataset -- its cross-environment "transfer" success was
really just "CICIoT2023's own benign/attack separation happens to be
clean in self-standardized space," independent of training source.

This script tests TWO honest alternatives, where the model is fit on
training data and test data is standardized using ONLY the training
environment's statistics (no test-side recalibration):

  (a) All four continuous features, train-derived standardization.
      Expected to struggle, because CICIoT2023's IAT-based duration
      proxy and IoT-23's true duration are not the same unit of
      measurement -- this was the original scale-mismatch problem,
      now allowed to surface honestly instead of being hidden.

  (b) Same, but EXCLUDING 'duration' -- the one feature with a real
      unit incompatibility across datasets. Only total_bytes,
      total_pkts, and avg_pkt_size are genuinely comparable
      measurements in both datasets. This tests whether real transfer
      exists once we stop comparing apples to oranges.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models_and_eval import StatBaseline, evaluate, balanced_test_set

FULL_CONTINUOUS = ["duration", "total_bytes", "total_pkts", "avg_pkt_size"]
COMPARABLE_ONLY = ["total_bytes", "total_pkts", "avg_pkt_size"]


class StatBaselineCustomFeatures(StatBaseline):
    """Same logic as StatBaseline, but lets us choose which continuous
    features to use, instead of the hardcoded CONTINUOUS_FEATURES."""
    def __init__(self, features, z_thresh=3.0):
        super().__init__(z_thresh)
        self.features = features

    def fit(self, X_benign):
        self.mean_ = X_benign[self.features].mean(axis=0)
        self.std_ = X_benign[self.features].std(axis=0).clip(lower=0.05)
        return self

    def predict(self, X):
        z = ((X[self.features] - self.mean_) / self.std_).abs()
        return (z > self.z_thresh).any(axis=1).astype(int).values


def honest_cross_environment(train_df, test_df, train_name, test_name, features, label):
    print(f"\n=== HONEST transfer test [{label}]: {train_name} -> {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_bal = balanced_test_set(test_benign, test_attacks, ratio=1.0)

    # Standardize using ONLY training statistics -- applied to train AND
    # test alike. The test set does NOT get to recalibrate itself.
    mean = train_benign[features].mean()
    std = train_benign[features].std().replace(0, 1e-6)

    train_std = train_benign.copy()
    train_std[features] = (train_benign[features] - mean) / std
    test_std = test_bal.copy()
    test_std[features] = (test_bal[features] - mean) / std

    model = StatBaselineCustomFeatures(features, z_thresh=3.0).fit(train_std)
    pred = model.predict(test_std)
    evaluate(test_bal.label.values, pred, label)


if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")

    print("##### (a) All 4 continuous features, honest train-only standardization #####")
    honest_cross_environment(iot23, ciciot, "IoT-23", "CICIoT2023", FULL_CONTINUOUS, "4-feature")
    honest_cross_environment(ciciot, iot23, "CICIoT2023", "IoT-23", FULL_CONTINUOUS, "4-feature")

    print("\n##### (b) Excluding duration/IAT (unit-incompatible feature) #####")
    honest_cross_environment(iot23, ciciot, "IoT-23", "CICIoT2023", COMPARABLE_ONLY, "3-feature")
    honest_cross_environment(ciciot, iot23, "CICIoT2023", "IoT-23", COMPARABLE_ONLY, "3-feature")
