"""
cluster_experiment.py
======================
Adds a THIRD modeling approach -- clustering-based anomaly detection --
alongside the statistical baseline and Isolation Forest already in
models_and_eval.py. This directly answers the project brief's own
suggestion to try "several modeling approaches (statistical profiling,
clustering, isolation forests, autoencoders)," which the project so far
only covered two of.

How it works: fit K-Means on benign-only training traffic. A flow's
anomaly score is its distance to the NEAREST cluster centroid -- the
idea being that benign traffic clusters into a handful of "normal
behavior modes" (e.g. different device types or activity states), and
a flow far from every one of those modes is anomalous. The detection
threshold is set at the 90th percentile of benign training distances,
chosen to match Isolation Forest's contamination=0.1 assumption so the
three models are compared on equivalent terms.

Deliberately restricted to CONTINUOUS_FEATURES only (same choice as
StatBaseline), for two reasons: (1) it keeps the three-model comparison
fair -- same feature set across all three -- and (2) K-Means distance
is itself a Euclidean-style metric, which inherits the same sensitivity
to a near-constant categorical column as StatBaseline's z-score did
(Section 4.5 of the paper) -- a single always-0.98-vs-1.0 binary flag
can distort centroid geometry the same way it distorted a z-score.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from models_and_eval import (
    CONTINUOUS_FEATURES, evaluate, balanced_test_set, standardize_by_reference
)


class ClusterBaseline:
    def __init__(self, n_clusters=8, contamination=0.1, random_state=1):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X_benign):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                               n_init=10).fit(X_benign[CONTINUOUS_FEATURES])
        train_distances = self._distances(X_benign)
        # Threshold at the (1 - contamination) percentile of BENIGN training
        # distances -- i.e. we expect ~10% of benign training points to sit
        # beyond "typical" cluster distance, matching IsolationForest's
        # contamination=0.1 assumption for a fair three-way comparison.
        self.threshold_ = np.percentile(train_distances, (1 - self.contamination) * 100)
        return self

    def _distances(self, X):
        centroids = self.kmeans_.cluster_centers_
        X_vals = X[CONTINUOUS_FEATURES].values
        # distance from each point to every centroid, take the minimum
        dists = np.linalg.norm(X_vals[:, None, :] - centroids[None, :, :], axis=2)
        return dists.min(axis=1)

    def predict(self, X):
        distances = self._distances(X)
        return (distances > self.threshold_).astype(int)


def run_within_environment_cluster(df, name):
    from sklearn.model_selection import train_test_split
    print(f"\n=== Within-environment (ClusterBaseline): {name} ===")
    benign = df[df.label == 0]
    train_benign, holdout_benign = train_test_split(benign, test_size=0.3, random_state=1)
    attacks = df[df.label == 1]
    test_df = balanced_test_set(holdout_benign, attacks, ratio=1.0)
    test_df = standardize_by_reference(train_benign, test_df)
    train_benign = standardize_by_reference(train_benign, train_benign)

    model = ClusterBaseline().fit(train_benign)
    pred = model.predict(test_df)
    evaluate(test_df.label.values, pred, "ClusterBaseline")


def run_cross_environment_cluster(train_df, test_df, train_name, test_name):
    print(f"\n=== Cross-environment (ClusterBaseline): {train_name} -> {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_df = balanced_test_set(test_benign, test_attacks, ratio=1.0)
    test_df = standardize_by_reference(test_benign, test_df)
    train_benign = standardize_by_reference(train_benign, train_benign)

    model = ClusterBaseline().fit(train_benign)
    pred = model.predict(test_df)
    evaluate(test_df.label.values, pred, "ClusterBaseline")


if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")

    run_within_environment_cluster(iot23, "IoT-23")
    run_within_environment_cluster(ciciot, "CICIoT2023")
    run_cross_environment_cluster(iot23, ciciot, "IoT-23", "CICIoT2023")
    run_cross_environment_cluster(ciciot, iot23, "CICIoT2023", "IoT-23")
