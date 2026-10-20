"""
verify_isoforest_not_degenerate.py
====================================
StatBaseline's cross-environment "transfer" turned out to be fake: its
model always collapses to mean=0, std=1 regardless of training data,
because it's just two numbers standardized to a fixed point.

Isolation Forest's model is real tree structure, not two numbers, so
it SHOULD be immune to this same degeneracy -- but that was reasoning,
not a verified fact. This script checks it directly: fit two Isolation
Forests on the two different self-standardized datasets, apply BOTH to
the exact same fixed set of test points, and measure how often they
agree.

  - Agreement near 1.0 -> training source doesn't matter -> same
    degeneracy as StatBaseline -> the Section 5.3/5.4 ablation results
    need the same honest re-testing per_attack_breakdown.py just got.
  - Agreement meaningfully below 1.0 -> the two models genuinely
    learned different things from their respective training data ->
    not degenerate -> existing Isolation Forest/ClusterBaseline
    results can be trusted as reported.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from models_and_eval import standardize_by_reference, FEATURES, CONTINUOUS_FEATURES

iot23 = pd.read_csv("output/features_iot23.csv")
ciciot = pd.read_csv("output/features_ciciot2023.csv")

iot23_benign = iot23[iot23.label == 0]
ciciot_benign = ciciot[ciciot.label == 0]

iot23_std = standardize_by_reference(iot23_benign, iot23_benign)
ciciot_std = standardize_by_reference(ciciot_benign, ciciot_benign)

# A fixed set of test points, standardized against CICIoT2023's own
# benign reference (the normal cross-environment evaluation approach).
n = min(2000, len(ciciot))
test_fixed_full = standardize_by_reference(ciciot_benign, ciciot.sample(n, random_state=1))

print("=== Isolation Forest ===")
iso_from_iot23 = IsolationForest(contamination=0.1, random_state=1).fit(iot23_std[FEATURES])
iso_from_ciciot = IsolationForest(contamination=0.1, random_state=1).fit(ciciot_std[FEATURES])
pred_a = iso_from_iot23.predict(test_fixed_full[FEATURES])
pred_b = iso_from_ciciot.predict(test_fixed_full[FEATURES])
agreement = (pred_a == pred_b).mean()
print(f"Agreement between IoT-23-trained and CICIoT2023-trained models: {agreement:.3f}")
print(f"(near 1.0 = degenerate like StatBaseline; well below 1.0 = genuinely different models)")

print("\n=== K-Means ClusterBaseline-style (distance to nearest centroid) ===")
import numpy as np
def cluster_predict(kmeans, X, threshold):
    centroids = kmeans.cluster_centers_
    dists = np.linalg.norm(X.values[:, None, :] - centroids[None, :, :], axis=2).min(axis=1)
    return (dists > threshold).astype(int)

km_iot23 = KMeans(n_clusters=8, random_state=1, n_init=10).fit(iot23_std[CONTINUOUS_FEATURES])
km_ciciot = KMeans(n_clusters=8, random_state=1, n_init=10).fit(ciciot_std[CONTINUOUS_FEATURES])

d_train_a = np.linalg.norm(iot23_std[CONTINUOUS_FEATURES].values[:, None, :] - km_iot23.cluster_centers_[None, :, :], axis=2).min(axis=1)
d_train_b = np.linalg.norm(ciciot_std[CONTINUOUS_FEATURES].values[:, None, :] - km_ciciot.cluster_centers_[None, :, :], axis=2).min(axis=1)
thresh_a = np.percentile(d_train_a, 90)
thresh_b = np.percentile(d_train_b, 90)

pred_a = cluster_predict(km_iot23, test_fixed_full[CONTINUOUS_FEATURES], thresh_a)
pred_b = cluster_predict(km_ciciot, test_fixed_full[CONTINUOUS_FEATURES], thresh_b)
agreement_km = (pred_a == pred_b).mean()
print(f"Agreement between IoT-23-trained and CICIoT2023-trained models: {agreement_km:.3f}")
