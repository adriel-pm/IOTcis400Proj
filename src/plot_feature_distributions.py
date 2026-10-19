"""
plot_feature_distributions.py
===============================
Empirically tests the paper's central mechanistic claim: that models
relying on the training distribution's geometric SHAPE (Isolation
Forest's partitions, K-Means' cluster centroids) fail to transfer
across environments even after per-feature mean/std standardization,
while StatBaseline's simple mean/std-based z-score succeeds.

If that claim is right, standardizing each dataset's benign traffic
to mean=0, std=1 should still leave visibly different DISTRIBUTION
SHAPES (skew, tails, modality) between IoT-23 and CICIoT2023 -- the
first two moments matching does not mean the distributions look alike.

This script:
  1. Standardizes each dataset's own benign traffic against itself
     (exactly as the modeling pipeline does).
  2. Plots overlaid histograms of both datasets' standardized benign
     distributions, one panel per continuous feature.
  3. Reports skewness and kurtosis for each -- if these differ
     substantially despite matching mean/std, that's direct,
     citable, quantitative evidence for the shape-mismatch claim.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

CONTINUOUS_FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size"]

iot23 = pd.read_csv("output/features_iot23.csv")
ciciot = pd.read_csv("output/features_ciciot2023.csv")

iot23_benign = iot23[iot23.label == 0]
ciciot_benign = ciciot[ciciot.label == 0]

# Standardize each dataset against its OWN benign mean/std -- identical
# to what standardize_by_reference() does in the real modeling pipeline.
def standardize_self(df):
    out = df.copy()
    for col in CONTINUOUS_FEATURES:
        mean = df[col].mean()
        std = df[col].std() if df[col].std() > 0 else 1e-6
        out[col] = (df[col] - mean) / std
    return out

iot23_std = standardize_self(iot23_benign)
ciciot_std = standardize_self(ciciot_benign)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

print(f"{'Feature':15s} {'IoT-23 skew':>12s} {'CICIoT skew':>12s} {'IoT-23 kurt':>12s} {'CICIoT kurt':>12s}")
for i, feat in enumerate(CONTINUOUS_FEATURES):
    ax = axes[i]
    # Clip extreme outliers purely for display range -- the printed
    # skew/kurtosis numbers below use the FULL unclipped data.
    lo, hi = -4, 4
    a = iot23_std[feat].clip(lo, hi)
    b = ciciot_std[feat].clip(lo, hi)
    ax.hist(a, bins=40, alpha=0.5, density=True, label="IoT-23 (benign)", color="#1f77b4")
    ax.hist(b, bins=40, alpha=0.5, density=True, label="CICIoT2023 (benign)", color="#ff7f0e")
    ax.set_title(feat)
    ax.set_xlabel("standardized value (z-score)")
    ax.legend(fontsize=8)

    s1, s2 = skew(iot23_std[feat]), skew(ciciot_std[feat])
    k1, k2 = kurtosis(iot23_std[feat]), kurtosis(ciciot_std[feat])
    print(f"{feat:15s} {s1:12.2f} {s2:12.2f} {k1:12.2f} {k2:12.2f}")

plt.tight_layout()
plt.savefig("output/feature_distribution_comparison.png", dpi=150)
print("\nSaved output/feature_distribution_comparison.png")
