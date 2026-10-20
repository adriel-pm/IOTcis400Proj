"""
diagnose_outliers.py
=====================
Before concluding that IoT-23 and CICIoT2023's benign traffic
distributions are fundamentally incompatible (honest_transfer_test.py),
check a simpler, cheaper explanation first: is the extreme kurtosis
we measured (e.g. 218,957 for IoT-23 duration) driven by the WHOLE
distribution, or by a tiny number of pathological/corrupted rows?

Two checks:
  1. Print the top 10 most extreme raw values for each continuous
     feature, with enough row context to judge whether they look like
     real traffic or a parsing/data artifact (e.g. a duration of 0
     paired with an absurd packet count).
  2. Recompute kurtosis after trimming the top/bottom 0.1%, 1%, and 5%
     of each feature, to quantify how much of the extreme shape comes
     from a thin tail versus being pervasive throughout the data.

If trimming 0.1% of rows collapses kurtosis from thousands down to
single digits, that's a data-quality issue worth handling explicitly
(e.g. a documented, justified outlier cap) rather than a fundamental
cross-dataset incompatibility. If kurtosis stays huge even after
trimming 5%, the mismatch is real and pervasive, not an artifact of
a few bad rows.
"""
import pandas as pd
from scipy.stats import kurtosis

CONTINUOUS_FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size"]

iot23 = pd.read_csv("output/features_iot23.csv")
ciciot = pd.read_csv("output/features_ciciot2023.csv")

for name, df in [("IoT-23", iot23), ("CICIoT2023", ciciot)]:
    benign = df[df.label == 0]
    print(f"\n{'='*70}\n{name} benign traffic (n={len(benign)})\n{'='*70}")

    for feat in CONTINUOUS_FEATURES:
        print(f"\n--- {feat} ---")
        top10 = benign.nlargest(10, feat)[CONTINUOUS_FEATURES]
        print("Top 10 most extreme rows (all continuous features shown for context):")
        print(top10.to_string())

        full_kurt = kurtosis(benign[feat])
        print(f"\nKurtosis at different trim levels:")
        print(f"  0% trimmed (full data):  {full_kurt:.2f}")
        for pct in [0.1, 1.0, 5.0]:
            lo = benign[feat].quantile(pct / 200)
            hi = benign[feat].quantile(1 - pct / 200)
            trimmed = benign[feat][(benign[feat] >= lo) & (benign[feat] <= hi)]
            print(f"  {pct}% trimmed (each tail): {kurtosis(trimmed):.2f}  "
                  f"(n={len(trimmed)} of {len(benign)})")
