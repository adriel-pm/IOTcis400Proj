"""
check_total_pkts_variance.py
==============================
diagnose_outliers.py's output suggests CICIoT2023's benign total_pkts
might be almost perfectly constant (every one of its "top 10 most
extreme" rows showed exactly 10, and kurtosis became numerically
unstable under trimming -- a classic "nearly identical values" signal).

This directly confirms or refutes that, the same way we originally
confirmed is_icmp's zero variance: describe() and value_counts().
"""
import pandas as pd

iot23 = pd.read_csv("output/features_iot23.csv")
ciciot = pd.read_csv("output/features_ciciot2023.csv")

for name, df in [("IoT-23", iot23), ("CICIoT2023", ciciot)]:
    benign = df[df.label == 0]
    print(f"\n{'='*60}\n{name} benign total_pkts\n{'='*60}")
    print(benign["total_pkts"].describe())
    print("\nTop 10 most common values:")
    print(benign["total_pkts"].value_counts().head(10))
    print(f"\nStandard deviation: {benign['total_pkts'].std():.6f}")
    print(f"% of rows equal to the single most common value: "
          f"{(benign['total_pkts'] == benign['total_pkts'].mode()[0]).mean()*100:.2f}%")
