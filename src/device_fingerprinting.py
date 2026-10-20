"""
device_fingerprinting.py
==========================
Tests one of the project brief's original research questions that the
main cross-dataset study never addressed: "Can network behavior
uniquely fingerprint different IoT device classes?"

Scope and framing, stated explicitly:
  - This stays entirely WITHIN IoT-23 -- it does not touch CICIoT2023,
    so it is immune to the cross-dataset flow-construction
    incompatibility documented in honest_transfer_test.py.
  - We label each flow by which of your IoT-23 SCENARIO FILES it came
    from. We do not claim to know the specific real-world device model
    behind each scenario unless that is separately verified against
    IoT-23's own documentation -- "scenario source" is the honest,
    defensible label here, not "device model."
  - Only BENIGN flows are used, consistent with the research question
    being about NORMAL behavioral fingerprints, not attack behavior.

Method: a Random Forest multi-class classifier, trained to predict
which scenario file a benign flow came from, using the same feature
set as the rest of the project (duration, total_bytes, total_pkts,
avg_pkt_size, protocol flags). Reports accuracy, a confusion matrix,
and feature importances -- the latter doubles as evidence for a
SECOND unaddressed research question: "which traffic characteristics
provide the strongest signal?"
"""
import glob
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size",
            "is_tcp", "is_udp", "is_icmp"]


def load_iot23_with_scenario_label(path):
    """Same parsing logic as extract_features.py's load_iot23, but tags
    each row with its source filename instead of collapsing scenarios
    together."""
    with open(path) as f:
        header_line = None
        for line in f:
            if line.startswith("#fields"):
                header_line = re.split(r"\s+", line.strip())[1:]
                break
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=header_line,
                      na_values="-", engine="python")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["orig_bytes"] = pd.to_numeric(df["orig_bytes"], errors="coerce").fillna(0)
    df["resp_bytes"] = pd.to_numeric(df["resp_bytes"], errors="coerce").fillna(0)
    df["orig_pkts"] = pd.to_numeric(df["orig_pkts"], errors="coerce").fillna(0)
    df["resp_pkts"] = pd.to_numeric(df["resp_pkts"], errors="coerce").fillna(0)

    out = pd.DataFrame()
    out["duration"] = df["duration"]
    out["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
    out["total_pkts"] = df["orig_pkts"] + df["resp_pkts"]
    out["avg_pkt_size"] = (out["total_bytes"] / out["total_pkts"].replace(0, pd.NA)).fillna(0)
    out["is_tcp"] = (df["proto"].str.lower() == "tcp").astype(int)
    out["is_udp"] = (df["proto"].str.lower() == "udp").astype(int)
    out["is_icmp"] = (df["proto"].str.lower() == "icmp").astype(int)
    out["label"] = (~df["label"].str.lower().str.contains("benign", na=False)).astype(int)
    out["scenario"] = path.split("/")[-1]  # the honest, defensible label
    return out


if __name__ == "__main__":
    files = glob.glob("data/iot23/*.log.labeled")
    print(f"Found {len(files)} IoT-23 scenario files: {files}\n")

    frames = [load_iot23_with_scenario_label(f) for f in files]
    all_data = pd.concat(frames, ignore_index=True)

    benign = all_data[all_data.label == 0]
    print("Benign row count per scenario (classes with very few rows are unreliable):")
    print(benign["scenario"].value_counts())

    X = benign[FEATURES]
    y = benign["scenario"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"\nOverall scenario-classification accuracy: {acc:.3f}")
    print(f"(A naive baseline of always guessing the most common scenario would "
          f"score {y_test.value_counts(normalize=True).max():.3f})")

    print("\nConfusion matrix (rows=true scenario, columns=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print("\nPer-scenario precision/recall:")
    print(classification_report(y_test, pred, zero_division=0))

    print("Feature importances (which signals distinguish scenarios best):")
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importances)
