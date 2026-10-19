import pandas as pd
from sklearn.ensemble import IsolationForest
from models_and_eval import standardize_by_reference, evaluate, balanced_test_set

# Use ALL features, including protocol flags -- this is the whole point
FULL_FEATURES = ["duration", "total_bytes", "total_pkts", "avg_pkt_size",
                  "is_tcp", "is_udp", "is_icmp"]

def run_iso_full_features(train_df, test_df, train_name, test_name):
    print(f"\n=== IsolationForest WITH categorical features: {train_name} -> {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_df = balanced_test_set(test_benign, test_attacks, ratio=1.0)

    # Standardize manually here since we're using FULL_FEATURES, not the
    # models_and_eval FEATURES constant each function assumes
    mean = train_benign[FULL_FEATURES].mean()
    std = train_benign[FULL_FEATURES].std().replace(0, 1e-6)
    train_std = train_benign.copy()
    train_std[FULL_FEATURES] = (train_benign[FULL_FEATURES] - mean) / std
    test_std = test_df.copy()
    test_mean = test_benign[FULL_FEATURES].mean()
    test_std_dev = test_benign[FULL_FEATURES].std().replace(0, 1e-6)
    test_std[FULL_FEATURES] = (test_df[FULL_FEATURES] - test_mean) / test_std_dev

    iso = IsolationForest(contamination=0.1, random_state=1).fit(train_std[FULL_FEATURES])
    pred = (iso.predict(test_std[FULL_FEATURES]) == -1).astype(int)
    evaluate(test_df.label.values, pred, "IsolationForest (with categorical)")

if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")
    run_iso_full_features(iot23, ciciot, "IoT-23", "CICIoT2023")
    run_iso_full_features(ciciot, iot23, "CICIoT2023", "IoT-23")