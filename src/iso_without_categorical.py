import pandas as pd
from sklearn.ensemble import IsolationForest
from models_and_eval import CONTINUOUS_FEATURES, evaluate, balanced_test_set

def run_iso_continuous_only(train_df, test_df, train_name, test_name):
    print(f"\n=== IsolationForest WITHOUT categorical features: {train_name} -> {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_df = balanced_test_set(test_benign, test_attacks, ratio=1.0)

    mean = train_benign[CONTINUOUS_FEATURES].mean()
    std = train_benign[CONTINUOUS_FEATURES].std().replace(0, 1e-6)
    train_std = train_benign.copy()
    train_std[CONTINUOUS_FEATURES] = (train_benign[CONTINUOUS_FEATURES] - mean) / std

    test_mean = test_benign[CONTINUOUS_FEATURES].mean()
    test_std_dev = test_benign[CONTINUOUS_FEATURES].std().replace(0, 1e-6)
    test_std = test_df.copy()
    test_std[CONTINUOUS_FEATURES] = (test_df[CONTINUOUS_FEATURES] - test_mean) / test_std_dev

    iso = IsolationForest(contamination=0.1, random_state=1).fit(train_std[CONTINUOUS_FEATURES])
    pred = (iso.predict(test_std[CONTINUOUS_FEATURES]) == -1).astype(int)
    evaluate(test_df.label.values, pred, "IsolationForest (continuous only)")

if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")
    run_iso_continuous_only(iot23, ciciot, "IoT-23", "CICIoT2023")
    run_iso_continuous_only(ciciot, iot23, "CICIoT2023", "IoT-23")