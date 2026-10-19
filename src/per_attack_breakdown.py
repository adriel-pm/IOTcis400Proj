import pandas as pd
from sklearn.ensemble import IsolationForest

from models_and_eval import FEATURES, StatBaseline, standardize_by_reference


def per_attack_recall(df_train_benign, df_test, model, model_name, direction):
    model.fit(df_train_benign[FEATURES])

    if isinstance(model, StatBaseline):
        pred = model.predict(df_test[FEATURES])
    else:
        pred = (model.predict(df_test[FEATURES]) == -1).astype(int)

    result = df_test.copy()
    result["predicted_anomalous"] = pred
    attacks_only = result[result.label == 1]

    summary = attacks_only.groupby("detail_label").agg(
        n_flows=("predicted_anomalous", "size"),
        n_caught=("predicted_anomalous", "sum"),
    )
    summary["recall"] = (summary.n_caught / summary.n_flows).round(3)
    summary["model"] = model_name
    summary["direction"] = direction
    return summary.sort_values("recall")


def run_cross_environment_breakdown(train_df, test_df, train_name, test_name):
    direction = f"{train_name} -> {test_name}"
    print(f"\n=== Per-attack breakdown: trained on {train_name}, "
          f"tested on {test_name} ===")
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]

    # Standardize exactly as models_and_eval.py's corrected cross-environment
    # logic does: the test side against ITS OWN benign reference, the train
    # side against itself. Skipping this was the bug in the earlier run --
    # raw, unstandardized features on two very different scales made both
    # models flag almost everything (or nothing) regardless of real content.
    test_df_std = standardize_by_reference(test_benign, test_df)
    train_benign_std = standardize_by_reference(train_benign, train_benign)

    stat_summary = per_attack_recall(
        train_benign_std, test_df_std, StatBaseline(z_thresh=3.0), "StatBaseline", direction)
    iso_summary = per_attack_recall(
        train_benign_std, test_df_std,
        IsolationForest(contamination=0.1, random_state=1), "IsolationForest", direction)

    combined = pd.concat([stat_summary, iso_summary])
    print(combined.to_string())
    return combined


if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")

    all_results = []
    all_results.append(
        run_cross_environment_breakdown(iot23, ciciot, "IoT-23", "CICIoT2023"))
    all_results.append(
        run_cross_environment_breakdown(ciciot, iot23, "CICIoT2023", "IoT-23"))

    out = pd.concat(all_results)
    out.to_csv("output/per_attack_breakdown.csv")
    print("\nSaved output/per_attack_breakdown.csv")