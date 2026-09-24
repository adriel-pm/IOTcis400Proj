"""
per_attack_breakdown.py
========================
Goal: go beyond a single F1 number and answer "WHICH attack types does
our fingerprint actually catch, and which ones does it miss?"

This is the difference between reporting "cross-environment F1 = 0.38"
(true, but not very illuminating) and reporting "the model catches
DDoS-style flooding attacks even across environments, but misses
reconnaissance/port-scan traffic entirely" (an actual finding you can
explain and discuss in your paper).

How it works:
  1. Train each model on ONE dataset's benign traffic (same as before).
  2. Get predictions on the OTHER dataset's full traffic.
  3. Instead of collapsing all attacks into one "attack" bucket, group
     the results by the real attack name (detail_label) and compute
     what fraction of EACH attack type got flagged (i.e. per-attack
     recall). This is just a groupby -- no new ML concept, just a
     finer-grained slice of the same predictions.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

# Re-use the exact same feature list and StatBaseline class as before,
# so this script's numbers are directly comparable to models_and_eval.py.
from models_and_eval import FEATURES, StatBaseline


def per_attack_recall(df_train_benign, df_test, model, model_name):
    """
    Fits `model` on benign rows from df_train_benign, then predicts on
    every row of df_test, and reports the detection rate (recall)
    broken down by df_test's detail_label column.

    Returns a DataFrame with one row per attack type: how many flows
    of that type existed, and what fraction were correctly flagged.
    """
    model.fit(df_train_benign[FEATURES])

    if isinstance(model, StatBaseline):
        pred = model.predict(df_test[FEATURES])
    else:  # sklearn's IsolationForest uses -1/1, not 0/1 -- convert
        pred = (model.predict(df_test[FEATURES]) == -1).astype(int)

    result = df_test.copy()
    result["predicted_anomalous"] = pred

    # Only attack rows matter for "did we catch it" -- benign rows are
    # about false positives, which the main script already reports as FPR.
    attacks_only = result[result.label == 1]

    summary = attacks_only.groupby("detail_label").agg(
        n_flows=("predicted_anomalous", "size"),
        n_caught=("predicted_anomalous", "sum"),
    )
    summary["recall"] = (summary.n_caught / summary.n_flows).round(3)
    summary["model"] = model_name
    return summary.sort_values("recall")


def run_cross_environment_breakdown(train_df, test_df, train_name, test_name):
    print(f"\n=== Per-attack breakdown: trained on {train_name}, "
          f"tested on {test_name} ===")
    train_benign = train_df[train_df.label == 0]

    stat_summary = per_attack_recall(
        train_benign, test_df, StatBaseline(z_thresh=3.0), "StatBaseline")
    iso_summary = per_attack_recall(
        train_benign, test_df,
        IsolationForest(contamination=0.1, random_state=1), "IsolationForest")

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

    # Save everything to one CSV so you can drop it straight into a
    # table in your interim report / final paper.
    out = pd.concat(all_results)
    out.to_csv("output/per_attack_breakdown.csv")
    print("\nSaved output/per_attack_breakdown.csv")
