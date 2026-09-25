"""
visualize_results.py
=====================
Produces ONE figure that will probably end up as the most-referenced
image in both your paper and your presentation slides: a grouped bar
chart comparing F1 score within-environment (the "easy" case: train
and test on the same dataset) against cross-environment (train on one,
test on the other -- the actual research question).

Design choice explained: we recompute the four scenarios here rather
than importing saved numbers, so this script is self-contained and
always reflects your current code/data -- if you tweak a model
parameter later, just re-run this and the figure updates automatically.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from models_and_eval import FEATURES, StatBaseline, balanced_test_set, standardize_by_reference

def get_f1(train_df, test_df, model):
    """Fits `model` on train_df's benign rows, evaluates on test_df,
    returns just the F1 score (the other metrics are in results.txt
    already -- this script is deliberately narrow, one number per bar)."""
    train_benign = train_df[train_df.label == 0]
    test_benign = test_df[test_df.label == 0]
    test_attacks = test_df[test_df.label == 1]
    test_df = balanced_test_set(test_benign, test_attacks, ratio=1.0)
    test_df = standardize_by_reference(train_benign, test_df)
    train_benign = standardize_by_reference(train_benign, train_benign)
    model.fit(train_benign[FEATURES])
    if isinstance(model, StatBaseline):
        pred = model.predict(test_df[FEATURES])
    else:
        pred = (model.predict(test_df[FEATURES]) == -1).astype(int)
    from sklearn.metrics import f1_score
    return f1_score(test_df.label.values, pred, zero_division=0)


if __name__ == "__main__":
    iot23 = pd.read_csv("output/features_iot23.csv")
    ciciot = pd.read_csv("output/features_ciciot2023.csv")

    # ---- Within-environment: hold out 30% of each dataset's benign
    # traffic plus all its attacks, same split logic as models_and_eval.py
    _, iot23_test = train_test_split(
        iot23[iot23.label == 0], test_size=0.3, random_state=1)
    iot23_within_test = pd.concat([iot23_test, iot23[iot23.label == 1]])

    _, ciciot_test = train_test_split(
        ciciot[ciciot.label == 0], test_size=0.3, random_state=1)
    ciciot_within_test = pd.concat([ciciot_test, ciciot[ciciot.label == 1]])

    scenarios = {
        "IoT-23\n(within)":       (iot23, iot23_within_test),
        "CICIoT2023\n(within)":   (ciciot, ciciot_within_test),
        "IoT-23 -> CICIoT2023\n(cross)": (iot23, ciciot),
        "CICIoT2023 -> IoT-23\n(cross)": (ciciot, iot23),
    }

    f1_stat, f1_iso = [], []
    labels = []
    for name, (train_df, test_df) in scenarios.items():
        labels.append(name)
        f1_stat.append(get_f1(train_df, test_df, StatBaseline(z_thresh=3.0)))
        f1_iso.append(get_f1(train_df, test_df,
                              IsolationForest(contamination=0.1, random_state=1)))

    # ---- Plot: one grouped bar chart, two bars (StatBaseline, IsoForest)
    # per scenario. This is plain matplotlib -- no styling library needed.
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width/2 for i in x], f1_stat, width, label="StatBaseline")
    ax.bar([i + width/2 for i in x], f1_iso, width, label="IsolationForest")

    ax.set_ylabel("F1 score")
    ax.set_title("Within-environment vs cross-environment detection (F1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/figure_transfer_comparison.png", dpi=150)
    print("Saved output/figure_transfer_comparison.png")
    print("\nRaw values used in the chart:")
    for name, s, i in zip(labels, f1_stat, f1_iso):
        print(f"  {name.replace(chr(10), ' '):35s} StatBaseline={s:.3f}  IsolationForest={i:.3f}")
