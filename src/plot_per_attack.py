"""
plot_per_attack.py
===================
Turns per_attack_breakdown.csv into one horizontal bar chart per
(direction, model) combination, sorted worst-to-best recall. A sorted
bar chart tells the "which attacks transfer, which get missed" story
in one glance -- much better for a paper than a raw CSV table.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/per_attack_breakdown.csv")
df = df.rename(columns={df.columns[0]: "detail_label"})

for (direction, model_name), group in df.groupby(["direction", "model"]):
    group = group.sort_values("recall")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(group))))
    ax.barh(group["detail_label"], group["recall"], color="#1f77b4")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"{direction}  —  {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    safe_name = direction.replace(" -> ", "_to_").replace(" ", "")
    out_path = f"output/per_attack_{safe_name}_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")