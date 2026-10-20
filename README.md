# Cross-Environment Behavioral Fingerprinting for IoT Device Security
### A Cautionary Tale in Evaluation Methodology

**Authors:** Adriel Pacheco, Mark Vu
**Course:** CIS 400/600 — IoT Security and Privacy, Fall 2026, Syracuse University

[Full Paper (PDF)](./paper/Final_Paper_v2.pdf) · [Presentation Slides (PDF)](./slides/Presentation_v2.pdf)

---

## Overview

This project set out to answer a simple question: **does a behavioral network fingerprint for IoT devices, trained to detect anomalies on one network, actually generalize to a different network?** We built an anomaly-detection pipeline spanning two independent public IoT traffic datasets (IoT-23, CICIoT2023) and three modeling approaches (a statistical z-score baseline, K-Means clustering, and Isolation Forest).

Our first set of results appeared to answer this question affirmatively — a simple baseline seemed to transfer almost perfectly (F1 = 0.974) across environments. **That result was wrong, and finding out why became the actual contribution of this project.**

We traced the apparent success to a flaw in our own evaluation protocol: standardizing test data against its own benign reference caused a threshold-based model's parameters to collapse to a fixed, training-data-independent point, making the "transfer" result a measurement of the test set's own internal structure rather than anything learned from training. Correcting this, we found that **none of the three models achieve genuine cross-environment transfer**, and root-caused this to an even more fundamental problem: one dataset's "packet count" feature turned out to be a fixed-window aggregation artifact of its own publication methodology, not an actual measurement of flow length — making it incomparable to the other dataset's native packet counts regardless of how it's scaled or standardized.

We verified this finding at multiple levels (aggregate, per-attack-type, and via an independent model-degeneracy check distinguishing which of the three models' failures were genuine versus artifacts of our own protocol), and report the full investigation — including the parts where our initial conclusion was wrong — as the paper's central contribution.

## Key Findings

1. **An apparent positive result (StatBaseline, F1 = 0.974 cross-environment) was fully explained as an evaluation-protocol artifact**, verified mathematically: standardizing a dataset against its own statistics always yields mean=0, std=1, regardless of the dataset's actual content.
2. **Once corrected, no model achieves genuine cross-environment transfer** between the two datasets, under any of four progressively narrower feature subsets tested.
3. **Root cause:** CICIoT2023's packet-count feature is a fixed 10/100-packet aggregation window per the dataset's own published methodology — not a measurement of true flow length — making it fundamentally incomparable to IoT-23's native per-connection packet counts, independent of scale.
4. **The evaluation artifact affects models unevenly.** A follow-up diagnostic confirmed Isolation Forest's cross-environment failure is genuine and training-source-dependent (64.4% prediction disagreement between differently-trained instances), while K-Means clustering is substantially — though not fully — subject to the same degeneracy that fully explained the statistical baseline's false success (94.8% agreement regardless of training source).
5. **A complementary, methodologically independent result:** within a single dataset, IoT-23's five capture scenarios are genuinely distinguishable by behavioral features alone (macro F1 = 0.88 against a 0.20 chance baseline), driven overwhelmingly by connection duration (59.7% feature importance) rather than protocol type.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── paper/
│   └── Final_Paper_v2.pdf         # Full written report
├── slides/
│   └── Presentation_v2.pdf        # Presentation deck
├── src/
│   ├── extract_features.py        # Maps both datasets to a common feature schema
│   ├── models_and_eval.py         # StatBaseline, ClusterBaseline, Isolation Forest + evaluation harness
│   ├── per_attack_breakdown.py    # Per-attack-type recall, both self-calibrated and honest versions
│   ├── plot_per_attack.py         # Per-attack recall visualizations
│   ├── visualize_results.py       # Aggregate F1 comparison chart
│   ├── honest_transfer_test.py    # The core methodology-correction diagnostic
│   ├── diagnose_outliers.py       # Rules out data corruption as an explanation for extreme kurtosis
│   ├── check_total_pkts_variance.py   # Confirms CICIoT2023's near-constant packet-count artifact
│   ├── verify_isoforest_not_degenerate.py  # Model-by-model degeneracy verification
│   ├── device_fingerprinting.py   # Within-dataset scenario-fingerprinting experiment
│   ├── cluster_experiment.py      # K-Means-based anomaly detector
│   ├── iso_with_categorical.py / iso_without_categorical.py  # Categorical-feature ablation
│   └── plot_feature_distributions.py  # Cross-dataset distribution shape comparison
└── data/                           # NOT included — see Data Access below
```

## Data Access

Raw dataset files are not included in this repository (several hundred MB–GB in size; also not ours to redistribute). To reproduce:

- **IoT-23**: download individual scenario folders from [stratosphereips.org/datasets-iot23](https://www.stratosphereips.org/datasets-iot23). This project uses 5 scenarios (2 benign honeypot captures, 3 malware captures) — any subset works, provided file names are updated in `extract_features.py`.
- **CICIoT2023**: download the `MERGED_CSV` files from the [official UNB CIC dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html). This project uses 8 of the merged CSVs (`Merged01.csv`–`Merged08.csv`).

Place IoT-23 files in `data/iot23/` and CICIoT2023 files in `data/ciciot2023/` before running the pipeline.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Reproducing the Results

Run scripts from the repository root, in this order:

```bash
# 1. Build the common feature schema from both datasets
python3 src/extract_features.py

# 2. Core evaluation: within-environment and cross-environment, all three models
python3 src/models_and_eval.py

# 3. Per-attack-type breakdown (self-calibrated AND honest versions)
python3 src/per_attack_breakdown.py
python3 src/plot_per_attack.py

# 4. The methodology-correction investigation (this is the paper's central finding)
python3 src/honest_transfer_test.py
python3 src/diagnose_outliers.py
python3 src/check_total_pkts_variance.py
python3 src/verify_isoforest_not_degenerate.py

# 5. Supplementary experiments
python3 src/device_fingerprinting.py
python3 src/cluster_experiment.py
python3 src/plot_feature_distributions.py
```

Outputs (tables, figures, CSVs) are written to `output/`.

## Example Output

Real output from `honest_transfer_test.py`, the core methodology-correction diagnostic, run on the actual datasets:

```
##### (c) Excluding duration + total_pkts (the two incompatible features) #####

=== HONEST transfer test [2-feature]: IoT-23 -> CICIoT2023 ===
  [2-feature] precision=0.500 recall=1.000 f1=0.667 FPR=1.000  (n=266784, attacks=133392)

=== HONEST transfer test [2-feature]: CICIoT2023 -> IoT-23 ===
  [2-feature] precision=0.000 recall=0.000 f1=0.000 FPR=0.000  (n=955636, attacks=477818)
```

## Summary of Results

| Scenario | Model | F1 | Notes |
|---|---|---|---|
| Within-environment, IoT-23 | StatBaseline | 0.017 | Valid, unaffected by the methodology issue below |
| Within-environment, CICIoT2023 | StatBaseline | 0.974 | Valid |
| Cross-environment (honest protocol) | All 3 models, all feature subsets | Degenerate (flags all or none) | No genuine transfer; see paper Section 5 |
| Within-dataset scenario fingerprinting | Random Forest | 0.88 (macro F1) | Methodologically independent result, unaffected by the above |

Full results, including the complete diagnostic trail, are in the [paper](./paper/Final_Paper_v2.pdf).

## Datasets Used

- S. Garcia, A. Parmisano, and M. J. Erquiaga, "IoT-23: A labeled dataset with malicious and benign IoT network traffic," Stratosphere Laboratory, CTU University, 2020.
- E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, p. 5941, 2023.

Full reference list, including the dataset-shift and transfer-learning literature this work is situated within, is in the paper.

## Limitations

Stated in full in the paper's Limitations section; briefly: the within-dataset scenario-fingerprinting result does not establish cross-dataset device identification; model hyperparameters were fixed rather than tuned; and the comparison does not extend to autoencoder-based anomaly detection. These are documented as explicit scope boundaries rather than oversights.

## License

This code is released under the MIT License (see [LICENSE](./LICENSE)). The datasets themselves are governed by their respective providers' own terms (see links above).

## Acknowledgments

Built for CIS 400/600, IoT Security and Privacy, Syracuse University, Fall 2026, under the guidance of Prof. Priyantha Kumarawadu.
