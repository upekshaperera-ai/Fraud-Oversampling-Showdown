# Bank Account Fraud — Imbalanced Classification: SMOTE vs. GAN Oversampling

An end-to-end study of **class-imbalance handling** for fraud detection. The notebook
[`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) runs full EDA and preprocessing on the
**Bank Account Fraud (BAF)** dataset, then compares several strategies for the extreme
class imbalance — interpolation-based oversampling (**SMOTE**, **SMOTE-NC**), generative
oversampling (**CTGAN**), cost-sensitive learning (**class weights**), and
**decision-threshold tuning** — all evaluated with a `RandomForestClassifier`.

**Headline result:** none of the oversampling methods beat the plain baseline's ranking
(PR-AUC 0.127), and CTGAN was ~32% worse. Simply moving the decision threshold from 0.5 to
0.13 on the untouched baseline took F1 from **0.001 → 0.212** and frauds caught from
**1 → 506** out of 2,206 — the same model, read at a sane cutoff.
[Full results ↓](#results)

---

## The dataset

**Bank Account Fraud (BAF)**, *Base* variant — a large-scale, realistic tabular dataset
of online bank-account-opening applications, published at **NeurIPS 2022** (Datasets and
Benchmarks Track). It was generated with a CTGAN model trained on a real-world
anonymised fraud dataset, then privacy-enhanced and validated to preserve the statistical
properties and predictive utility of the original.

| Property | Value |
|---|---|
| Rows | 1,000,000 |
| Columns | 32 (31 features + target) |
| Target | `fraud_bool` (1 = fraudulent application) |
| Fraud rate | 11,029 / 1,000,000 = **1.10%** (≈ **1:90**) |
| Feature types | 19 numeric, 7 binary flags, 5 categorical |
| Missing values | encoded as **`-1` sentinels**, not `NaN` |
| Duplicates | none |
| Time span | 8 months (`month` column, 0–7) |

### Data files

The `data/` folder contains the six official BAF variants. This project uses **`Base.csv`**.


### Key characteristics that drive the code

- **Sentinel-encoded missingness.** Several columns use `-1` to mean *"not available"*
  rather than `NaN`. A plain `df.isna()` reports the dataset as complete, which is
  misleading. The notebook detects sentinels data-driven (a column is sentinel-encoded if
  it contains `-1` **and** has no values below `-1`), which correctly spares
  `intended_balcon_amount` — a feature with genuinely negative values (min ≈ −15.5).
  `prev_address_months_count` is ~71% missing and is dropped; `bank_months_count`
  (~25%) is kept and imputed.
- **Mixed types.** Five genuine categorical columns make this a natural test bed for
  SMOTE-NC (which handles categoricals correctly) and CTGAN (designed for mixed tabular
  data) — unlike all-numeric fraud datasets where those methods' advantages are idle.
- **Discrete "numeric" columns.** `customer_age` (decades only: 10, 20, … 90), `income`
  (0.1 steps), and `proposed_credit_limit` (fixed tiers) are numeric in dtype but
  genuinely discrete; the notebook declares these to CTGAN so it does not emit impossible
  values like `customer_age = 35`.

### Feature groups

| Group | Columns |
|---|---|
| **Target** | `fraud_bool` |
| **Categorical (5)** | `payment_type`, `employment_status`, `housing_status`, `source`, `device_os` |
| **Binary flags (7)** | `email_is_free`, `phone_home_valid`, `phone_mobile_valid`, `has_other_cards`, `foreign_request`, `keep_alive_session`, `device_fraud_count` |
| **Numeric (19)** | `income`, `name_email_similarity`, `prev_address_months_count`, `current_address_months_count`, `customer_age`, `days_since_request`, `intended_balcon_amount`, `zip_count_4w`, `velocity_6h`, `velocity_24h`, `velocity_4w`, `bank_branch_count_8w`, `date_of_birth_distinct_emails_4w`, `credit_risk_score`, `bank_months_count`, `proposed_credit_limit`, `session_length_in_minutes`, `device_distinct_emails_8w`, `month` |

---

## Notebook structure

| Section | Content |
|---|---|
| 1–2 | Dataset overview and variable typing |
| 3 | Class distribution / imbalance |
| 4 | Missing values (`-1` sentinel decoding) and duplicates |
| 5–7 | Feature distributions, outlier detection, correlation matrix |
| 8 | Preprocessing — sentinel decoding, drop high-missingness columns, stratified split, impute + `RobustScaler` + one-hot (all fit on train only) |
| 9 | **Baseline** Random Forest on the imbalanced data |
| 10 | **SMOTE** (naive — plain interpolation on the one-hot matrix) |
| 11 | **SMOTE-NC** (fair baseline — handles categoricals correctly) |
| 12 | **CTGAN** — generative oversampling trained on the fraud minority |
| 13 | Four-way comparison |
| 14 | **Class weights** (`class_weight="balanced"`) and **threshold tuning** — the practical alternative |

### Methodology notes

- **No data leakage.** All imputers, scalers, encoders, resamplers, and the decision
  threshold are fit on training data only; the test set is touched exactly once, at
  evaluation.
- **Fair comparison.** Every resampler targets the *same* minority ratio
  (`SAMPLING_STRATEGY = 0.10`, i.e. 1:10 after resampling — not full balance), so the
  comparison isolates *how* synthetic samples are made, not *how many*.
- **Thresholds tuned on validation, never on test.** Section 14 carves a 25% validation
  split out of *train* to pick the cutoff, so those models are fit on 600k rows rather than
  800k. That is why the baseline's PR-AUC differs slightly between sections (0.1273 in
  Section 9, 0.1268 in Section 14) — less training data, not a different method.
- **Honest metrics.** At 1:90 imbalance, **accuracy is meaningless** and ROC-AUC is
  flattered by the large true-negative count. **PR-AUC** is treated as the headline
  metric, alongside precision, recall, and F1.
- **Generative diagnostics.** CTGAN includes a training-loss convergence plot and
  real-vs-synthetic fidelity checks (numeric distributions, categorical proportions with
  total-variation distance, and per-feature mean-shift / mode-collapse tables) — because
  a converged loss curve does **not** guarantee usable samples.

---

## Setup

Requires Python 3.9+.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn ctgan
```

`ctgan` pulls in PyTorch. Training runs on CPU by default; a CUDA GPU speeds up the
CTGAN section considerably.

## Running

1. Ensure `data/Base.csv` is present (extract from the BAF release if needed).
2. Open [`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) and run all cells
   top to bottom. The notebook reads `../data/Base.csv`, so it works with the notebook's own
   folder as the working directory (the Jupyter/VS Code default).

**Runtime:** roughly one hour on CPU, dominated by CTGAN training (`CTGAN_EPOCHS = 300`,
~40 min) and several Random Forest fits on ~800k rows. Lower `CTGAN_EPOCHS` for a faster
pass; Section 14 does not depend on CTGAN.

---

## Results

Test set: **200,000** applications, **2,206** fraudulent (single stratified split,
`random_state=42`; exact numbers vary by run and epoch count).

| Approach | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline RF @ 0.5 | 0.333 | 0.001 | 0.001 | 0.832 | 0.127 |
| SMOTE RF | 0.475 | 0.017 | 0.033 | 0.850 | **0.129** |
| SMOTE-NC RF | 0.252 | 0.075 | 0.116 | **0.859** | 0.111 |
| CTGAN RF | 0.237 | 0.031 | 0.055 | 0.835 | 0.087 |
| Balanced RF @ 0.5 | 0.800 | 0.002 | 0.004 | 0.824 | 0.112 |
| **RF @ best-F1 threshold (0.13)** | 0.197 | 0.229 | **0.212** | 0.830 | 0.127 |
| RF @ recall≥50% threshold (0.05) | 0.080 | **0.532** | 0.139 | 0.830 | 0.127 |
| Balanced RF @ best-F1 threshold | 0.156 | 0.259 | 0.195 | 0.828 | 0.107 |

Accuracy is omitted deliberately — every row scores 0.93–0.99 and none of it is
informative at 1:90.

### The same table, in operational terms

| Approach | Frauds caught | Frauds missed | False alarms |
|---|---|---|---|
| Baseline RF @ 0.5 | 1 | 2,205 | 2 |
| SMOTE RF | 38 | 2,168 | 42 |
| SMOTE-NC RF | 166 | 2,040 | 494 |
| CTGAN RF | 68 | 2,138 | 219 |
| Balanced RF @ 0.5 | 4 | 2,202 | 1 |
| **RF @ best-F1 threshold** | **506** | 1,700 | 2,066 |
| RF @ recall≥50% threshold | **1,173** | 1,033 | 13,502 |
| Balanced RF @ best-F1 threshold | 571 | 1,635 | 3,088 |

## Summary of findings

- **No oversampling method beat the baseline on PR-AUC.** SMOTE was nominally best
  (0.129 vs 0.127), a gap well inside single-split noise. SMOTE-NC (0.111) landed ~13%
  *below* baseline and CTGAN (0.087) ~32% below. SMOTE-NC did buy a genuinely better
  ranking by ROC-AUC (0.859, best overall) and the best F1 of any oversampler (0.116) —
  but not better precision-recall trade-off where it counts.
- **CTGAN did not earn its complexity.** It is the worst configuration tested on PR-AUC,
  after ~40 minutes of training, despite passing its fidelity checks. Its synthetic fraud
  was systematically *milder* than real fraud (feature means shifted toward the legitimate
  class), so it blurred the decision boundary instead of sharpening it.
- **The bottleneck is the decision threshold, not a shortage of synthetic fraud.** The
  baseline ranked fraud far better than chance (ROC-AUC 0.83) yet caught **1 fraud out of
  2,206** at the default 0.5 cutoff. Moving the cutoff to 0.13 takes the *same model* to
  506 frauds caught — F1 0.001 → 0.212, a ~235× improvement at zero data-fabrication cost.
- **Threshold tuning does not improve the model; it makes an already-adequate model
  usable.** All three threshold rows share an identical PR-AUC of 0.127 because they are
  one model with one ranking, read at three cutoffs. This is the central result: the
  oversampling methods were competing to fix a problem that was never in the ranking.
- **Class weights alone were not the answer.** `class_weight="balanced"` *lowered* PR-AUC
  (0.112 vs 0.127) and still caught only 4 frauds at the default cutoff. Combined with
  threshold tuning it reached F1 0.195 — respectable, but still short of plain RF +
  threshold tuning (0.212). It trades precision for recall (571 caught vs 506, at 3,088
  false alarms vs 2,066), which is a reasonable choice operationally but not a free win.
- Sub-percent gaps are within noise on a single split; confirm with repeated stratified
  splits before drawing firm conclusions. The threshold effect is two orders of magnitude
  and does not need that caveat.

> **Takeaway:** on a dataset well suited to it (large minority, genuine mixed types,
> verified generator fidelity), both interpolation-based and generative oversampling
> failed to beat a plain baseline's *ranking* — and generative oversampling actively hurt.
> Tuning the decision threshold on the untouched baseline was the most effective and by
> far the cheapest intervention. Pick the operating point from the business cost of a
> missed fraud versus a manual review: threshold 0.13 for balance, 0.05 to catch half of
> all fraud at ~13.5k reviews per 200k applications.

---

## Citation

If you use the BAF dataset, please cite:

> Sérgio Jesus, José Pombal, Duarte Alves, André F. Cruz, Pedro Saleiro, Rita P. Ribeiro,
> João Gama, and Pedro Bizarro. **"Turning the Tables: Biased, Imbalanced, Dynamic Tabular
> Datasets for ML Evaluation."** *Advances in Neural Information Processing Systems (NeurIPS)*,
> Datasets and Benchmarks Track, 2022.

```bibtex
@article{jesusTurningTablesBiased2022,
  title   = {Turning the {{Tables}}: {{Biased}}, {{Imbalanced}}, {{Dynamic Tabular Datasets}} for {{ML Evaluation}}},
  author  = {Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} F. and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {35},
  year    = {2022}
}
```

- **Source / download:** [github.com/feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud)
  · also mirrored on [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- **License:** CC BY-NC-SA 4.0 (non-commercial, share-alike)

### Methods referenced

- **SMOTE / SMOTE-NC** — Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.
  (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
- **CTGAN** — Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
  *Modeling Tabular Data using Conditional GAN.* NeurIPS 2019.

---

*This project is for research and educational purposes. The BAF dataset is licensed for
non-commercial use.*
