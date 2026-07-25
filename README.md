# Bank Account Fraud — Imbalanced Classification: SMOTE vs. GAN Oversampling

An end-to-end study of **class-imbalance handling** for fraud detection. The notebook
[`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) runs full EDA and preprocessing on the
**Bank Account Fraud (BAF)** dataset, then compares several strategies for the extreme
class imbalance — interpolation-based oversampling (**SMOTE**, **SMOTE-NC**), generative
oversampling (**CTGAN**), cost-sensitive learning (**class weights**), and
**decision-threshold tuning** — all evaluated with a `RandomForestClassifier`.

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
  (`SAMPLING_STRATEGY`), so the comparison isolates *how* synthetic samples are made, not
  *how many*.
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

## Summary of findings

Representative results (single stratified split; exact numbers vary by run and epoch count):

- **No oversampling method meaningfully beat the baseline on PR-AUC.** SMOTE was
  marginally best; SMOTE-NC and CTGAN were level with or slightly below baseline.
- **CTGAN did not earn its complexity here.** Despite good fidelity, its synthetic fraud
  was systematically *milder* than real fraud (feature means shifted toward the
  legitimate class), which nudged PR-AUC down rather than up.
- **The real bottleneck is the decision threshold, not a shortage of synthetic fraud.**
  Every model ranked fraud far better than chance yet had near-zero recall at the default
  0.5 cutoff. **Threshold tuning** (and `class_weight="balanced"`) converts that good
  ranking into usable recall at **zero data-fabrication cost** — the practical
  recommendation.
- Differences of a few tenths of a percent are within noise on a single split; confirm
  with repeated stratified splits before drawing firm conclusions.

> Takeaway: on a well-suited dataset (large minority, genuine mixed types, verified
> generator fidelity), interpolation-based and generative oversampling still failed to
> beat a plain baseline's *ranking*, and cost-sensitive learning plus threshold tuning
> was the most effective, cheapest intervention.

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
