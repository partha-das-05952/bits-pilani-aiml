# Credit Risk Classification — Statlog (German Credit Data)

Five classification models compared on the UCI German Credit dataset, with an
interactive Streamlit front end for scoring uploaded test data.

**Live app:** `https://bits-pilani-aiml-partha-das-01.streamlit.app/`
**Repository:** `https://github.com/partha-das-05952/bits-pilani-aiml`

---

## a. Problem Statement

Banks and NBFCs must decide whether a loan applicant is likely to repay before
sanctioning credit. Approving an applicant who later defaults causes a direct
financial loss; rejecting a creditworthy applicant means lost business. The two
errors are not equally expensive.

This project treats the decision as a **binary classification** problem: given
20 attributes describing an applicant — account history, loan purpose and
amount, employment, age, existing obligations and so on — predict whether the
applicant is a **good credit risk (0)** or a **bad credit risk (1)**.

The positive class is deliberately set to **bad credit**, because the costly
mistake in lending is failing to identify a likely defaulter. Recall on class 1
therefore matters more than raw accuracy. The UCI documentation makes this
explicit by supplying a cost matrix: misclassifying a bad customer as good
costs **5 units**, while misclassifying a good customer as bad costs **1 unit**.
That cost is reported alongside the six required metrics.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Statlog (German Credit Data) |
| Source | UCI Machine Learning Repository, Dataset ID 144 (Hofmann, 1994) |
| Licence | CC BY 4.0 |
| Instances | 1,000 |
| Features | 20 (13 categorical, 7 numerical) |
| Target | `credit_risk` — 0 = Good credit (700), 1 = Bad credit (300) |
| Missing values | None |
| Task type | Binary classification |

Both minimum requirements are met: 20 features (at least 12 required) and
1,000 instances (at least 500 required).

**Feature groups**

- *Numerical (7):* `duration_months`, `credit_amount`, `installment_rate_pct`,
  `residence_since_years`, `age_years`, `existing_credits_count`,
  `dependents_count`
- *Categorical (13):* `checking_account_status`, `credit_history`, `purpose`,
  `savings_account`, `employment_since`, `personal_status_sex`, `other_debtors`,
  `property_type`, `other_installment_plans`, `housing`, `job_type`,
  `telephone`, `foreign_worker`

Categorical columns arrive as coded values (`A11`, `A34`, `A143` and so on)
whose meanings are documented in the UCI `german.doc` file. For example
`checking_account_status` ranges from A11 (balance below 0 DM) to A14 (no
checking account).

**Class imbalance.** The 70:30 split matters a great deal. A model that predicts
"good" for every applicant still scores 70% accuracy while catching zero
defaulters. This is why `class_weight="balanced"` is applied where supported,
and why **MCC** — not accuracy — is used to pick the winner.

**Preprocessing.** Numerical features are standardised with `StandardScaler`;
categorical features are one-hot encoded with `handle_unknown="ignore"`. Both
steps live inside a scikit-learn `Pipeline` together with the classifier, so the
identical transformation is applied at training time and inside the deployed
app. This prevents both data leakage and train/serve mismatch.

**Split.** 75% train / 25% test, stratified on the target, `random_state=42`.
The 250-row test partition is saved as `test_data.csv` and is exactly what the
Streamlit app scores.

---

## c. GitHub Repository Link

`https://github.com/partha-das-05952/bits-pilani-aiml`

**Repository structure**

```
bits-pilani-aiml/
│-- app.py                      Streamlit front end
│-- requirements.txt            pinned dependencies
|-- german.doc                  description of german credit dataset
│-- README.md                   this file
│-- test_data.csv               held-out test partition (250 rows)
│-- data/
│   └── german.csv              raw UCI records (1000 rows, no header)
└── model/
    │-- train_models.py         trains and evaluates all five models
    │-- metrics_summary.csv     generated metric table
    │-- readme_table.md         generated markdown table
    └── artifacts/              saved .joblib pipelines (one per model)
```

---

## d. Models Used

All five models are trained on the identical train split and scored on the
identical held-out test split, so the comparison is fair.

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=15`, `class_weight="balanced"` |
| kNN | `n_neighbors=15`, `weights="distance"` |
| Naive Bayes | `GaussianNB` (default settings) |
| Random Forest (Ensemble) | `n_estimators=400`, `max_depth=12`, `class_weight="balanced_subsample"` |

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.748 | **0.807** | 0.556 | **0.800** | **0.656** | **0.486** |
| Decision Tree | 0.636 | 0.683 | 0.427 | 0.627 | 0.508 | 0.246 |
| kNN | **0.760** | 0.761 | **0.703** | 0.347 | 0.464 | 0.366 |
| Naive Bayes | 0.704 | 0.722 | 0.505 | 0.640 | 0.565 | 0.351 |
| Random Forest (Ensemble) | **0.760** | 0.799 | 0.603 | 0.587 | 0.595 | 0.42 |

*Positive class = bad credit (1). Precision, Recall, F1 and MCC are reported for
that class. Test set = 250 rows (175 good, 75 bad).*

**Business cost, using the UCI cost matrix (5 × missed defaulters + 1 × wrongly rejected)**

| ML Model Name | Missed defaulters (FN) | Wrongly rejected (FP) | Total cost |
|---|---|---|---|
| Logistic Regression | 15 | 48 | **123** |
| Decision Tree | 28 | 63 | 203 |
| kNN | 49 | 11 | 256 |
| Naive Bayes | 27 | 47 | 182 |
| Random Forest (Ensemble) | 31 | 29 | 184 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The strongest model on this dataset, which is a genuinely notable result — the simplest algorithm beat the ensemble. It leads on AUC (0.807), F1 (0.656) and MCC (0.486), and catches 80% of the actual defaulters. This suggests the relationship between applicant attributes and default risk is largely linear once the categorical columns are one-hot encoded, leaving little non-linear structure for a more flexible model to exploit. Its precision of 0.556 means roughly half its rejections are false alarms, but under the UCI cost matrix that trade-off is clearly worth it: total cost 123, the lowest of all five. It is also the most interpretable — each coefficient can be explained to a credit committee or a regulator, which matters in lending. |
| **Decision Tree** | The weakest model overall (MCC 0.246, AUC 0.683). A single depth-limited tree captures the dominant splits — checking account status, credit history, loan duration — but it commits to hard axis-aligned cuts and cannot hedge between them. It is also high variance: the result shifts noticeably if the random seed changes. Its value here is explanatory rather than predictive, since the top few splits show clearly which factors drive the decision. |
| **kNN** | The clearest illustration of why accuracy is a misleading metric on imbalanced data. It ties for the highest accuracy (0.760) and has the best precision (0.703), yet its recall is only 0.347 — it misses 49 of the 75 real defaulters. It has no class-weighting mechanism, and one-hot encoding expands 20 features into a much wider sparse space where distance measures lose their meaning, so its neighbourhoods end up dominated by the majority "good" class. Under the cost matrix it is the most expensive model of all (cost 256), despite looking joint-best on accuracy. |
| **Naive Bayes** | Middling accuracy (0.704) with reasonable recall (0.640). Its conditional independence assumption is plainly violated here — loan amount and duration are strongly correlated, as are employment history and savings — so its probability estimates are poorly calibrated and its AUC (0.722) is second lowest. It compensates by being less conservative than kNN, so it does catch a fair share of defaulters. It trains almost instantly, which makes it a useful sanity-check baseline. |
| **Random Forest (Ensemble)** | The runner-up, second on both AUC (0.799) and MCC (0.424). Averaging 400 decorrelated trees fixes the variance problem that crippled the single tree, lifting MCC from 0.246 to 0.424. It produces the most balanced confusion matrix (31 FN, 29 FP), so if false alarms were expensive it would be the sensible pick. It falls slightly short of logistic regression here because the underlying signal is close to linear, and it pays for its performance in interpretability, since it cannot easily justify an individual rejection. |
| **Overall Winner for this dataset** | **Logistic Regression.** It leads on AUC, F1 and MCC, and MCC is the right deciding metric because it accounts for all four confusion-matrix cells and is not inflated by the 70:30 class imbalance. The kNN result shows why accuracy alone would give the wrong answer: kNN matches Random Forest on accuracy while missing two-thirds of the defaulters. Logistic regression also wins on the business criterion, with the lowest cost under the UCI cost matrix (123 against 184 for Random Forest and 256 for kNN). For a lending use case it is therefore the right choice on all three counts — statistical performance, business cost, and the explainability that regulated credit decisions require. |

---


## Reference

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5NC77 — licensed CC BY 4.0.

---

## Author

`Partha Das, 2025ac05952`
