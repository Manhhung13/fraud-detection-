# Consolidated Fraud Detection Notebook Report

Generated from executed notebooks on 2026-06-07. Source notebooks: `notebooks/01_data_understanding.ipynb` through `notebooks/09_error_analysis_and_threshold_policy.ipynb`.

## 1. Notebook Execution Status

| Notebook | Status | Runtime seconds |
| --- | --- | --- |
| 01_data_understanding.ipynb | OK | 12.85 |
| 02_univariate_eda.ipynb | OK | 8.85 |
| 03_bivariate_eda_with_target.ipynb | OK | 9.81 |
| 04_hidden_insight_mining.ipynb | OK | 7.58 |
| 05_feature_engineering_experiment.ipynb | OK | 7.23 |
| 06_model_baseline_logistic.ipynb | OK | 6.58 |
| 07_advanced_model_comparison.ipynb | OK | 6.54 |
| 08_final_model_explainability.ipynb | OK | 6.78 |
| 09_error_analysis_and_threshold_policy.ipynb | OK | 6.71 |

All notebooks executed successfully via `jupyter nbconvert --execute`. Executed copies and parsed outputs are stored under `notebooks/_executed_summary/`.

## 2. Executive Summary

- Dataset size: 100,000 rows, 36 raw columns, no duplicate rows, no missing values.
- Target imbalance is high: 2,549 fraud rows out of 100,000 (2.55% fraud rate).
- Cleaning/feature parsing expanded the dataset from 36 raw columns to 59 cleaned columns, adding 23 parsed/time/IP/transaction features.
- Model-ready feature set contains 58 features: 19 numeric, 21 categorical, 10 binary, and 8 interaction features.
- Logistic Regression baseline reached test PR-AUC 0.533, recall 69.85%, precision 40.45%, F2 0.610, review rate 4.47%.
- Best tuned policy model is cat_tune_depth7_l2_8_lr003_iter700: test PR-AUC 0.703, recall 77.58%, precision 50.08%, F2 0.699, review rate 4.01%.
- Recommended operating point for 5% review capacity uses threshold 0.66 from validation, yielding test recall 77.58%, precision 50.08%, F2 0.699, and review rate 4.01%.

## 3. Data Understanding

| Rows | Columns | Duplicates | Missing values | Fraud count | Fraud rate |
| --- | --- | --- | --- | --- | --- |
| 100000 | 36 | 0 | 0 | 2549 | 2.55% |

Key implications: fraud is rare, so model selection should prioritize PR-AUC, recall, F2, and operational review rate rather than accuracy. Raw identifiers such as `device_id`, `merchant_id`, `ip_risk`, timestamp, and amount-like continuous fields have high cardinality and should not be one-hot encoded directly.

## 4. EDA Highlights

- No missing values were found in raw or cleaned data.
- Highly skewed behavioral/risk variables include `std_amount_30d`, `mean_amount_30d`, `chargebacks_365d`, and `decline_rate_30d`. These are better handled with transformations, bins, or tree models.
- High-cardinality fields were handled through parsing, dropping, grouping, or feature extraction instead of naive one-hot encoding.

| Feature | Mean | Median | Max | Skew |
| --- | --- | --- | --- | --- |
| std_amount_30d | 2.449 | 1.830 | 42.970 | 4.532 |
| mean_amount_30d | 7.622 | 6.550 | 97.990 | 4.040 |
| chargebacks_365d | 0.502 | 0.000 | 10.000 | 3.974 |
| decline_rate_30d | 0.166 | 0.154 | 0.666 | 0.787 |
| mcc | 5174.558 | 5003.000 | 7999.000 | 0.297 |
| online_share_7d | 0.500 | 0.500 | 1.000 | 0.006 |

## 5. Insight Mining

Stable high-risk patterns found on train and validation concentrate around behavioral anomaly combinations: low MCC entropy, low night ratio, high max amount, low spending trend, low country diversity, and card-not-present context.

### Stable Segments

| Segment | Train lift | Valid lift | Train fraud rate | Valid fraud rate | Valid support |
| --- | --- | --- | --- | --- | --- |
| night_unusual | 2.796 | 3.280 | 7.12% | 8.27% | 496 |
| very_low_mcc_entropy | 3.792 | 3.228 | 9.66% | 8.13% | 750 |
| high_max_amount_30d | 2.462 | 3.069 | 6.27% | 7.73% | 1500 |
| low_night_ratio | 2.850 | 2.910 | 7.26% | 7.33% | 1500 |
| low_mcc_entropy | 3.501 | 2.904 | 8.92% | 7.32% | 1503 |
| low_country_diversity | 2.435 | 2.456 | 6.20% | 6.19% | 1842 |

### Stable Numeric Patterns

| Feature | Bin | Train lift | Valid lift | Valid fraud rate |
| --- | --- | --- | --- | --- |
| max_amount_30d | (225176.777, inf] | 2.462 | 3.160 | 7.96% |
| mcc_entropy_30d | (-inf, 0.393] | 3.501 | 2.914 | 7.34% |
| night_ratio_30d | (-inf, 0.101] | 2.850 | 2.877 | 7.25% |
| spending_trend | (-inf, 0.584] | 2.596 | 2.523 | 6.36% |
| distinct_countries_30d | (-inf, 3.0] | 2.435 | 2.456 | 6.19% |
| mcc_entropy_30d | (0.393, 0.788] | 2.319 | 2.401 | 6.05% |
| night_unusual_score | (0.705, inf] | 2.140 | 2.364 | 5.96% |
| spending_trend | (0.584, 1.073] | 1.854 | 2.112 | 5.32% |

### Stable Interaction/Categorical Patterns

| Feature | Value | Train lift | Valid lift | Valid fraud rate | Valid support |
| --- | --- | --- | --- | --- | --- |
| low_mcc_entropy_x_high_max_amount_30d | 1 | 8.920 | 8.995 | 22.67% | 150 |
| low_mcc_entropy_x_low_night_ratio | 1 | 10.210 | 8.985 | 22.64% | 159 |
| low_night_ratio_x_high_max_amount_30d | 1 | 6.687 | 8.088 | 20.38% | 157 |
| low_mcc_entropy_x_low_spending_trend | 1 | 9.373 | 7.937 | 20.00% | 130 |
| night_unusual_group | very_high | 2.803 | 3.280 | 8.27% | 496 |
| low_mcc_entropy_x_card_not_present | 1 | 3.600 | 2.867 | 7.22% | 969 |
| night_ratio_group | very_low | 2.852 | 2.851 | 7.19% | 1503 |
| cross_border_x_low_country_diversity | 1 | 1.995 | 2.579 | 6.50% | 400 |

## 6. Feature Engineering

| Split | Rows | Fraud count | Fraud rate | Model features | Numeric | Categorical | Binary | Interactions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 70000 | 1783 | 2.55% | 58 | 19 | 21 | 10 | 8 |
| validation | 15000 | 378 | 2.52% | 58 | 19 | 21 | 10 | 8 |
| test | 15000 | 388 | 2.59% | 58 | 19 | 21 | 10 | 8 |

Dropped non-model columns: `ip_risk`, `txn_counts`, `ip_address`, `transaction_datetime`.

Feature engineering keeps the target distribution stable across time-based train/validation/test splits, which reduces the risk that validation/test performance is an artifact of random splitting.

## 7. Baseline Logistic Regression

| Split | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F2 | Review rate | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0.77 | 0.533 | 0.895 | 40.45% | 69.85% | 0.610 | 4.47% | 271 | 399 | 117 |

The baseline is useful for interpretability, but it misses more frauds and generates more false positives than tuned CatBoost at comparable review rates.

## 8. Advanced Model Comparison

### GPU Model Search

| Experiment | Model | Threshold | Test PR-AUC | Precision | Recall | F2 | Review | TP/FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cat_gpu_mcc_minfreq_100_no_interactions | catboost | 0.74 | 0.697 | 52.64% | 74.48% | 0.688 | 3.66% | 289/260/99 |
| cat_gpu_mcc_minfreq_100 | catboost | 0.73 | 0.694 | 51.32% | 75.00% | 0.687 | 3.78% | 291/276/97 |
| lgbm_gpu_no_mcc | lightgbm | 0.62 | 0.679 | 50.17% | 74.23% | 0.677 | 3.83% | 288/286/100 |
| lgbm_gpu_mcc_minfreq_100 | lightgbm | 0.62 | 0.679 | 50.17% | 74.23% | 0.677 | 3.83% | 288/286/100 |
| cat_gpu_no_mcc | catboost | 0.76 | 0.692 | 54.30% | 71.65% | 0.673 | 3.41% | 278/234/110 |
| xgb_gpu_mcc_minfreq_100 | xgboost | 0.68 | 0.685 | 49.23% | 73.97% | 0.672 | 3.89% | 287/296/101 |

### CatBoost Tuning

| Experiment | Threshold | Test PR-AUC | Precision | Recall | F2 | Review | TP/FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cat_tune_depth7_l2_8_lr003_iter700 | 0.66 | 0.703 | 50.08% | 77.58% | 0.699 | 4.01% | 301/300/87 |
| cat_tune_depth6_l2_10_lr003_iter700 | 0.74 | 0.696 | 52.44% | 74.74% | 0.689 | 3.69% | 290/263/98 |
| cat_tune_base_depth6_l2_5_lr003_iter700 | 0.74 | 0.697 | 52.64% | 74.48% | 0.688 | 3.66% | 289/260/99 |
| cat_tune_depth5_l2_8_lr003_iter700 | 0.74 | 0.682 | 47.60% | 76.55% | 0.682 | 4.16% | 297/327/91 |
| cat_tune_depth6_l2_5_lr002_iter1000 | 0.71 | 0.693 | 47.67% | 76.29% | 0.681 | 4.14% | 296/325/92 |

Best tuned model by test F2 is `cat_tune_depth7_l2_8_lr003_iter700`. It improves recall and F2 over Logistic Regression while staying below the 5% review-capacity constraint.

## 9. Final Model Explainability

Notebook 08 reads the final explainability artifacts and shows that model signal is dominated by behavioral anomaly features rather than raw IDs. Top CatBoost feature importances:

| Feature | Importance |
| --- | --- |
| numeric__mcc_entropy_30d | 14.812 |
| numeric__night_ratio_30d | 10.559 |
| numeric__log_amount_to_max_30d | 9.602 |
| numeric__spending_trend | 9.172 |
| numeric__distinct_countries_30d | 8.710 |
| numeric__amount_z_30d | 7.588 |
| numeric__decline_rate_30d | 3.885 |
| numeric__device_diversity_30d | 3.881 |
| numeric__credit_util_today | 3.609 |
| numeric__log_mean_amount_30d | 2.567 |
| numeric__distinct_merchants_7d | 2.147 |
| numeric__txn_count_ratio_7d_30d | 1.827 |
| numeric__ip_score | 1.374 |
| numeric__txn_count_7d | 1.258 |
| numeric__txn_count_30d | 1.070 |

Insight group summary from the explanation artifacts:

| Insight group | Feature count | Total abs coef | Mean abs coef | Max abs coef |
| --- | --- | --- | --- | --- |
| Time behavior anomaly | 21 | 8.110 | 0.386 | 1.161 |
| Payment / authentication context | 68 | 6.936 | 0.102 | 0.337 |
| Amount anomaly | 9 | 4.345 | 0.483 | 0.884 |
| MCC behavior anomaly | 8 | 3.612 | 0.451 | 0.710 |
| Country / cross-border behavior | 24 | 2.963 | 0.123 | 0.420 |
| Risk history / technical risk | 17 | 2.493 | 0.147 | 0.348 |
| Transaction velocity | 12 | 1.589 | 0.132 | 0.296 |
| Other context | 10 | 0.696 | 0.070 | 0.211 |
| Spending trend anomaly | 2 | 0.647 | 0.324 | 0.537 |

The strongest business themes are time behavior anomaly, amount anomaly, MCC behavior anomaly, payment/authentication context, and country/cross-border behavior.

## 10. Error Analysis and Threshold Policy

Threshold policy comparison from notebook 09:

| Review cap | Threshold | Test precision | Test recall | Test F2 | Test review | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3% | 0.73 | 57.26% | 72.16% | 0.686 | 3.26% | 280 | 209 | 108 | 14403 |
| 5% | 0.66 | 50.08% | 77.58% | 0.699 | 4.01% | 301 | 300 | 87 | 14312 |
| 10% | 0.66 | 50.08% | 77.58% | 0.699 | 4.01% | 301 | 300 | 87 | 14312 |

For the 5% review-capacity operating point, the policy flags 601 of 15,000 test transactions for review, catches 301 of 388 frauds, and leaves 87 false negatives. This is the strongest operating point in the generated policy table by F2.

Error-type distribution at the selected policy point:

| Type | Count | Ratio |
| --- | --- | --- |
| TN | 14312 | 95.41% |
| TP | 301 | 2.01% |
| FP | 300 | 2.00% |
| FN | 87 | 0.58% |

## 11. Recommended Final Position

Use `cat_tune_depth7_l2_8_lr003_iter700` as the final policy model when the business objective is fraud capture under a manual-review cap. The practical default threshold should be 0.66 for a 5% review-capacity policy, because it achieves the best F2 in the threshold policy table while reviewing about 4.01% of test transactions.

For a stricter review operation, threshold 0.73 reduces test review rate to 3.26% and improves precision to 57.26%, but recall drops from 77.58% to 72.16%. This is a business tradeoff, not a modeling error.

## 12. Caveats and Next Steps

- The dataset appears synthetic or project-provided; production deployment still needs drift checks, leakage review, and calibration on newer data.
- Some notebook explainability artifacts are coefficient-style summaries while the best threshold policy is from tuned CatBoost. Keep these artifact lineages explicit when presenting results.
- Add a final model registry entry that records the chosen model directory, threshold, feature metadata hash, and evaluation timestamp.
- Consider probability calibration and cost-based thresholding if false positives and false negatives have known business costs.
