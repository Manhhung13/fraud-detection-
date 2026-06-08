# Fraud Detection - Logistic Regression Baseline Report

## 1. Model configuration

- `model_name`: `logistic_regression`
- `class_weight`: `balanced`
- `C`: `1.0`
- `use_rare_category_grouping`: `True`
- `min_frequency`: `50`
- `threshold_metric`: `f2`
- `numeric_features`: `19`
- `categorical_features`: `21`
- `binary_features`: `10`
- `interaction_features`: `8`

## 2. Metrics at threshold 0.5

| split      |   threshold |   rows |   fraud_count |   fraud_rate |   roc_auc |   pr_auc |   precision |   recall |       f1 |       f2 |   balanced_accuracy |   review_rate |   predicted_positive |    tn |    fp |   fn |   tp |
|:-----------|------------:|-------:|--------------:|-------------:|----------:|---------:|------------:|---------:|---------:|---------:|--------------------:|--------------:|---------------------:|------:|------:|-----:|-----:|
| train      |         0.5 |  70000 |          1783 |    0.0254714 |  0.902776 | 0.506004 |    0.125261 | 0.8424   | 0.218092 | 0.392721 |            0.844321 |      0.1713   |                11991 | 57728 | 10489 |  281 | 1502 |
| validation |         0.5 |  15000 |           378 |    0.0252    |  0.86805  | 0.455446 |    0.116738 | 0.798942 | 0.20371  | 0.368383 |            0.821335 |      0.172467 |                 2587 | 12337 |  2285 |   76 |  302 |

## 3. Best threshold selected on validation

| split                     |   threshold |   rows |   fraud_count |   fraud_rate |   roc_auc |   pr_auc |   precision |   recall |       f1 |       f2 |   balanced_accuracy |   review_rate |   predicted_positive |    tn |   fp |   fn |   tp |
|:--------------------------|------------:|-------:|--------------:|-------------:|----------:|---------:|------------:|---------:|---------:|---------:|--------------------:|--------------:|---------------------:|------:|-----:|-----:|-----:|
| validation_best_threshold |        0.76 |  15000 |           378 |       0.0252 |   0.86805 | 0.455446 |    0.325333 | 0.645503 | 0.432624 | 0.539346 |            0.805449 |          0.05 |                  750 | 14116 |  506 |  134 |  244 |

## 4. Final test metrics using selected validation threshold

| split   |   threshold |   rows |   fraud_count |   fraud_rate |   roc_auc |   pr_auc |   precision |   recall |       f1 |       f2 |   balanced_accuracy |   review_rate |   predicted_positive |    tn |   fp |   fn |   tp |
|:--------|------------:|-------:|--------------:|-------------:|----------:|---------:|------------:|---------:|---------:|---------:|--------------------:|--------------:|---------------------:|------:|-----:|-----:|-----:|
| test    |        0.76 |  15000 |           388 |    0.0258667 |  0.887131 | 0.448793 |    0.359249 | 0.690722 | 0.472663 | 0.583116 |            0.829004 |     0.0497333 |                  746 | 14134 |  478 |  120 |  268 |

## 5. Top positive coefficients: increase fraud risk

| feature               |   coefficient |   abs_coefficient | direction     |
|:----------------------|--------------:|------------------:|:--------------|
| categorical__mcc_3070 |       3.35938 |           3.35938 | increase_risk |
| categorical__mcc_3064 |       3.09228 |           3.09228 | increase_risk |
| categorical__mcc_3220 |       2.6887  |           2.6887  | increase_risk |
| categorical__mcc_3061 |       2.63007 |           2.63007 | increase_risk |
| categorical__mcc_3293 |       2.40131 |           2.40131 | increase_risk |
| categorical__mcc_3004 |       2.28781 |           2.28781 | increase_risk |
| categorical__mcc_3181 |       2.25734 |           2.25734 | increase_risk |
| categorical__mcc_3010 |       2.1827  |           2.1827  | increase_risk |
| categorical__mcc_3304 |       2.1521  |           2.1521  | increase_risk |
| categorical__mcc_3311 |       2.10824 |           2.10824 | increase_risk |
| categorical__mcc_3322 |       2.06427 |           2.06427 | increase_risk |
| categorical__mcc_3253 |       1.93821 |           1.93821 | increase_risk |
| categorical__mcc_3270 |       1.93239 |           1.93239 | increase_risk |
| categorical__mcc_3138 |       1.90003 |           1.90003 | increase_risk |
| categorical__mcc_3080 |       1.8895  |           1.8895  | increase_risk |
| categorical__mcc_3089 |       1.85801 |           1.85801 | increase_risk |
| categorical__mcc_3191 |       1.79613 |           1.79613 | increase_risk |
| categorical__mcc_3124 |       1.77046 |           1.77046 | increase_risk |
| categorical__mcc_3097 |       1.72355 |           1.72355 | increase_risk |
| categorical__mcc_3006 |       1.65508 |           1.65508 | increase_risk |

## 6. Top negative coefficients: decrease fraud risk

| feature               |   coefficient |   abs_coefficient | direction     |
|:----------------------|--------------:|------------------:|:--------------|
| categorical__mcc_3059 |      -1.97081 |           1.97081 | decrease_risk |
| categorical__mcc_3047 |      -1.86691 |           1.86691 | decrease_risk |
| categorical__mcc_3313 |      -1.86102 |           1.86102 | decrease_risk |
| categorical__mcc_3132 |      -1.841   |           1.841   | decrease_risk |
| categorical__mcc_3221 |      -1.78522 |           1.78522 | decrease_risk |
| categorical__mcc_3148 |      -1.7806  |           1.7806  | decrease_risk |
| categorical__mcc_3167 |      -1.76238 |           1.76238 | decrease_risk |
| categorical__mcc_3260 |      -1.76223 |           1.76223 | decrease_risk |
| categorical__mcc_3337 |      -1.73911 |           1.73911 | decrease_risk |
| categorical__mcc_3312 |      -1.71778 |           1.71778 | decrease_risk |
| categorical__mcc_3144 |      -1.70677 |           1.70677 | decrease_risk |
| categorical__mcc_3048 |      -1.70599 |           1.70599 | decrease_risk |
| categorical__mcc_3152 |      -1.68957 |           1.68957 | decrease_risk |
| categorical__mcc_3222 |      -1.66877 |           1.66877 | decrease_risk |
| categorical__mcc_3002 |      -1.66076 |           1.66076 | decrease_risk |
| categorical__mcc_3301 |      -1.66046 |           1.66046 | decrease_risk |
| categorical__mcc_3067 |      -1.64784 |           1.64784 | decrease_risk |
| categorical__mcc_3041 |      -1.64095 |           1.64095 | decrease_risk |
| categorical__mcc_3299 |      -1.63728 |           1.63728 | decrease_risk |
| categorical__mcc_3255 |      -1.62878 |           1.62878 | decrease_risk |

## 7. Business interpretation


- PR-AUC quan trọng hơn accuracy vì fraud là lớp thiểu số.
- Recall cho biết model bắt được bao nhiêu fraud thật.
- Precision cho biết trong các giao dịch bị cảnh báo, bao nhiêu là fraud thật.
- Review rate cho biết tỷ lệ giao dịch bị đẩy sang kiểm tra thủ công.
- Threshold được chọn trên validation, sau đó áp dụng nguyên sang test.
- Test set không được dùng để chọn threshold.
