# Final Fraud Detection Model Report

## 1. Mục tiêu báo cáo


Báo cáo này tổng hợp kết quả cuối cùng của project fraud detection, so sánh:

1. Logistic Regression baseline.
2. CatBoost GPU có manual interaction features.
3. CatBoost GPU bỏ manual interaction features.

Mục tiêu là chọn model cuối cùng dựa trên PR-AUC, F2, precision, recall, review rate và khả năng giải thích bằng insight nghiệp vụ.


## 2. Bảng so sánh model

| model_label                             | model_role                              |   threshold |   n_features |   pr_auc |   roc_auc | precision   | recall   |     f1 |     f2 | review_rate   |   tp |   fp |   fn |    tn | note                                                           |
|:----------------------------------------|:----------------------------------------|------------:|-------------:|---------:|----------:|:------------|:---------|-------:|-------:|:--------------|-----:|-----:|-----:|------:|:---------------------------------------------------------------|
| lr_no_mcc                               | Baseline - dễ giải thích                |        0.77 |          171 |   0.5332 |    0.8952 | 40.45%      | 69.85%   | 0.5123 | 0.6098 | 4.47%         |  271 |  399 |  117 | 14213 | Logistic Regression baseline, không dùng MCC.                  |
| cat_gpu_mcc_minfreq_100                 | Advanced - có manual interactions       |        0.73 |          172 |   0.6944 |    0.8974 | 51.32%      | 75.00%   | 0.6094 | 0.6866 | 3.78%         |  291 |  276 |   97 | 14336 | CatBoost GPU, dùng MCC rare grouping, có interaction thủ công. |
| cat_gpu_mcc_minfreq_100_no_interactions | Final advanced - bỏ manual interactions |        0.74 |          164 |   0.6967 |    0.8978 | 52.64%      | 74.48%   | 0.6169 | 0.6878 | 3.66%         |  289 |  260 |   99 | 14352 | CatBoost GPU, dùng MCC rare grouping, bỏ interaction thủ công. |

## 3. Model được chọn cuối cùng


Model được chọn cuối cùng là:

**`cat_gpu_mcc_minfreq_100_no_interactions`**

Lý do:
- Có PR-AUC và F2 tốt nhất hoặc gần tốt nhất.
- Precision cao hơn bản có interaction.
- Review rate thấp hơn.
- Số feature ít hơn vì bỏ manual interaction features.
- Feature importance vẫn khớp với insight giai đoạn 3.
- CatBoost có khả năng tự học quan hệ phi tuyến và interaction từ feature gốc.


## 4. Diễn giải nghiệp vụ trên tập test


Trên tập test có **15,000** giao dịch, trong đó có **388** fraud thật.

Với threshold **0.74**, model:
- Cảnh báo **549** giao dịch cần review.
- Bắt đúng **289** fraud.
- Báo nhầm **260** giao dịch thường.
- Bỏ sót **99** fraud.
- Xác định đúng **14,352** giao dịch thường.
- Review rate: **3.66%**.
- Precision: **52.64%**.
- Recall: **74.48%**.


## 5. Kiểm tra overfit: train vs validation vs test

| split                     |   threshold |   pr_auc | precision   | recall   |     f1 |     f2 | review_rate   |
|:--------------------------|------------:|---------:|:------------|:---------|-------:|-------:|:--------------|
| train_threshold_0.5       |        0.5  |   0.8549 | 33.86%      | 97.98%   | 0.5033 | 0.7107 | 7.37%         |
| validation_best_threshold |        0.74 |   0.6404 | 52.45%      | 67.99%   | 0.5922 | 0.6419 | 3.27%         |
| test                      |        0.74 |   0.6967 | 52.64%      | 74.48%   | 0.6169 | 0.6878 | 3.66%         |

Nhận xét:
- Train PR-AUC cao hơn validation/test là bình thường với boosting model.
- Validation và test không bị sụp; test còn tốt hơn validation.
- Threshold chọn trên validation áp dụng sang test vẫn cho kết quả tốt.
- Không có dấu hiệu overfit nghiêm trọng.


## 6. Top feature importance của final model

| feature                                    |   importance |
|:-------------------------------------------|-------------:|
| numeric__mcc_entropy_30d                   |      14.8122 |
| numeric__night_ratio_30d                   |      10.5591 |
| numeric__log_amount_to_max_30d             |       9.6016 |
| numeric__spending_trend                    |       9.1716 |
| numeric__distinct_countries_30d            |       8.7102 |
| numeric__amount_z_30d                      |       7.5883 |
| numeric__decline_rate_30d                  |       3.8854 |
| numeric__device_diversity_30d              |       3.8812 |
| numeric__credit_util_today                 |       3.6087 |
| numeric__log_mean_amount_30d               |       2.5674 |
| numeric__distinct_merchants_7d             |       2.1471 |
| numeric__txn_count_ratio_7d_30d            |       1.8273 |
| numeric__ip_score                          |       1.374  |
| numeric__txn_count_7d                      |       1.2582 |
| numeric__txn_count_30d                     |       1.07   |
| numeric__amount_to_mean_30d                |       1.027  |
| numeric__hour                              |       1.0195 |
| binary__cross_border_flag                  |       0.8215 |
| binary__high_max_amount_30d_flag           |       0.8182 |
| categorical__night_ratio_group_medium      |       0.5061 |
| categorical__night_ratio_group_high        |       0.4388 |
| categorical__amount_to_max_group_above_max |       0.3838 |
| binary__low_country_diversity_flag         |       0.3348 |
| categorical__term_location_WEARABLE_PAY    |       0.3272 |
| numeric__night_unusual_score               |       0.3036 |

## 7. Giải thích bằng insight nghiệp vụ


Các feature quan trọng nhất của final CatBoost model tập trung vào các nhóm insight đã khai phá:

- **MCC behavior**: `mcc_entropy_30d` là feature quan trọng nhất. Model học hành vi đa dạng ngành hàng thay vì học thuộc mã MCC cụ thể.
- **Night behavior**: `night_ratio_30d` đứng top đầu, phản ánh thói quen giao dịch theo thời gian.
- **Amount anomaly**: `log_amount_to_max_30d`, `amount_z_30d` cho thấy số tiền hiện tại lệch khỏi lịch sử user là tín hiệu mạnh.
- **Spending trend**: `spending_trend` có importance cao, khớp với insight spending trend bất thường.
- **Country behavior**: `distinct_countries_30d` thể hiện lịch sử đa dạng quốc gia của user.
- **Technical/risk history**: `decline_rate_30d`, `device_diversity_30d`, `credit_util_today`, `ip_score` hỗ trợ thêm cho model.

Việc bỏ manual interaction features không làm giảm performance vì CatBoost có thể tự học interaction thông qua các split nhiều tầng.


## 8. Kết luận cuối


Logistic Regression được giữ làm baseline vì dễ giải thích bằng coefficient và giúp kiểm chứng feature engineering.
Tuy nhiên, CatBoost GPU với MCC rare grouping `min_frequency=100` và không dùng manual interaction features được chọn làm final advanced model vì đạt hiệu năng tốt hơn rõ rệt, review rate thấp hơn và vẫn học đúng các insight nghiệp vụ chính.
