# Final Data Science Project Report - Fraud Detection

## 1. Business problem


Mục tiêu của project là xây dựng mô hình phát hiện gian lận giao dịch tài chính.
Do dữ liệu mất cân bằng mạnh, accuracy không được dùng làm metric chính. 
Các metric chính gồm PR-AUC, Precision, Recall, F1, F2, Confusion Matrix và Review Rate.


## 2. Data science workflow


Quy trình thực hiện:

1. Khảo sát dữ liệu và metadata.
2. EDA và kiểm tra mất cân bằng nhãn.
3. Insight mining trên train để tránh leakage.
4. Feature engineering dựa trên insight.
5. Train Logistic Regression baseline.
6. Train advanced boosting models: LightGBM, XGBoost, CatBoost.
7. Tuning CatBoost.
8. Chọn final model.
9. Error analysis.
10. Threshold policy theo năng lực review.
11. Inference pipeline, FastAPI, Streamlit demo.


---

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


---

# Error Analysis Report

## 1. Error type counts

| error_type   |   count |     ratio |
|:-------------|--------:|----------:|
| TN           |   14312 | 0.954133  |
| TP           |     301 | 0.0200667 |
| FP           |     300 | 0.02      |
| FN           |      87 | 0.0058    |

## 2. Numeric summary by error type

| error_type   |   rows |   mcc_entropy_30d_mean |   mcc_entropy_30d_median |   mcc_entropy_30d_p90 |   night_ratio_30d_mean |   night_ratio_30d_median |   night_ratio_30d_p90 |   log_amount_to_max_30d_mean |   log_amount_to_max_30d_median |   log_amount_to_max_30d_p90 |   spending_trend_mean |   spending_trend_median |   spending_trend_p90 |   distinct_countries_30d_mean |   distinct_countries_30d_median |   distinct_countries_30d_p90 |   amount_z_30d_mean |   amount_z_30d_median |   amount_z_30d_p90 |   decline_rate_30d_mean |   decline_rate_30d_median |   decline_rate_30d_p90 |   device_diversity_30d_mean |   device_diversity_30d_median |   device_diversity_30d_p90 |   credit_util_today_mean |   credit_util_today_median |   credit_util_today_p90 |   ip_score_mean |   ip_score_median |   ip_score_p90 |   txn_count_ratio_7d_30d_mean |   txn_count_ratio_7d_30d_median |   txn_count_ratio_7d_30d_p90 |
|:-------------|-------:|-----------------------:|-------------------------:|----------------------:|-----------------------:|-------------------------:|----------------------:|-----------------------------:|-------------------------------:|----------------------------:|----------------------:|------------------------:|---------------------:|------------------------------:|--------------------------------:|-----------------------------:|--------------------:|----------------------:|-------------------:|------------------------:|--------------------------:|-----------------------:|----------------------------:|------------------------------:|---------------------------:|-------------------------:|---------------------------:|------------------------:|----------------:|------------------:|---------------:|------------------------------:|--------------------------------:|-----------------------------:|
| FN           |     87 |               1.84552  |                    1.674 |                3.6292 |               0.444218 |                   0.424  |                0.8264 |                     0.673249 |                       0.525867 |                    1.37981  |               2.78152 |                   2.997 |               4.573  |                      12.3908  |                              12 |                           23 |             36263.8 |               28418.1 |            75619   |                0.17008  |                     0.153 |                 0.3018 |                     19.5747 |                            19 |                       35.4 |                 0.94292  |                     0.885  |                  1.7098 |        0.298206 |           0.2709  |        0.54284 |                      0.275334 |                        0.174545 |                     0.71991  |
| FP           |    300 |               0.71918  |                    0.631 |                1.4964 |               0.234277 |                   0.199  |                0.4618 |                     0.472225 |                       0.44773  |                    0.821491 |               1.43179 |                   1.239 |               2.9046 |                       8.77    |                               7 |                           18 |             34747   |               31818.1 |            65918.6 |                0.149683 |                     0.138 |                 0.2421 |                     17.8367 |                            17 |                       34.1 |                 0.888407 |                     0.8725 |                  1.6917 |        0.271411 |           0.25125 |        0.48218 |                      0.28516  |                        0.219566 |                     0.623674 |
| TN           |  14312 |               2.0479   |                    2.05  |                3.6159 |               0.510874 |                   0.5175 |                0.906  |                     0.794473 |                       0.59394  |                    1.63978  |               2.59429 |                   2.611 |               4.519  |                      13.1895  |                              13 |                           23 |             33969.2 |               30596.7 |            67590.6 |                0.168142 |                     0.156 |                 0.285  |                     20.6686 |                            21 |                       37   |                 1.0064   |                     1.008  |                  1.804  |        0.284379 |           0.2649  |        0.50749 |                      0.256695 |                        0.190648 |                     0.609756 |
| TP           |    301 |               0.703837 |                    0.563 |                1.52   |               0.201681 |                   0.16   |                0.444  |                     0.440032 |                       0.431062 |                    0.71118  |               1.23748 |                   0.998 |               2.614  |                       7.02326 |                               6 |                           14 |             42042.1 |               39114.9 |            80802.8 |                0.137625 |                     0.126 |                 0.228  |                     16.3854 |                            16 |                       32   |                 0.78285  |                     0.685  |                  1.58   |        0.279373 |           0.259   |        0.4912  |                      0.262765 |                        0.2      |                     0.6      |

## 3. Top categorical patterns by error type

| error_type   | feature          | value        |     ratio |   count |
|:-------------|:-----------------|:-------------|----------:|--------:|
| FN           | payment_channel  | MOBILE_APP   | 0.183908  |      16 |
| FN           | payment_channel  | SMART_WATCH  | 0.172414  |      15 |
| FN           | payment_channel  | IVR_PHONE    | 0.149425  |      13 |
| FN           | payment_channel  | API_SERVER   | 0.149425  |      13 |
| FN           | payment_channel  | WEB_BROWSER  | 0.137931  |      12 |
| FN           | payment_channel  | ATM          | 0.114943  |      10 |
| FN           | payment_channel  | POS_TERMINAL | 0.091954  |       8 |
| FP           | payment_channel  | IVR_PHONE    | 0.19      |      57 |
| FP           | payment_channel  | SMART_WATCH  | 0.17      |      51 |
| FP           | payment_channel  | ATM          | 0.15      |      45 |
| FP           | payment_channel  | WEB_BROWSER  | 0.126667  |      38 |
| FP           | payment_channel  | API_SERVER   | 0.123333  |      37 |
| FP           | payment_channel  | MOBILE_APP   | 0.12      |      36 |
| FP           | payment_channel  | POS_TERMINAL | 0.12      |      36 |
| TN           | payment_channel  | API_SERVER   | 0.14652   |    2097 |
| TN           | payment_channel  | SMART_WATCH  | 0.145123  |    2077 |
| TN           | payment_channel  | ATM          | 0.144564  |    2069 |
| TN           | payment_channel  | MOBILE_APP   | 0.143167  |    2049 |
| TN           | payment_channel  | POS_TERMINAL | 0.143097  |    2048 |
| TN           | payment_channel  | WEB_BROWSER  | 0.141979  |    2032 |
| TN           | payment_channel  | IVR_PHONE    | 0.135551  |    1940 |
| TP           | payment_channel  | POS_TERMINAL | 0.162791  |      49 |
| TP           | payment_channel  | ATM          | 0.159468  |      48 |
| TP           | payment_channel  | MOBILE_APP   | 0.142857  |      43 |
| TP           | payment_channel  | API_SERVER   | 0.142857  |      43 |
| TP           | payment_channel  | WEB_BROWSER  | 0.136213  |      41 |
| TP           | payment_channel  | IVR_PHONE    | 0.136213  |      41 |
| TP           | payment_channel  | SMART_WATCH  | 0.119601  |      36 |
| FN           | merchant_country | VN           | 0.0804598 |       7 |
| FN           | merchant_country | ZA           | 0.0804598 |       7 |
| FN           | merchant_country | SG           | 0.0804598 |       7 |
| FN           | merchant_country | MY           | 0.0689655 |       6 |
| FN           | merchant_country | HK           | 0.0689655 |       6 |
| FN           | merchant_country | FR           | 0.0689655 |       6 |
| FN           | merchant_country | AU           | 0.0689655 |       6 |
| FN           | merchant_country | BR           | 0.0574713 |       5 |
| FN           | merchant_country | ID           | 0.0574713 |       5 |
| FN           | merchant_country | CA           | 0.0574713 |       5 |
| FP           | merchant_country | KR           | 0.08      |      24 |
| FP           | merchant_country | US           | 0.0733333 |      22 |
| FP           | merchant_country | PH           | 0.0666667 |      20 |
| FP           | merchant_country | ZA           | 0.0666667 |      20 |
| FP           | merchant_country | MY           | 0.0566667 |      17 |
| FP           | merchant_country | VN           | 0.0533333 |      16 |
| FP           | merchant_country | AE           | 0.0533333 |      16 |
| FP           | merchant_country | GB           | 0.05      |      15 |
| FP           | merchant_country | HK           | 0.05      |      15 |
| FP           | merchant_country | FR           | 0.05      |      15 |
| TN           | merchant_country | AU           | 0.0554779 |     794 |
| TN           | merchant_country | JP           | 0.0514952 |     737 |
| TN           | merchant_country | KR           | 0.0512158 |     733 |
| TN           | merchant_country | CN           | 0.051076  |     731 |
| TN           | merchant_country | ID           | 0.0510061 |     730 |
| TN           | merchant_country | BR           | 0.0509363 |     729 |
| TN           | merchant_country | MY           | 0.0505869 |     724 |
| TN           | merchant_country | IN           | 0.0503074 |     720 |
| TN           | merchant_country | VN           | 0.0503074 |     720 |
| TN           | merchant_country | AE           | 0.0502376 |     719 |
| TP           | merchant_country | JP           | 0.0730897 |      22 |
| TP           | merchant_country | GB           | 0.0697674 |      21 |
| TP           | merchant_country | US           | 0.0664452 |      20 |
| TP           | merchant_country | CN           | 0.0631229 |      19 |
| TP           | merchant_country | BR           | 0.0564784 |      17 |
| TP           | merchant_country | VN           | 0.0531561 |      16 |
| TP           | merchant_country | ZA           | 0.0531561 |      16 |
| TP           | merchant_country | SG           | 0.0498339 |      15 |
| TP           | merchant_country | AE           | 0.0498339 |      15 |
| TP           | merchant_country | TH           | 0.0498339 |      15 |
| FN           | card_entry_mode  | MAGSTRIPE    | 0.241379  |      21 |
| FN           | card_entry_mode  | MANUAL_KEYED | 0.172414  |      15 |
| FN           | card_entry_mode  | CONTACTLESS  | 0.172414  |      15 |
| FN           | card_entry_mode  | CHIP         | 0.16092   |      14 |
| FN           | card_entry_mode  | QR_CODE      | 0.149425  |      13 |
| FN           | card_entry_mode  | TOKEN_IN_APP | 0.103448  |       9 |
| FP           | card_entry_mode  | MANUAL_KEYED | 0.22      |      66 |
| FP           | card_entry_mode  | QR_CODE      | 0.18      |      54 |
| FP           | card_entry_mode  | CONTACTLESS  | 0.16      |      48 |
| FP           | card_entry_mode  | MAGSTRIPE    | 0.153333  |      46 |
| FP           | card_entry_mode  | CHIP         | 0.153333  |      46 |
| FP           | card_entry_mode  | TOKEN_IN_APP | 0.133333  |      40 |

## 4. Cách đọc báo cáo

- **FP**: model cảnh báo fraud nhưng thực tế không fraud. Nhóm này gây tải review.
- **FN**: model bỏ sót fraud. Nhóm này nguy hiểm nhất về nghiệp vụ.
- Cần xem FN có đặc điểm gì: amount thấp hơn, entropy bình thường hơn, hoặc thiếu tín hiệu bất thường.
- Cần xem FP có tập trung vào kênh/thời điểm/quốc gia nào không để cân nhắc rule hoặc threshold phụ.


---

# Threshold Policy Report

## 1. Review capacity scenarios

|   review_capacity |   selected_threshold_from_validation |   validation_precision |   validation_recall |   validation_f2 |   validation_review_rate |   test_precision |   test_recall |   test_f1 |   test_f2 |   test_review_rate |   test_tp |   test_fp |   test_fn |   test_tn |
|------------------:|-------------------------------------:|-----------------------:|--------------------:|----------------:|-------------------------:|-----------------:|--------------:|----------:|----------:|-------------------:|----------:|----------:|----------:|----------:|
|              0.03 |                                 0.73 |                 0.5751 |              0.6587 |          0.6401 |                   0.0289 |           0.5726 |        0.7216 |    0.6385 |    0.6859 |             0.0326 |       280 |       209 |       108 |     14403 |
|              0.05 |                                 0.66 |                 0.4972 |              0.7037 |          0.6497 |                   0.0357 |           0.5008 |        0.7758 |    0.6087 |    0.699  |             0.0401 |       301 |       300 |        87 |     14312 |
|              0.1  |                                 0.66 |                 0.4972 |              0.7037 |          0.6497 |                   0.0357 |           0.5008 |        0.7758 |    0.6087 |    0.699  |             0.0401 |       301 |       300 |        87 |     14312 |

## 2. Cách đọc

- `review_capacity`: năng lực review tối đa giả định của đội vận hành.
- Threshold được chọn trên validation, sau đó áp dụng sang test.
- Nếu review capacity tăng, recall thường tăng nhưng precision có thể giảm.
- Chọn policy phụ thuộc vào chi phí bỏ sót fraud và chi phí review nhầm.


## Final conclusion


Final model được chọn là CatBoost tuned `cat_tune_depth7_l2_8_lr003_iter700`.

Model này được chọn vì có PR-AUC/F2 tốt nhất, recall cao, review rate vẫn dưới ngưỡng vận hành 5%, 
và feature importance vẫn khớp với các insight nghiệp vụ chính như mcc entropy, night behavior, amount anomaly, spending trend và country diversity.

Logistic Regression vẫn được giữ làm baseline giải thích được, còn CatBoost là model advanced dùng cho hiệu năng vận hành.
