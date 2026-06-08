# Final Baseline Model Report - Logistic Regression

## 1. Model được chọn

Model baseline được chọn là **`lr_no_mcc`**.

Lý do chọn model này:
- Test PR-AUC: **0.5332**
- Test F2: **0.6098**
- Test Precision: **40.45%**
- Test Recall: **69.85%**
- Review rate: **4.47%**
- Số chiều sau preprocessing: **171**
- Feature categorical bị bỏ trong experiment: **`mcc`**

Kết quả experiment cho thấy bản bỏ `mcc` không làm giảm hiệu năng, ngược lại còn cải thiện PR-AUC/F2 và giảm số chiều, nên phù hợp hơn để chọn làm baseline chính.

## 2. So sánh các experiment

| experiment_name    | dropped_categorical   |   min_frequency |   n_preprocessed_features |   selected_threshold |   test_pr_auc |   test_precision |   test_recall |   test_f1 |   test_f2 |   test_review_rate |   test_tp |   test_fp |   test_fn |   test_tn |
|:-------------------|:----------------------|----------------:|--------------------------:|---------------------:|--------------:|-----------------:|--------------:|----------:|----------:|-------------------:|----------:|----------:|----------:|----------:|
| lr_no_mcc          | mcc                   |              50 |                       171 |                 0.77 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_100 | nan                   |             100 |                       172 |                 0.77 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_200 | nan                   |             200 |                       172 |                 0.77 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_50  | nan                   |              50 |                       355 |                 0.76 |      0.448793 |         0.359249 |      0.690722 |  0.472663 |  0.583116 |          0.0497333 |       268 |       478 |       120 |     14134 |

## 3. Validation và Test metrics của model cuối

### Validation
| split                     |   threshold |   rows |   fraud_count |   fraud_rate |   roc_auc |   pr_auc |   precision |   recall |       f1 |       f2 |   balanced_accuracy |   review_rate |   predicted_positive |    tn |   fp |   fn |   tp |
|:--------------------------|------------:|-------:|--------------:|-------------:|----------:|---------:|------------:|---------:|---------:|---------:|--------------------:|--------------:|---------------------:|------:|-----:|-----:|-----:|
| validation_best_threshold |        0.77 |  15000 |           378 |       0.0252 |  0.871787 | 0.506007 |      0.3769 | 0.656085 | 0.478764 | 0.571429 |            0.814022 |     0.0438667 |                  658 | 14212 |  410 |  130 |  248 |

### Test
| split   |   threshold |   rows |   fraud_count |   fraud_rate |   roc_auc |   pr_auc |   precision |   recall |       f1 |       f2 |   balanced_accuracy |   review_rate |   predicted_positive |    tn |   fp |   fn |   tp |
|:--------|------------:|-------:|--------------:|-------------:|----------:|---------:|------------:|---------:|---------:|---------:|--------------------:|--------------:|---------------------:|------:|-----:|-----:|-----:|
| test    |        0.77 |  15000 |           388 |    0.0258667 |  0.895196 | 0.533238 |    0.404478 | 0.698454 | 0.512287 | 0.609811 |            0.835574 |     0.0446667 |                  670 | 14213 |  399 |  117 |  271 |

## 4. Diễn giải nghiệp vụ từ confusion matrix

Trên tập test có **15,000** giao dịch, trong đó có **388** fraud thật.
Model cảnh báo **670** giao dịch cần review (**4.47%** tổng giao dịch).
Trong các giao dịch bị cảnh báo, model bắt đúng **271** fraud và báo nhầm **399** giao dịch thường.
Model bỏ sót **117** fraud và xác định đúng **14,213** giao dịch thường.

## 5. Tổng hợp coefficient theo nhóm insight

| insight_group                    |   feature_count |   total_abs_coefficient |   mean_abs_coefficient |   max_abs_coefficient |
|:---------------------------------|----------------:|------------------------:|-----------------------:|----------------------:|
| Time behavior anomaly            |              21 |                8.11006  |              0.386194  |              1.16065  |
| Payment / authentication context |              68 |                6.93616  |              0.102002  |              0.337189 |
| Amount anomaly                   |               9 |                4.34452  |              0.482724  |              0.884215 |
| MCC behavior anomaly             |               8 |                3.61155  |              0.451443  |              0.710432 |
| Country / cross-border behavior  |              24 |                2.9631   |              0.123462  |              0.419856 |
| Risk history / technical risk    |              17 |                2.49316  |              0.146657  |              0.347736 |
| Transaction velocity             |              12 |                1.58893  |              0.132411  |              0.29622  |
| Other context                    |              10 |                0.696062 |              0.0696062 |              0.210836 |
| Spending trend anomaly           |               2 |                0.647348 |              0.323674  |              0.537134 |

## 6. Top feature làm tăng fraud risk

| readable_feature                                     |   coefficient | insight_group                   | business_meaning                                                                                                             |
|:-----------------------------------------------------|--------------:|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| night_unusual_group=not_night                        |      1.16065  | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| amount_to_max_group=below_half_max                   |      0.884215 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| low_mcc_entropy_x_high_max_amount_30d                |      0.678902 | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |
| night_transaction_flag                               |      0.66631  | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| amount_z_30d                                         |      0.664834 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| low_mcc_entropy_x_low_spending_trend                 |      0.661424 | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |
| low_night_ratio_x_high_max_amount_30d                |      0.611161 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| low_mcc_entropy_x_low_night_ratio_x_card_not_present |      0.424502 | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |
| low_mcc_entropy_x_low_night_ratio                    |      0.415451 | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |
| time_period=late_evening                             |      0.379148 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| amount_to_max_group=near_max                         |      0.367657 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| low_country_diversity_flag                           |      0.301294 | Country / cross-border behavior | Hành vi giao dịch theo quốc gia và mức độ đa dạng quốc gia của user.                                                         |
| time_period=night                                    |      0.287162 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| high_max_amount_30d_flag                             |      0.266724 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| very_low_mcc_entropy_flag                            |      0.25422  | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |

## 7. Top feature làm giảm fraud risk

| readable_feature                   |   coefficient | insight_group                   | business_meaning                                                                                                             |
|:-----------------------------------|--------------:|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| amount_to_max_group=extreme        |     -0.802835 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| mcc_entropy_30d                    |     -0.710432 | MCC behavior anomaly            | User có lịch sử ngành hàng/MCC ít đa dạng. Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng. |
| night_unusual_group=very_high      |     -0.594836 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| night_ratio_30d                    |     -0.588195 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| amount_to_max_group=much_above_max |     -0.56501  | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| spending_trend                     |     -0.537134 | Spending trend anomaly          | Xu hướng chi tiêu thấp hoặc giảm có thể phản ánh hành vi bất thường khi kết hợp với các tín hiệu khác.                       |
| time_period=evening                |     -0.535149 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| distinct_countries_30d             |     -0.419856 | Country / cross-border behavior | Hành vi giao dịch theo quốc gia và mức độ đa dạng quốc gia của user.                                                         |
| amount_to_max_group=above_max      |     -0.415628 | Amount anomaly                  | Giao dịch hiện tại lệch so với lịch sử amount của user.                                                                      |
| night_unusual_group=high           |     -0.413851 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| night_unusual_group=medium         |     -0.412692 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| night_ratio_group=medium           |     -0.384532 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| low_night_ratio_flag               |     -0.352326 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |
| ip_score_risk_group=low            |     -0.347736 | Risk history / technical risk   | Các tín hiệu lịch sử rủi ro hoặc tín hiệu kỹ thuật như IP risk, decline rate, chargeback.                                    |
| time_period=afternoon              |     -0.346522 | Time behavior anomaly           | Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm so với lịch sử của user.                                     |

## 8. Các coefficient khớp với insight giai đoạn 3

| readable_feature                                     |   coefficient | direction     | insight_group                   | expected_direction                                                                                                                        | stage3_evidence                                                                                     |
|:-----------------------------------------------------|--------------:|:--------------|:--------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| night_unusual_group=not_night                        |      1.16065  | increase_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| amount_to_max_group=below_half_max                   |      0.884215 | increase_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| amount_to_max_group=extreme                          |     -0.802835 | decrease_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| mcc_entropy_30d                                      |     -0.710432 | decrease_risk | MCC behavior anomaly            | mcc_entropy_30d cao thường giảm risk; low_mcc_entropy flag/interaction thường tăng risk.                                                  | Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy là các segment có lift cao và ổn định. |
| low_mcc_entropy_x_high_max_amount_30d                |      0.678902 | increase_risk | MCC behavior anomaly            | mcc_entropy_30d cao thường giảm risk; low_mcc_entropy flag/interaction thường tăng risk.                                                  | Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy là các segment có lift cao và ổn định. |
| night_transaction_flag                               |      0.66631  | increase_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| amount_z_30d                                         |      0.664834 | increase_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| low_mcc_entropy_x_low_spending_trend                 |      0.661424 | increase_risk | MCC behavior anomaly            | mcc_entropy_30d cao thường giảm risk; low_mcc_entropy flag/interaction thường tăng risk.                                                  | Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy là các segment có lift cao và ổn định. |
| low_night_ratio_x_high_max_amount_30d                |      0.611161 | increase_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| night_unusual_group=very_high                        |     -0.594836 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| night_ratio_30d                                      |     -0.588195 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| amount_to_max_group=much_above_max                   |     -0.56501  | decrease_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| spending_trend                                       |     -0.537134 | decrease_risk | Spending trend anomaly          | spending_trend cao thường giảm risk; low_spending_trend flag/interaction tăng risk.                                                       | Giai đoạn 3 cho thấy spending_trend thấp có lift ổn định.                                           |
| time_period=evening                                  |     -0.535149 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| low_mcc_entropy_x_low_night_ratio_x_card_not_present |      0.424502 | increase_risk | MCC behavior anomaly            | mcc_entropy_30d cao thường giảm risk; low_mcc_entropy flag/interaction thường tăng risk.                                                  | Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy là các segment có lift cao và ổn định. |
| distinct_countries_30d                               |     -0.419856 | decrease_risk | Country / cross-border behavior | distinct_countries_30d cao thường giảm risk; low_country_diversity hoặc cross-border interaction có thể tăng risk.                        | Giai đoạn 3 cho thấy low_country_diversity là segment có lift cao.                                  |
| amount_to_max_group=above_max                        |     -0.415628 | decrease_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| low_mcc_entropy_x_low_night_ratio                    |      0.415451 | increase_risk | MCC behavior anomaly            | mcc_entropy_30d cao thường giảm risk; low_mcc_entropy flag/interaction thường tăng risk.                                                  | Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy là các segment có lift cao và ổn định. |
| night_unusual_group=high                             |     -0.413851 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| night_unusual_group=medium                           |     -0.412692 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| night_ratio_group=medium                             |     -0.384532 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| time_period=late_evening                             |      0.379148 | increase_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| amount_to_max_group=near_max                         |      0.367657 | increase_risk | Amount anomaly                  | amount_z_30d cao hoặc các interaction với high_max_amount_30d thường làm tăng risk.                                                       | Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group có lift cao hơn baseline.           |
| low_night_ratio_flag                                 |     -0.352326 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |
| time_period=afternoon                                |     -0.346522 | decrease_risk | Time behavior anomaly           | night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; night transaction hoặc low_night_ratio interaction có thể tăng risk. | Giai đoạn 3 cho thấy low_night_ratio và night_unusual có lift cao trên train và validation.         |

## 9. Kết luận giải thích model


Model baseline không chỉ dựa vào category cụ thể mà học được các nhóm behavioral anomaly đã khai phá ở giai đoạn 3:

- `mcc_entropy_30d` có coefficient âm: entropy càng cao thì risk càng giảm, nghĩa là entropy thấp làm fraud risk tăng.
- Các interaction như `low_mcc_entropy_x_high_max_amount_30d`, `low_mcc_entropy_x_low_spending_trend`, `low_mcc_entropy_x_low_night_ratio` có coefficient dương, khớp với insight rằng fraud tăng mạnh khi nhiều tín hiệu bất thường cùng xuất hiện.
- `amount_z_30d` có coefficient dương, cho thấy amount hiện tại lệch khỏi lịch sử user làm tăng risk.
- `night_ratio_30d` và `spending_trend` có coefficient âm, khớp với insight rằng nhóm night ratio thấp hoặc spending trend thấp có risk cao hơn.


## 10. Lưu ý khi diễn giải coefficient


Coefficient của Logistic Regression thể hiện tác động lên log-odds của fraud sau khi đã xét đồng thời các feature khác.
Vì vậy không nên diễn giải là quan hệ nhân quả. Một coefficient dương nghĩa là feature đó liên quan đến việc model tăng fraud risk,
còn coefficient âm nghĩa là feature đó liên quan đến việc model giảm fraud risk.

Các one-hot category như `currency`, `merchant_country`, `time_period` nên được diễn giải cẩn thận vì chúng phụ thuộc vào category tham chiếu
và các feature khác trong model.
