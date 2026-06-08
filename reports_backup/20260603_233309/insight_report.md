# Fraud Detection - Hidden Insight Mining Report

## 1. Tổng quan

- Fraud rate toàn bộ dataset: **0.0255 (2.55%)**
- Mục tiêu giai đoạn này là tìm segment/rule có fraud rate cao hơn baseline, sau đó chuyển các pattern có ý nghĩa thành feature cho model.

## 2. Segment nghiệp vụ nổi bật

| segment_name           |   support |   support_rate |   fraud_count |   fraud_rate |     lift | strength   | description                                                                                                                         |
|:-----------------------|----------:|---------------:|--------------:|-------------:|---------:|:-----------|:------------------------------------------------------------------------------------------------------------------------------------|
| very_low_mcc_entropy   |      5017 |        0.05017 |           454 |    0.0904923 | 3.55011  | STRONG     | Nhóm cực thấp về mcc_entropy_30d, thường cần soi kỹ hơn.                                                                            |
| low_mcc_entropy        |     10012 |        0.10012 |           854 |    0.0852976 | 3.34632  | STRONG     | User có lịch sử MCC ít đa dạng. Đây là tín hiệu behavioral anomaly vì giao dịch mới có thể lệch khỏi hành vi ngành hàng quen thuộc. |
| night_unusual          |      3417 |        0.03417 |           255 |    0.0746269 | 2.92769  | MEDIUM     | Giao dịch ban đêm nhưng user có lịch sử night_ratio_30d thấp. Đây là tín hiệu bất thường về thời gian.                              |
| low_night_ratio        |     10003 |        0.10003 |           742 |    0.0741777 | 2.91007  | MEDIUM     | User gần như không có thói quen giao dịch ban đêm.                                                                                  |
| high_max_amount_30d    |     10000 |        0.1     |           663 |    0.0663    | 2.60102  | MEDIUM     | User có max_amount_30d thuộc nhóm rất cao.                                                                                          |
| low_country_diversity  |     12114 |        0.12114 |           748 |    0.0617467 | 2.42239  | MEDIUM     | User có lịch sử giao dịch ở ít quốc gia.                                                                                            |
| has_chargeback_history |     15016 |        0.15016 |           433 |    0.0288359 | 1.13126  | LOW_SIGNAL | User có lịch sử chargeback trong 365 ngày.                                                                                          |
| card_not_present       |     65166 |        0.65166 |          1714 |    0.0263021 | 1.03186  | LOW_SIGNAL | Giao dịch không có thẻ vật lý, thường rủi ro hơn trong fraud online.                                                                |
| night_transaction      |     33235 |        0.33235 |           869 |    0.0261471 | 1.02578  | LOW_SIGNAL | Giao dịch hiện tại xảy ra ban đêm.                                                                                                  |
| tokenised_transaction  |     45024 |        0.45024 |          1132 |    0.0251421 | 0.986353 | LOW_SIGNAL | Giao dịch tokenised.                                                                                                                |
| high_current_amount    |     10000 |        0.1     |           251 |    0.0251    | 0.9847   | LOW_SIGNAL | Giao dịch hiện tại có amount thuộc nhóm cao.                                                                                        |
| high_ip_score          |     10008 |        0.10008 |           247 |    0.0246803 | 0.968233 | LOW_SIGNAL | IP risk score thuộc nhóm cao.                                                                                                       |
| high_amount_to_mean    |     10000 |        0.1     |           215 |    0.0215    | 0.843468 | LOW_SIGNAL | Amount hiện tại cao hơn nhiều so với mean_amount_30d. Đây là tín hiệu amount anomaly.                                               |
| cross_border           |     19941 |        0.19941 |           414 |    0.0207612 | 0.814486 | LOW_SIGNAL | Giao dịch xuyên biên giới.                                                                                                          |
| high_device_diversity  |     12414 |        0.12414 |           184 |    0.014822  | 0.581482 | LOW_SIGNAL | User dùng nhiều thiết bị trong 30 ngày. Có thể liên quan account takeover hoặc account sharing.                                     |

## 3. Interaction nghiệp vụ ưu tiên

| rule                                                        |   support |   support_rate |   fraud_count |   fraud_rate |      lift | strength    | business_interpretation                                                                                                                                                     |
|:------------------------------------------------------------|----------:|---------------:|--------------:|-------------:|----------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| low_mcc_entropy AND low_night_ratio AND high_max_amount_30d |       108 |        0.00108 |            58 |    0.537037  | 21.0685   | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio AND card_not_present    |       664 |        0.00664 |           175 |    0.263554  | 10.3395   | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio                         |      1003 |        0.01003 |           249 |    0.248255  |  9.73932  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND high_max_amount_30d                     |      1008 |        0.01008 |           226 |    0.224206  |  8.79586  | VERY_STRONG | User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần.                   |
| low_night_ratio AND high_max_amount_30d                     |      1029 |        0.01029 |           196 |    0.190476  |  7.47258  | VERY_STRONG | Rule có tín hiệu thống kê. Cần kiểm tra support, lift, fraud_rate và ý nghĩa nghiệp vụ trước khi đưa vào model.                                                             |
| low_mcc_entropy AND card_not_present                        |      6495 |        0.06495 |           566 |    0.087144  |  3.41875  | STRONG      | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| low_night_ratio AND night_transaction                       |      3417 |        0.03417 |           255 |    0.0746269 |  2.92769  | MEDIUM      | Rule có tín hiệu thống kê. Cần kiểm tra support, lift, fraud_rate và ý nghĩa nghiệp vụ trước khi đưa vào model.                                                             |
| cross_border AND low_country_diversity                      |      2443 |        0.02443 |           131 |    0.0536226 |  2.10367  | MEDIUM      | Giao dịch xuyên biên giới trong khi user có lịch sử giao dịch ở ít quốc gia. Đây là tín hiệu country anomaly.                                                               |
| high_ip_score AND card_not_present                          |      6445 |        0.06445 |           170 |    0.026377  |  1.0348   | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| high_decline_rate AND card_not_present                      |      6572 |        0.06572 |            96 |    0.0146074 |  0.573065 | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| high_device_diversity AND card_not_present                  |      8101 |        0.08101 |           115 |    0.0141958 |  0.556916 | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |

## 4. Actionable rules nên chuyển thành feature

| rule                                                                                               |   support |   fraud_count |   fraud_rate |     lift | strength    | business_interpretation                                                                                                                                                     |
|:---------------------------------------------------------------------------------------------------|----------:|--------------:|-------------:|---------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| low_night_ratio_flag AND low_mcc_entropy_x_card_not_present                                        |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| card_not_present_flag AND low_mcc_entropy_x_low_night_ratio                                        |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present                           |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND card_not_present_flag                            |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND low_mcc_entropy_x_card_not_present               |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND card_not_present_flag AND low_mcc_entropy_x_low_night_ratio               |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present  |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND card_not_present_flag AND low_mcc_entropy_x_low_night_ratio               |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND card_not_present_flag AND low_mcc_entropy_x_card_not_present              |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present  |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| card_not_present_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio AND card_not_present                                           |       664 |           175 |     0.263554 | 10.3395  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag                                                      |      1003 |           249 |     0.248255 |  9.73932 | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_mcc_entropy_x_low_night_ratio                                         |      1003 |           249 |     0.248255 |  9.73932 | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio                                         |      1003 |           249 |     0.248255 |  9.73932 | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio                |      1003 |           249 |     0.248255 |  9.73932 | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio                                                                |      1003 |           249 |     0.248255 |  9.73932 | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| high_max_amount_30d_flag AND low_mcc_entropy_x_card_not_present                                    |       652 |           153 |     0.234663 |  9.20606 | VERY_STRONG | User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần.                   |
| card_not_present_flag AND low_mcc_entropy_x_high_max_amount_30d                                    |       652 |           153 |     0.234663 |  9.20606 | VERY_STRONG | User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần.                   |
| low_mcc_entropy_x_high_max_amount_30d AND low_mcc_entropy_x_card_not_present                       |       652 |           153 |     0.234663 |  9.20606 | VERY_STRONG | User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần.                   |

## 5. Numeric patterns nổi bật

| feature                | bin                      |   count |   fraud_count |   fraud_rate |   global_fraud_rate |   fraud_rate_diff |    lift |   min_value |   max_value |
|:-----------------------|:-------------------------|--------:|--------------:|-------------:|--------------------:|------------------:|--------:|------------:|------------:|
| mcc_entropy_30d        | (-0.001, 0.392]          |   10012 |           854 |    0.0852976 |             0.02549 |         0.0598076 | 3.34632 |       0     |       0.392 |
| night_ratio_30d        | (-0.001, 0.1]            |   10003 |           742 |    0.0741777 |             0.02549 |         0.0486877 | 2.91007 |       0     |       0.1   |
| max_amount_30d         | (225071.135, 249999.11]  |   10000 |           663 |    0.0663    |             0.02549 |         0.04081   | 2.60102 |  225074     |  249999     |
| distinct_countries_30d | (0.999, 3.0]             |   12114 |           748 |    0.0617467 |             0.02549 |         0.0362567 | 2.42239 |       1     |       3     |
| mcc_entropy_30d        | (0.392, 0.791]           |    9996 |           613 |    0.0613245 |             0.02549 |         0.0358345 | 2.40583 |       0.393 |       0.791 |
| night_unusual_score    | (0.704, 1.0]             |    9970 |           548 |    0.0549649 |             0.02549 |         0.0294749 | 2.15633 |       0.705 |       1     |
| std_amount_30d         | (0.999, 1.12]            |   10510 |           552 |    0.0525214 |             0.02549 |         0.0270314 | 2.06047 |       1     |       1.12  |
| night_ratio_30d        | (0.1, 0.2]               |   10018 |           504 |    0.0503094 |             0.02549 |         0.0248194 | 1.97369 |       0.101 |       0.2   |
| max_amount_30d         | (200269.424, 225071.135] |   10000 |           496 |    0.0496    |             0.02549 |         0.02411   | 1.94586 |  200273     |  225071     |
| distinct_countries_30d | (3.0, 5.0]               |    7967 |           384 |    0.0481988 |             0.02549 |         0.0227088 | 1.89089 |       4     |       5     |
| std_amount_30d         | (1.12, 1.25]             |    9539 |           433 |    0.0453926 |             0.02549 |         0.0199026 | 1.7808  |       1.13  |       1.25  |
| decline_rate_30d       | (0.001, 0.067]           |   10322 |           426 |    0.0412711 |             0.02549 |         0.0157811 | 1.61911 |       0.002 |       0.067 |

## 6. Categorical patterns nổi bật

| feature                               | value          |   count |   fraud_count |   fraud_rate |   global_fraud_rate |   fraud_rate_diff |    lift |
|:--------------------------------------|:---------------|--------:|--------------:|-------------:|--------------------:|------------------:|--------:|
| low_mcc_entropy_x_low_night_ratio     | 1              |    1003 |           249 |    0.248255  |             0.02549 |        0.222765   | 9.73932 |
| low_mcc_entropy_x_high_max_amount_30d | 1              |    1008 |           226 |    0.224206  |             0.02549 |        0.198716   | 8.79586 |
| low_night_ratio_x_high_max_amount_30d | 1              |    1029 |           196 |    0.190476  |             0.02549 |        0.164986   | 7.47258 |
| low_mcc_entropy_x_card_not_present    | 1              |    6495 |           566 |    0.087144  |             0.02549 |        0.061654   | 3.41875 |
| night_unusual_group                   | very_high      |    3373 |           253 |    0.0750074 |             0.02549 |        0.0495174  | 2.94262 |
| night_ratio_group                     | very_low       |    9941 |           736 |    0.0740368 |             0.02549 |        0.0485468  | 2.90454 |
| cross_border_x_low_country_diversity  | 1              |    2443 |           131 |    0.0536226 |             0.02549 |        0.0281326  | 2.10367 |
| night_ratio_group                     | low            |   20112 |           883 |    0.0439041 |             0.02549 |        0.0184141  | 1.72241 |
| night_unusual_group                   | high           |   10064 |           407 |    0.0404412 |             0.02549 |        0.0149512  | 1.58655 |
| credit_util_risk_group                | low            |   15112 |           558 |    0.0369243 |             0.02549 |        0.0114343  | 1.44858 |
| decline_rate_risk_group               | low            |   24124 |           889 |    0.0368513 |             0.02549 |        0.0113613  | 1.44571 |
| amount_to_max_group                   | below_half_max |   31134 |          1124 |    0.036102  |             0.02549 |        0.010612   | 1.41632 |
| amount_to_max_group                   | near_max       |   28644 |           979 |    0.0341782 |             0.02549 |        0.00868819 | 1.34085 |

## 7. Khuyến nghị feature engineering


Dựa trên insight mining, nên ưu tiên tạo các nhóm feature sau:

1. **Behavioral anomaly features**
   - amount_to_mean_30d
   - amount_to_max_30d
   - night_unusual_score
   - txn_count_ratio_7d_30d

2. **MCC behavior features**
   - low_mcc_entropy_flag
   - mcc_entropy_30d
   - low_mcc_entropy_x_low_night_ratio
   - low_mcc_entropy_x_high_max_amount_30d

3. **Time anomaly features**
   - is_night_txn
   - night_ratio_30d
   - night_unusual_group
   - low_night_ratio_x_high_max_amount_30d

4. **Country/channel behavior**
   - cross_border_flag
   - low_country_diversity_flag
   - cross_border_x_low_country_diversity
   - payment_channel_x_merchant_country

5. **Interaction features cho Logistic Regression**
   - low_mcc_entropy_x_low_night_ratio
   - low_mcc_entropy_x_high_max_amount_30d
   - low_mcc_entropy_x_low_spending_trend
   - low_mcc_entropy_x_card_not_present
   - high_ip_score_x_card_not_present


## 8. Lưu ý leakage


Các biến lịch sử như `mean_amount_30d`, `max_amount_30d`, `decline_rate_30d`,
`chargebacks_365d`, `txn_counts` cần đảm bảo được tính từ dữ liệu **trước thời điểm giao dịch hiện tại**.

Nếu các biến này bao gồm giao dịch hiện tại hoặc thông tin tương lai, model sẽ bị leakage.
