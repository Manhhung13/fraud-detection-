# Fraud Detection - Hidden Insight Mining Report

## 1. Tổng quan

- Fraud rate toàn bộ dataset: **0.0255 (2.55%)**
- Mục tiêu giai đoạn này là tìm segment/rule có fraud rate cao hơn baseline, sau đó chuyển các pattern có ý nghĩa thành feature cho model.

## 2. Segment nghiệp vụ nổi bật

| segment_name           |   support |   support_rate |   fraud_count |   fraud_rate |     lift | strength   | description                                                                                                                         |
|:-----------------------|----------:|---------------:|--------------:|-------------:|---------:|:-----------|:------------------------------------------------------------------------------------------------------------------------------------|
| very_low_mcc_entropy   |      3510 |      0.0501429 |           339 |    0.0965812 | 3.79175  | STRONG     | Nhóm cực thấp về mcc_entropy_30d, thường cần soi kỹ hơn.                                                                            |
| low_mcc_entropy        |      7009 |      0.100129  |           625 |    0.0891711 | 3.50083  | STRONG     | User có lịch sử MCC ít đa dạng. Đây là tín hiệu behavioral anomaly vì giao dịch mới có thể lệch khỏi hành vi ngành hàng quen thuộc. |
| low_night_ratio        |      7038 |      0.100543  |           511 |    0.0726059 | 2.85048  | MEDIUM     | User gần như không có thói quen giao dịch ban đêm.                                                                                  |
| night_unusual          |      2401 |      0.0343    |           171 |    0.0712203 | 2.79609  | MEDIUM     | Giao dịch ban đêm nhưng user có lịch sử night_ratio_30d thấp. Đây là tín hiệu bất thường về thời gian.                              |
| high_max_amount_30d    |      7000 |      0.1       |           439 |    0.0627143 | 2.46214  | MEDIUM     | User có max_amount_30d thuộc nhóm rất cao.                                                                                          |
| low_country_diversity  |      8464 |      0.120914  |           525 |    0.0620274 | 2.43518  | MEDIUM     | User có lịch sử giao dịch ở ít quốc gia.                                                                                            |
| has_chargeback_history |     10439 |      0.149129  |           306 |    0.0293132 | 1.15082  | LOW_SIGNAL | User có lịch sử chargeback trong 365 ngày.                                                                                          |
| card_not_present       |     45672 |      0.652457  |          1201 |    0.0262962 | 1.03238  | LOW_SIGNAL | Giao dịch không có thẻ vật lý, thường rủi ro hơn trong fraud online.                                                                |
| night_transaction      |     23339 |      0.333414  |           611 |    0.0261794 | 1.02779  | LOW_SIGNAL | Giao dịch hiện tại xảy ra ban đêm.                                                                                                  |
| tokenised_transaction  |     31516 |      0.450229  |           784 |    0.0248763 | 0.976634 | LOW_SIGNAL | Giao dịch tokenised.                                                                                                                |
| high_current_amount    |      7000 |      0.1       |           173 |    0.0247143 | 0.970275 | LOW_SIGNAL | Giao dịch hiện tại có amount thuộc nhóm cao.                                                                                        |
| high_ip_score          |      7001 |      0.100014  |           172 |    0.0245679 | 0.964529 | LOW_SIGNAL | IP risk score thuộc nhóm cao.                                                                                                       |
| high_amount_to_mean    |      7000 |      0.1       |           147 |    0.021     | 0.824453 | LOW_SIGNAL | Amount hiện tại cao hơn nhiều so với mean_amount_30d. Đây là tín hiệu amount anomaly.                                               |
| cross_border           |     13963 |      0.199471  |           280 |    0.020053  | 0.787274 | LOW_SIGNAL | Giao dịch xuyên biên giới.                                                                                                          |
| high_device_diversity  |      8633 |      0.123329  |           125 |    0.0144793 | 0.568454 | LOW_SIGNAL | User dùng nhiều thiết bị trong 30 ngày. Có thể liên quan account takeover hoặc account sharing.                                     |

## 3. Interaction nghiệp vụ ưu tiên

| rule                                                        |   support |   support_rate |   fraud_count |   fraud_rate |      lift | strength    | business_interpretation                                                                                                                                                     |
|:------------------------------------------------------------|----------:|---------------:|--------------:|-------------:|----------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| low_mcc_entropy AND low_night_ratio AND high_max_amount_30d |        74 |     0.00105714 |            39 |    0.527027  | 20.6909   | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio AND card_not_present    |       479 |     0.00684286 |           128 |    0.267223  | 10.4911   | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio                         |       696 |     0.00994286 |           181 |    0.260057  | 10.2098   | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_spending_trend                      |       733 |     0.0104714  |           175 |    0.238745  |  9.37305  | VERY_STRONG | User có ngành hàng giao dịch hẹp và xu hướng chi tiêu thấp hoặc giảm. Nếu phát sinh giao dịch mới lệch khỏi hành vi này thì rủi ro tăng.                                    |
| low_mcc_entropy AND high_max_amount_30d                     |       713 |     0.0101857  |           162 |    0.227209  |  8.92015  | VERY_STRONG | User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần.                   |
| low_night_ratio AND high_max_amount_30d                     |       728 |     0.0104     |           124 |    0.17033   |  6.68709  | VERY_STRONG | Rule có tín hiệu thống kê. Cần kiểm tra support, lift, fraud_rate và ý nghĩa nghiệp vụ trước khi đưa vào model.                                                             |
| low_mcc_entropy AND card_not_present                        |      4570 |     0.0652857  |           419 |    0.0916849 |  3.59952  | STRONG      | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| low_night_ratio AND night_transaction                       |      2401 |     0.0343     |           171 |    0.0712203 |  2.79609  | MEDIUM      | Rule có tín hiệu thống kê. Cần kiểm tra support, lift, fraud_rate và ý nghĩa nghiệp vụ trước khi đưa vào model.                                                             |
| cross_border AND low_country_diversity                      |      1692 |     0.0241714  |            86 |    0.0508274 |  1.99547  | WEAK        | Giao dịch xuyên biên giới trong khi user có lịch sử giao dịch ở ít quốc gia. Đây là tín hiệu country anomaly.                                                               |
| high_ip_score AND card_not_present                          |      4474 |     0.0639143  |           113 |    0.025257  |  0.991583 | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| high_device_diversity AND card_not_present                  |      5657 |     0.0808143  |            78 |    0.0137882 |  0.541321 | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |
| high_decline_rate AND card_not_present                      |      4577 |     0.0653857  |            62 |    0.013546  |  0.531811 | LOW_SIGNAL  | Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, đặc biệt khi đi kèm tín hiệu bất thường khác.                                          |

## 4. Actionable rules nên chuyển thành feature

| rule                                                                                               |   support |   fraud_count |   fraud_rate |     lift | strength    | business_interpretation                                                                                                                                                     |
|:---------------------------------------------------------------------------------------------------|----------:|--------------:|-------------:|---------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| low_night_ratio_flag AND low_mcc_entropy_x_card_not_present                                        |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| card_not_present_flag AND low_mcc_entropy_x_low_night_ratio                                        |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present                           |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND card_not_present_flag                            |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND low_mcc_entropy_x_card_not_present               |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND card_not_present_flag AND low_mcc_entropy_x_low_night_ratio               |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present  |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND card_not_present_flag AND low_mcc_entropy_x_low_night_ratio               |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND card_not_present_flag AND low_mcc_entropy_x_card_not_present              |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present  |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| card_not_present_flag AND low_mcc_entropy_x_low_night_ratio AND low_mcc_entropy_x_card_not_present |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio AND card_not_present                                           |       479 |           128 |     0.267223 | 10.4911  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag                                                      |       696 |           181 |     0.260057 | 10.2098  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_mcc_entropy_x_low_night_ratio                                         |       696 |           181 |     0.260057 | 10.2098  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio                                         |       696 |           181 |     0.260057 | 10.2098  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy_flag AND low_night_ratio_flag AND low_mcc_entropy_x_low_night_ratio                |       696 |           181 |     0.260057 | 10.2098  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_mcc_entropy AND low_night_ratio                                                                |       696 |           181 |     0.260057 | 10.2098  | VERY_STRONG | User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. Đây là nhóm behavioral anomaly quan trọng. |
| low_spending_trend_flag AND low_mcc_entropy_x_card_not_present                                     |       481 |           121 |     0.251559 |  9.87613 | VERY_STRONG | User có ngành hàng giao dịch hẹp và xu hướng chi tiêu thấp hoặc giảm. Nếu phát sinh giao dịch mới lệch khỏi hành vi này thì rủi ro tăng.                                    |
| card_not_present_flag AND low_mcc_entropy_x_low_spending_trend                                     |       481 |           121 |     0.251559 |  9.87613 | VERY_STRONG | User có ngành hàng giao dịch hẹp và xu hướng chi tiêu thấp hoặc giảm. Nếu phát sinh giao dịch mới lệch khỏi hành vi này thì rủi ro tăng.                                    |
| low_mcc_entropy_x_low_spending_trend AND low_mcc_entropy_x_card_not_present                        |       481 |           121 |     0.251559 |  9.87613 | VERY_STRONG | User có ngành hàng giao dịch hẹp và xu hướng chi tiêu thấp hoặc giảm. Nếu phát sinh giao dịch mới lệch khỏi hành vi này thì rủi ro tăng.                                    |

## 5. Numeric patterns nổi bật

| feature                | bin                      |   count |   fraud_count |   fraud_rate |   global_fraud_rate |   fraud_rate_diff |    lift |   min_value |   max_value |
|:-----------------------|:-------------------------|--------:|--------------:|-------------:|--------------------:|------------------:|--------:|------------:|------------:|
| mcc_entropy_30d        | (-inf, 0.393]            |    7009 |           625 |    0.0891711 |           0.0254714 |         0.0636996 | 3.50083 |       0     |       0.393 |
| night_ratio_30d        | (-inf, 0.101]            |    7038 |           511 |    0.0726059 |           0.0254714 |         0.0471344 | 2.85048 |       0     |       0.101 |
| spending_trend         | (-inf, 0.584]            |    7002 |           463 |    0.066124  |           0.0254714 |         0.0406525 | 2.59601 |       0.1   |       0.584 |
| max_amount_30d         | (225176.777, inf]        |    7000 |           439 |    0.0627143 |           0.0254714 |         0.0372429 | 2.46214 |  225178     |  249999     |
| distinct_countries_30d | (-inf, 3.0]              |    8464 |           525 |    0.0620274 |           0.0254714 |         0.036556  | 2.43518 |       1     |       3     |
| mcc_entropy_30d        | (0.393, 0.788]           |    6992 |           413 |    0.0590675 |           0.0254714 |         0.0335961 | 2.31897 |       0.394 |       0.788 |
| night_unusual_score    | (0.705, inf]             |    6991 |           381 |    0.0544986 |           0.0254714 |         0.0290272 | 2.1396  |       0.706 |       1     |
| max_amount_30d         | (200126.978, 225176.777] |    7000 |           370 |    0.0528571 |           0.0254714 |         0.0273857 | 2.07515 |  200128     |  225177     |
| std_amount_30d         | (-inf, 1.12]             |    7397 |           389 |    0.0525889 |           0.0254714 |         0.0271175 | 2.06462 |       1     |       1.12  |
| night_ratio_30d        | (0.101, 0.201]           |    6999 |           362 |    0.0517217 |           0.0254714 |         0.0262502 | 2.03058 |       0.102 |       0.201 |
| spending_trend         | (0.584, 1.073]           |    7008 |           331 |    0.0472317 |           0.0254714 |         0.0217603 | 1.8543  |       0.585 |       1.073 |
| distinct_countries_30d | (3.0, 6.0]               |    8280 |           363 |    0.0438406 |           0.0254714 |         0.0183692 | 1.72117 |       4     |       6     |
| std_amount_30d         | (1.12, 1.25]             |    6710 |           292 |    0.0435171 |           0.0254714 |         0.0180457 | 1.70847 |       1.13  |       1.25  |
| decline_rate_30d       | (-inf, 0.067]            |    7161 |           309 |    0.0431504 |           0.0254714 |         0.017679  | 1.69407 |       0.002 |       0.067 |

## 6. Categorical patterns nổi bật

| feature                               | value          |   count |   fraud_count |   fraud_rate |   global_fraud_rate |   fraud_rate_diff |     lift |
|:--------------------------------------|:---------------|--------:|--------------:|-------------:|--------------------:|------------------:|---------:|
| low_mcc_entropy_x_low_night_ratio     | 1              |     696 |           181 |    0.260057  |           0.0254714 |        0.234586   | 10.2098  |
| low_mcc_entropy_x_low_spending_trend  | 1              |     733 |           175 |    0.238745  |           0.0254714 |        0.213273   |  9.37305 |
| low_mcc_entropy_x_high_max_amount_30d | 1              |     713 |           162 |    0.227209  |           0.0254714 |        0.201738   |  8.92015 |
| low_night_ratio_x_high_max_amount_30d | 1              |     728 |           124 |    0.17033   |           0.0254714 |        0.144858   |  6.68709 |
| low_mcc_entropy_x_card_not_present    | 1              |    4570 |           419 |    0.0916849 |           0.0254714 |        0.0662135  |  3.59952 |
| night_ratio_group                     | very_low       |    6924 |           503 |    0.0726459 |           0.0254714 |        0.0471744  |  2.85205 |
| night_unusual_group                   | very_high      |    2339 |           167 |    0.071398  |           0.0254714 |        0.0459266  |  2.80306 |
| cross_border_x_low_country_diversity  | 1              |    1692 |            86 |    0.0508274 |           0.0254714 |        0.025356   |  1.99547 |
| night_ratio_group                     | low            |   14006 |           622 |    0.0444095 |           0.0254714 |        0.0189381  |  1.7435  |
| night_unusual_group                   | high           |    7144 |           294 |    0.0411534 |           0.0254714 |        0.015682   |  1.61567 |
| decline_rate_risk_group               | low            |   16873 |           630 |    0.0373378 |           0.0254714 |        0.0118663  |  1.46587 |
| credit_util_risk_group                | low            |   10611 |           381 |    0.0359061 |           0.0254714 |        0.0104347  |  1.40966 |
| amount_to_max_group                   | below_half_max |   21845 |           777 |    0.0355688 |           0.0254714 |        0.0100974  |  1.39642 |
| amount_to_max_group                   | near_max       |   19962 |           693 |    0.034716  |           0.0254714 |        0.00924453 |  1.36294 |

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
