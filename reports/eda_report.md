# Fraud Detection - Data Understanding Report

## 1. Dataset Overview

- Số dòng: **100000**
- Số cột: **36**
- Số dòng duplicate: **0**
- Tổng missing values: **0**
- Fraud rate: **0.0255 (2.55%)**

## 2. Target Distribution

|   fraud |   count |    rate |
|--------:|--------:|--------:|
|       0 |   97451 | 0.97451 |
|       1 |    2549 | 0.02549 |

## 3. Column Overview

| column                 | dtype   |   non_null_count |   null_count |   null_rate |   unique_count |   unique_rate | sample_values                                                                                                                                                                                                   |
|:-----------------------|:--------|-----------------:|-------------:|------------:|---------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| device_id              | str     |           100000 |            0 |           0 |         100000 |       1       | DEV-7726FB1A8A274C4B, DEV-68CD2988A8C345AC, DEV-C75EBE4E1BCB4362, DEV-DFE8A6350785435B, DEV-07B9EB6E54CC459C                                                                                                    |
| merchant_id            | str     |           100000 |            0 |           0 |         100000 |       1       | MID-2585C551D674, MID-CA3988467DD9, MID-9D430F327DF7, MID-6E0F7E9155DB, MID-0C99A14622F8                                                                                                                        |
| ip_risk                | str     |           100000 |            0 |           0 |         100000 |       1       | {'ip': '190.3.86.171', 'score': 0.1467}, {'ip': '79.201.116.230', 'score': 0.5243}, {'ip': '16.247.34.115', 'score': 0.4811}, {'ip': '23.130.177.181', 'score': 0.3218}, {'ip': '9.44.12.213', 'score': 0.1168} |
| max_amount_30d         | float64 |           100000 |            0 |           0 |          99827 |       0.99827 | 34046.16, 7146.37, 72447.86, 228799.51, 108426.28                                                                                                                                                               |
| amount                 | float64 |           100000 |            0 |           0 |          99765 |       0.99765 | 100606.12, 146953.67, 161096.38, 131491.0, 71265.99                                                                                                                                                             |
| local_timestamp        | str     |           100000 |            0 |           0 |          99371 |       0.99371 | 2025-03-19 10:23:03, 2025-05-16 00:35:42, 2025-04-10 03:03:24, 2025-03-28 02:39:22, 2025-05-18 12:14:09                                                                                                         |
| txn_counts             | str     |           100000 |            0 |           0 |          39665 |       0.39665 | (13, 21), (22, 256), (13, 29), (58, 375), (60, 266)                                                                                                                                                             |
| spending_trend         | float64 |           100000 |            0 |           0 |           4901 |       0.04901 | 1.46, 4.98, 4.514, 1.316, 3.805                                                                                                                                                                                 |
| card_activation_age    | int64   |           100000 |            0 |           0 |           4381 |       0.04381 | 4282, 3804, 2292, 2711, 2845                                                                                                                                                                                    |
| mcc_entropy_30d        | float64 |           100000 |            0 |           0 |           4001 |       0.04001 | 0.801, 0.651, 0.646, 1.638, 2.033                                                                                                                                                                               |
| mcc                    | int64   |           100000 |            0 |           0 |           3351 |       0.03351 | 7736, 3242, 5638, 7800, 5789                                                                                                                                                                                    |
| mean_amount_30d        | float64 |           100000 |            0 |           0 |           2252 |       0.02252 | 5.19, 6.18, 5.8, 5.29, 6.69                                                                                                                                                                                     |
| credit_util_today      | float64 |           100000 |            0 |           0 |           2001 |       0.02001 | 0.943, 1.269, 0.291, 1.202, 1.846                                                                                                                                                                               |
| std_amount_30d         | float64 |           100000 |            0 |           0 |           1494 |       0.01494 | 6.62, 35.5, 4.32, 3.84, 1.82                                                                                                                                                                                    |
| online_share_7d        | float64 |           100000 |            0 |           0 |           1001 |       0.01001 | 0.21, 0.665, 0.322, 0.233, 0.909                                                                                                                                                                                |
| night_ratio_30d        | float64 |           100000 |            0 |           0 |           1001 |       0.01001 | 0.084, 0.751, 0.637, 0.497, 0.831                                                                                                                                                                               |
| decline_rate_30d       | float64 |           100000 |            0 |           0 |            560 |       0.0056  | 0.271, 0.098, 0.092, 0.055, 0.201                                                                                                                                                                               |
| days_since_last_txn    | int64   |           100000 |            0 |           0 |            181 |       0.00181 | 100, 177, 34, 132, 162                                                                                                                                                                                          |
| distinct_merchants_7d  | int64   |           100000 |            0 |           0 |            120 |       0.0012  | 112, 3, 40, 21, 100                                                                                                                                                                                             |
| device_diversity_30d   | int64   |           100000 |            0 |           0 |             40 |       0.0004  | 27, 17, 11, 23, 16                                                                                                                                                                                              |
| distinct_countries_30d | int64   |           100000 |            0 |           0 |             25 |       0.00025 | 2, 9, 16, 3, 6                                                                                                                                                                                                  |
| merchant_country       | str     |           100000 |            0 |           0 |             20 |       0.0002  | PH, GB, US, JP, ID                                                                                                                                                                                              |
| currency               | str     |           100000 |            0 |           0 |             15 |       0.00015 | USD, CNY, EUR, THB, CAD                                                                                                                                                                                         |
| chargebacks_365d       | int64   |           100000 |            0 |           0 |             11 |       0.00011 | 3, 0, 7, 5, 1                                                                                                                                                                                                   |
| message_type           | str     |           100000 |            0 |           0 |              8 |       8e-05   | 0110-AuthRsp, 0420-RepeatRev, 0200-FinReq, 0400-Reversal, 0100-AuthReq                                                                                                                                          |
| auth_result            | str     |           100000 |            0 |           0 |              8 |       8e-05   | 3DS_FAIL, BIOMETRIC_PASS, CVV_FAIL, 3DS_PASS, CVV_PASS                                                                                                                                                          |
| payment_channel        | str     |           100000 |            0 |           0 |              7 |       7e-05   | ATM, WEB_BROWSER, MOBILE_APP, SMART_WATCH, API_SERVER                                                                                                                                                           |
| term_location          | str     |           100000 |            0 |           0 |              7 |       7e-05   | BRANCH_COUNTER, KIOSK, CALL_CENTER, POS_TERMINAL, MOBILE_APP                                                                                                                                                    |
| card_entry_mode        | str     |           100000 |            0 |           0 |              6 |       6e-05   | CHIP, TOKEN_IN_APP, MAGSTRIPE, QR_CODE, MANUAL_KEYED                                                                                                                                                            |
| auth_characteristics   | str     |           100000 |            0 |           0 |              6 |       6e-05   | FALLBACK_MAG, CRYPTOGRAM_PRESENT, E_COMMERCE, CARD_PRESENT, MOBILE_TOKEN                                                                                                                                        |

## 4. High Cardinality Columns

| column              | dtype   |   unique_count |   unique_rate | note                                           |
|:--------------------|:--------|---------------:|--------------:|:-----------------------------------------------|
| amount              | float64 |          99765 |       0.99765 | High cardinality - không nên one-hot trực tiếp |
| local_timestamp     | str     |          99371 |       0.99371 | High cardinality - không nên one-hot trực tiếp |
| mcc                 | int64   |           3351 |       0.03351 | High cardinality - không nên one-hot trực tiếp |
| ip_risk             | str     |         100000 |       1       | High cardinality - không nên one-hot trực tiếp |
| device_id           | str     |         100000 |       1       | High cardinality - không nên one-hot trực tiếp |
| merchant_id         | str     |         100000 |       1       | High cardinality - không nên one-hot trực tiếp |
| card_activation_age | int64   |           4381 |       0.04381 | High cardinality - không nên one-hot trực tiếp |
| mean_amount_30d     | float64 |           2252 |       0.02252 | High cardinality - không nên one-hot trực tiếp |
| std_amount_30d      | float64 |           1494 |       0.01494 | High cardinality - không nên one-hot trực tiếp |
| max_amount_30d      | float64 |          99827 |       0.99827 | High cardinality - không nên one-hot trực tiếp |
| txn_counts          | str     |          39665 |       0.39665 | High cardinality - không nên one-hot trực tiếp |
| online_share_7d     | float64 |           1001 |       0.01001 | High cardinality - không nên one-hot trực tiếp |
| night_ratio_30d     | float64 |           1001 |       0.01001 | High cardinality - không nên one-hot trực tiếp |
| mcc_entropy_30d     | float64 |           4001 |       0.04001 | High cardinality - không nên one-hot trực tiếp |
| credit_util_today   | float64 |           2001 |       0.02001 | High cardinality - không nên one-hot trực tiếp |
| spending_trend      | float64 |           4901 |       0.04901 | High cardinality - không nên one-hot trực tiếp |

## 5. Nhận xét ban đầu


- Dataset cần được kiểm tra mất cân bằng nhãn vì fraud thường chiếm tỷ lệ nhỏ.
- Các cột có cardinality cao như ID, device, merchant hoặc IP không nên one-hot trực tiếp.
- Các feature dạng timestamp nên được tách thành hour, day_of_week, is_weekend, is_night.
- Các feature lịch sử hành vi như mean_amount_30d, max_amount_30d, mcc_entropy_30d, night_ratio_30d cần được khai phá kỹ.
- Cần kiểm tra leakage với các biến được tính theo lịch sử 30 ngày/365 ngày.
