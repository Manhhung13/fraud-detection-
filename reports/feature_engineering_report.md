# Fraud Detection - Feature Engineering Report

## 1. Tổng quan output

| split      |   rows |   columns |   numeric_features |   categorical_features |   binary_features |   interaction_features |   all_model_features | target_col   |   fraud_count |   fraud_rate |
|:-----------|-------:|----------:|-------------------:|-----------------------:|------------------:|-----------------------:|---------------------:|:-------------|--------------:|-------------:|
| train      |  70000 |       115 |                 19 |                     21 |                10 |                      8 |                   58 | fraud        |          1783 |    0.0254714 |
| validation |  15000 |       115 |                 19 |                     21 |                10 |                      8 |                   58 | fraud        |           378 |    0.0252    |
| test       |  15000 |       115 |                 19 |                     21 |                10 |                      8 |                   58 | fraud        |           388 |    0.0258667 |

## 2. Numeric features

- `night_ratio_30d`
- `mcc_entropy_30d`
- `distinct_merchants_7d`
- `distinct_countries_30d`
- `device_diversity_30d`
- `decline_rate_30d`
- `chargebacks_365d`
- `credit_util_today`
- `spending_trend`
- `ip_score`
- `txn_count_7d`
- `txn_count_30d`
- `txn_count_ratio_7d_30d`
- `hour`
- `log_mean_amount_30d`
- `amount_to_mean_30d`
- `log_amount_to_max_30d`
- `amount_z_30d`
- `night_unusual_score`

## 3. Categorical features

- `currency`
- `payment_channel`
- `merchant_country`
- `mcc`
- `card_entry_mode`
- `auth_result`
- `pin_verif_method`
- `recurring_flag`
- `auth_characteristics`
- `message_type`
- `term_location`
- `day_name`
- `time_period`
- `night_ratio_group`
- `night_unusual_group`
- `amount_to_max_group`
- `ip_score_risk_group`
- `decline_rate_risk_group`
- `credit_util_risk_group`
- `txn_count_7d_group`
- `txn_velocity_group`

## 4. Binary behavior features

- `very_low_mcc_entropy_flag`
- `low_mcc_entropy_flag`
- `low_night_ratio_flag`
- `high_max_amount_30d_flag`
- `low_spending_trend_flag`
- `low_country_diversity_flag`
- `night_transaction_flag`
- `card_not_present_flag`
- `cross_border_flag`
- `tokenised_flag`

## 5. Interaction features

- `low_mcc_entropy_x_low_night_ratio`
- `low_mcc_entropy_x_low_night_ratio_x_card_not_present`
- `low_mcc_entropy_x_high_max_amount_30d`
- `low_mcc_entropy_x_low_spending_trend`
- `low_night_ratio_x_high_max_amount_30d`
- `low_night_ratio_x_night_transaction`
- `low_mcc_entropy_x_card_not_present`
- `cross_border_x_low_country_diversity`

## 6. Thresholds fit từ train

- `very_low_mcc_entropy_q05`: `0.197`
- `low_mcc_entropy_q10`: `0.393`
- `low_night_ratio_q10`: `0.101`
- `high_max_amount_q90`: `225176.777`
- `low_spending_trend_q10`: `0.584`
- `low_country_diversity_q10`: `3.0`
- `high_device_diversity_q90`: `36.0`
- `high_decline_rate_q90`: `0.283`
- `high_ip_score_q90`: `0.5123`

## 7. Ghi chú leakage


Các threshold trong file này được fit từ train và apply sang validation/test.
Giai đoạn này không fit scaler hoặc encoder. Việc scale numeric và one-hot categorical
sẽ được thực hiện trong model pipeline ở giai đoạn 5.
