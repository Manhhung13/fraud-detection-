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
