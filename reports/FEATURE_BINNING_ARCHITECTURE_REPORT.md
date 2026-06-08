# Báo cáo kiến trúc chia bin và tạo feature

Nguồn tham chiếu chính:

- `src/features/build_features.py`
- `src/features/amount_features.py`
- `src/features/behavior_features.py`
- `src/features/interaction_features.py`
- `scripts/04_build_features.py`
- `data/processed/feature_metadata.json`
- `models/final_model/final_model_metadata.json`

## 1. Mục tiêu của binning trong kiến trúc này

Binning trong project này không chỉ dùng để làm EDA, mà là một phần của kiến trúc feature engineering. Mục tiêu là chuyển các biến liên tục hoặc biến hành vi khó đọc thành các nhóm rủi ro có ý nghĩa nghiệp vụ, ví dụ `low_mcc_entropy_flag`, `night_ratio_group`, `amount_to_max_group`, `ip_score_risk_group`.

Các bin sau đó được dùng theo 3 cách:

- Dùng trực tiếp như categorical feature, sau đó one-hot encode trong model pipeline.
- Dùng để tạo binary risk flag 0/1.
- Dùng để tạo interaction feature giữa các tín hiệu rủi ro đã được chứng minh có lift cao trong EDA.

## 2. Vị trí binning trong pipeline

Luồng xử lý chính ở `scripts/04_build_features.py`:

1. Load `train.parquet`, `validation.parquet`, `test.parquet`.
2. Khởi tạo `FraudFeatureBuilder`.
3. `builder.fit_transform(train_df)`: fit threshold trên train và transform train.
4. `builder.transform(validation_df)` và `builder.transform(test_df)`: dùng lại threshold từ train.
5. Lưu full feature dataset:
   - `data/processed/train_features.parquet`
   - `data/processed/validation_features.parquet`
   - `data/processed/test_features.parquet`
6. Lưu model-ready dataset:
   - `data/processed/model_train_features.parquet`
   - `data/processed/model_validation_features.parquet`
   - `data/processed/model_test_features.parquet`
7. Lưu metadata:
   - `data/processed/feature_metadata.json`

Điểm quan trọng: các threshold theo percentile chỉ được fit trên train, sau đó apply sang validation/test. Cách này tránh data leakage.

## 3. Kiến trúc feature builder

`FraudFeatureBuilder.transform()` chạy theo thứ tự:

```text
raw split
  -> create_amount_features()
  -> create_behavior_features(thresholds from train)
  -> create_interaction_features()
  -> _normalize_model_columns()
```

Ý nghĩa từng bước:

| Bước | File | Vai trò |
| --- | --- | --- |
| Amount features | `amount_features.py` | Tạo ratio, log transform, z-score liên quan amount |
| Behavior bins/flags | `behavior_features.py` | Tạo categorical bins cố định và binary flags theo train thresholds |
| Interaction features | `interaction_features.py` | Nhân các flag 0/1 để tạo interaction |
| Normalize columns | `build_features.py` | Ép numeric, categorical, binary về đúng dtype |

## 4. Các feature numeric làm nền trước khi chia bin

Một số bin được tạo từ các feature đã biến đổi, không dùng raw amount trực tiếp.

| Feature | Công thức | Ý nghĩa |
| --- | --- | --- |
| `log_mean_amount_30d` | `log1p(mean_amount_30d clipped at 0)` | Giảm lệch phải của mean amount lịch sử |
| `amount_to_mean_30d` | `amount / (mean_amount_30d + 1)` | Giao dịch hiện tại lớn hơn trung bình lịch sử bao nhiêu lần |
| `log_amount_to_max_30d` | `log1p(amount / (max_amount_30d + 1))` | Giao dịch hiện tại so với max lịch sử, có log để giảm skew |
| `amount_z_30d` | `(amount - mean_amount_30d) / (std_amount_30d + 1)` | Độ lệch amount hiện tại so với lịch sử user |
| `night_unusual_score` | `is_night_txn * (1 - night_ratio_30d)` | Giao dịch ban đêm có bất thường so với thói quen của user không |

Các feature này vẫn được giữ dạng numeric trong model-ready data, không nhất thiết bị chia bin. Tuy nhiên một số biến ratio/score cũng được dùng để tạo categorical bin như `amount_to_max_group`, `night_unusual_group`.

## 5. Nhóm bin cố định theo nghiệp vụ

Các bin này được tạo bằng `pd.cut()` với ngưỡng cố định, không cần fit từ train. Chúng được dùng như categorical features và sẽ được one-hot encode ở model pipeline.

### 5.1 `ip_score_risk_group`

Nguồn: `ip_score`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0.2]` | `low` |
| `(0.2, 0.5]` | `medium` |
| `(0.5, 0.8]` | `high` |
| `(0.8, inf)` | `very_high` |

Ý nghĩa: nhóm mức rủi ro IP theo score kỹ thuật.

### 5.2 `decline_rate_risk_group`

Nguồn: `decline_rate_30d`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0.1]` | `low` |
| `(0.1, 0.3]` | `medium` |
| `(0.3, 0.6]` | `high` |
| `(0.6, inf)` | `very_high` |

Ý nghĩa: tỷ lệ bị decline trong 30 ngày càng cao thì hành vi càng rủi ro.

### 5.3 `credit_util_risk_group`

Nguồn: `credit_util_today`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0.3]` | `low` |
| `(0.3, 0.6]` | `medium` |
| `(0.6, 0.9]` | `high` |
| `(0.9, inf)` | `very_high` |

Ý nghĩa: mức sử dụng credit trong ngày.

### 5.4 `txn_count_7d_group`

Nguồn: `txn_count_7d`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 1]` | `0_1` |
| `(1, 5]` | `2_5` |
| `(5, 10]` | `6_10` |
| `(10, 20]` | `11_20` |
| `(20, inf)` | `gt_20` |

Ý nghĩa: nhóm tần suất giao dịch 7 ngày gần nhất.

### 5.5 `txn_velocity_group`

Nguồn: `txn_count_ratio_7d_30d = txn_count_7d / (txn_count_30d + 1)`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0.2]` | `very_low` |
| `(0.2, 0.5]` | `low` |
| `(0.5, 0.8]` | `medium` |
| `(0.8, 1.2]` | `high` |
| `(1.2, inf)` | `very_high` |

Ý nghĩa: tốc độ giao dịch ngắn hạn so với lịch sử 30 ngày.

### 5.6 `night_ratio_group`

Nguồn: `night_ratio_30d`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0]` | `zero` |
| `(0, 0.1]` | `very_low` |
| `(0.1, 0.3]` | `low` |
| `(0.3, 0.6]` | `medium` |
| `(0.6, inf)` | `high` |

Ý nghĩa: user có thói quen giao dịch ban đêm thấp hay cao.

### 5.7 `amount_to_mean_group`

Nguồn: `amount_to_mean_30d`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 1]` | `normal` |
| `(1, 2]` | `slightly_high` |
| `(2, 5]` | `high` |
| `(5, 10]` | `very_high` |
| `(10, inf)` | `extreme` |

Ghi chú: feature này được tạo trong full feature dataset nhưng hiện không nằm trong `categorical_features` của `data/processed/feature_metadata.json`, nên không được đưa vào model-ready feature set mặc định.

### 5.8 `amount_to_max_group`

Nguồn: `amount_to_max_30d = amount / (max_amount_30d + 1)`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0.5]` | `below_half_max` |
| `(0.5, 1]` | `near_max` |
| `(1, 2]` | `above_max` |
| `(2, 5]` | `much_above_max` |
| `(5, inf)` | `extreme` |

Ý nghĩa: giao dịch hiện tại so với max amount trong lịch sử 30 ngày của user.

### 5.9 `night_unusual_group`

Nguồn: `night_unusual_score = is_night_txn * (1 - night_ratio_30d)`.

| Khoảng giá trị | Label |
| --- | --- |
| `(-inf, 0]` | `not_night` |
| `(0, 0.3]` | `low` |
| `(0.3, 0.6]` | `medium` |
| `(0.6, 0.9]` | `high` |
| `(0.9, inf)` | `very_high` |

Ý nghĩa: nếu giao dịch xảy ra ban đêm nhưng user thường không giao dịch ban đêm, score sẽ cao.

## 6. Nhóm threshold theo percentile fit từ train

Các threshold này nằm trong `data/processed/feature_metadata.json`. Chúng được fit từ train bằng quantile, sau đó apply cố định sang validation/test.

| Threshold | Cột gốc | Quantile | Giá trị đã fit | Rule tạo flag |
| --- | --- | ---: | ---: | --- |
| `very_low_mcc_entropy_q05` | `mcc_entropy_30d` | 5% | `0.197` | `mcc_entropy_30d <= 0.197` |
| `low_mcc_entropy_q10` | `mcc_entropy_30d` | 10% | `0.393` | `mcc_entropy_30d <= 0.393` |
| `low_night_ratio_q10` | `night_ratio_30d` | 10% | `0.101` | `night_ratio_30d <= 0.101` |
| `high_max_amount_q90` | `max_amount_30d` | 90% | `225176.777` | `max_amount_30d >= 225176.777` |
| `low_spending_trend_q10` | `spending_trend` | 10% | `0.584` | `spending_trend <= 0.584` |
| `low_country_diversity_q10` | `distinct_countries_30d` | 10% | `3.0` | `distinct_countries_30d <= 3.0` |
| `high_device_diversity_q90` | `device_diversity_30d` | 90% | `36.0` | `device_diversity_30d >= 36.0` |
| `high_decline_rate_q90` | `decline_rate_30d` | 90% | `0.283` | `decline_rate_30d >= 0.283` |
| `high_ip_score_q90` | `ip_score` | 90% | `0.5123` | `ip_score >= 0.5123` |

## 7. Binary flags được đưa vào model-ready metadata

Trong `data/processed/feature_metadata.json`, các binary features chính thức gồm 10 cột:

| Feature | Cách tạo | Ý nghĩa |
| --- | --- | --- |
| `very_low_mcc_entropy_flag` | `mcc_entropy_30d <= 0.197` | User có hành vi MCC cực ít đa dạng |
| `low_mcc_entropy_flag` | `mcc_entropy_30d <= 0.393` | User có hành vi MCC ít đa dạng |
| `low_night_ratio_flag` | `night_ratio_30d <= 0.101` | User ít giao dịch ban đêm |
| `high_max_amount_30d_flag` | `max_amount_30d >= 225176.777` | Lịch sử có max amount rất cao |
| `low_spending_trend_flag` | `spending_trend <= 0.584` | Xu hướng chi tiêu thấp hoặc giảm |
| `low_country_diversity_flag` | `distinct_countries_30d <= 3.0` | User có ít quốc gia giao dịch |
| `night_transaction_flag` | `is_night_txn == 1` | Giao dịch hiện tại xảy ra ban đêm |
| `card_not_present_flag` | `card_present` là false/0/no/n | Giao dịch không có thẻ vật lý |
| `cross_border_flag` | `cross_border` là true/1/yes/y | Giao dịch cross-border |
| `tokenised_flag` | `tokenised` là true/1/yes/y | Giao dịch tokenised |

Ghi chú: code cũng có thể tạo thêm `high_device_diversity_flag`, `high_decline_rate_flag`, `high_ip_score_flag`, nhưng các cột này không nằm trong `binary_features` chính thức của metadata hiện tại, nên không được chọn vào model-ready dataset mặc định.

## 8. Interaction features từ các flag

Interaction được tạo bằng cách nhân các flag 0/1. Nếu cả hai hoặc ba điều kiện cùng đúng thì interaction bằng 1, ngược lại bằng 0.

Các interaction features chính thức trong metadata:

| Feature | Công thức |
| --- | --- |
| `low_mcc_entropy_x_low_night_ratio` | `low_mcc_entropy_flag * low_night_ratio_flag` |
| `low_mcc_entropy_x_low_night_ratio_x_card_not_present` | `low_mcc_entropy_flag * low_night_ratio_flag * card_not_present_flag` |
| `low_mcc_entropy_x_high_max_amount_30d` | `low_mcc_entropy_flag * high_max_amount_30d_flag` |
| `low_mcc_entropy_x_low_spending_trend` | `low_mcc_entropy_flag * low_spending_trend_flag` |
| `low_night_ratio_x_high_max_amount_30d` | `low_night_ratio_flag * high_max_amount_30d_flag` |
| `low_night_ratio_x_night_transaction` | `low_night_ratio_flag * night_transaction_flag` |
| `low_mcc_entropy_x_card_not_present` | `low_mcc_entropy_flag * card_not_present_flag` |
| `cross_border_x_low_country_diversity` | `cross_border_flag * low_country_diversity_flag` |

Lý do tạo interaction: EDA/insight mining cho thấy fraud risk tăng mạnh khi nhiều tín hiệu hành vi bất thường cùng xuất hiện. Ví dụ `low_mcc_entropy AND low_night_ratio` có lift cao hơn nhiều so với từng biến đơn lẻ.

## 9. Feature set cuối của stage feature engineering

Theo `data/processed/feature_metadata.json`, output model-ready sau feature engineering có:

| Nhóm feature | Số lượng | Ví dụ |
| --- | ---: | --- |
| Numeric | 19 | `mcc_entropy_30d`, `night_ratio_30d`, `amount_z_30d` |
| Categorical | 21 | `night_ratio_group`, `amount_to_max_group`, `txn_velocity_group` |
| Binary | 10 | `low_mcc_entropy_flag`, `card_not_present_flag` |
| Interaction | 8 | `low_mcc_entropy_x_low_night_ratio` |
| Tổng model features | 58 | Numeric + categorical + binary + interaction |

Các cột bị drop khỏi model-ready data:

- `ip_risk`
- `txn_counts`
- `ip_address`
- `transaction_datetime`

Lý do: đây là cột raw, high-cardinality hoặc đã được parse thành feature có cấu trúc hơn.

## 10. Cách các bin đi vào model pipeline

Sau feature engineering, model pipeline xử lý feature như sau.

### Logistic Regression pipeline

File: `src/pipelines/preprocessing_pipeline.py`.

| Nhóm | Xử lý |
| --- | --- |
| Numeric | Median imputer + StandardScaler |
| Categorical bins | Fill missing bằng `"missing"` + OneHotEncoder |
| Binary | Fill missing bằng 0 |
| Interaction | Fill missing bằng 0 |

OneHotEncoder cho Logistic Regression dùng rare category grouping nếu bật:

- `handle_unknown="infrequent_if_exist"`
- `min_frequency=50`

### Tree/boosting pipeline

File: `src/pipelines/tree_preprocessing_pipeline.py`.

| Nhóm | Xử lý |
| --- | --- |
| Numeric | Median imputer, không scale |
| Categorical bins | Fill missing bằng `"missing"` + OneHotEncoder |
| Binary | Fill missing bằng 0 |
| Interaction | Fill missing bằng 0 |

OneHotEncoder cho tree models dùng rare category grouping:

- `handle_unknown="infrequent_if_exist"`
- `min_frequency=100`

Final promoted model trong `models/final_model/final_model_metadata.json` là CatBoost tuned `cat_tune_depth7_l2_8_lr003_iter700`, dùng `min_frequency=100` và `drop_manual_interactions=true`.

Điều này có nghĩa:

- Các categorical bins, binary flags và numeric behavior features vẫn là phần lõi của kiến trúc.
- Manual interaction features được tạo ở stage feature engineering, nhưng final CatBoost có thể bỏ nhóm manual interaction vì CatBoost tự học được interaction phi tuyến từ các feature gốc.

## 11. Ví dụ end-to-end một dòng dữ liệu

Giả sử một giao dịch có:

| Input | Giá trị |
| --- | ---: |
| `mcc_entropy_30d` | `0.25` |
| `night_ratio_30d` | `0.05` |
| `max_amount_30d` | `300000` |
| `spending_trend` | `0.50` |
| `distinct_countries_30d` | `2` |
| `is_night_txn` | `1` |
| `card_present` | `False` |

Các flags sẽ được tạo:

| Feature | Kết quả | Lý do |
| --- | ---: | --- |
| `low_mcc_entropy_flag` | `1` | `0.25 <= 0.393` |
| `low_night_ratio_flag` | `1` | `0.05 <= 0.101` |
| `high_max_amount_30d_flag` | `1` | `300000 >= 225176.777` |
| `low_spending_trend_flag` | `1` | `0.50 <= 0.584` |
| `low_country_diversity_flag` | `1` | `2 <= 3` |
| `night_transaction_flag` | `1` | Giao dịch ban đêm |
| `card_not_present_flag` | `1` | `card_present=False` |

Interaction tương ứng:

| Interaction | Kết quả |
| --- | ---: |
| `low_mcc_entropy_x_low_night_ratio` | `1` |
| `low_mcc_entropy_x_low_night_ratio_x_card_not_present` | `1` |
| `low_mcc_entropy_x_high_max_amount_30d` | `1` |
| `low_mcc_entropy_x_low_spending_trend` | `1` |
| `low_night_ratio_x_high_max_amount_30d` | `1` |
| `low_night_ratio_x_night_transaction` | `1` |
| `low_mcc_entropy_x_card_not_present` | `1` |
| `cross_border_x_low_country_diversity` | phụ thuộc `cross_border_flag` |

Đây là kiểu giao dịch mà insight mining đã chỉ ra là có rủi ro cao vì nhiều tín hiệu behavioral anomaly cùng xuất hiện.

## 12. Vì sao cách chia bin này hợp lý cho fraud detection

Các bin được thiết kế theo hai nguyên tắc:

- Binning cố định cho các feature có ý nghĩa nghiệp vụ rõ như risk score, decline rate, transaction velocity.
- Binning theo percentile train cho các feature phụ thuộc phân phối dữ liệu như MCC entropy, night ratio, max amount, spending trend.

Cách này giúp:

- Giảm phụ thuộc vào giá trị raw khó giải thích.
- Tạo feature dễ diễn giải trong báo cáo nghiệp vụ.
- Tránh leakage vì threshold phân phối được fit từ train.
- Giúp Logistic Regression học được rule phi tuyến thông qua binary và interaction features.
- Giúp CatBoost/tree models có thêm tín hiệu categorical/binary rõ ràng bên cạnh numeric features.

## 13. Câu mô tả ngắn có thể đưa vào báo cáo chính

Trong kiến trúc feature engineering, các biến liên tục được chuyển thành ba dạng tín hiệu: numeric anomaly features, categorical risk bins và binary risk flags. Những bin có ngưỡng nghiệp vụ cố định như `ip_score_risk_group`, `decline_rate_risk_group`, `txn_velocity_group` được tạo bằng `pd.cut`, còn các flag hành vi như `low_mcc_entropy_flag`, `low_night_ratio_flag`, `high_max_amount_30d_flag` được tạo từ percentile fit trên train để tránh leakage. Sau đó, các flag rủi ro được nhân với nhau để tạo interaction features như `low_mcc_entropy_x_low_night_ratio`, giúp mô hình học các tổ hợp hành vi có fraud lift cao.

