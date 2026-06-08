# Fraud Detection - Experiment Comparison Report

## 1. Mục tiêu


So sánh các cấu hình Logistic Regression để kiểm tra độ ổn định của feature `mcc`:

- Giữ `mcc` với rare category grouping min_frequency=50.
- Bỏ `mcc`.
- Giữ `mcc` nhưng tăng min_frequency=100.
- Giữ `mcc` nhưng tăng min_frequency=200.

Mục tiêu là xem model có phụ thuộc quá mạnh vào `mcc` hay không.


## 2. Bảng so sánh tổng thể

| experiment_name    |   min_frequency | dropped_categorical   |   selected_threshold |   n_preprocessed_features |   test_pr_auc |   test_precision |   test_recall |   test_f1 |   test_f2 |   test_review_rate |   test_tp |   test_fp |   test_fn |   test_tn |
|:-------------------|----------------:|:----------------------|---------------------:|--------------------------:|--------------:|-----------------:|--------------:|----------:|----------:|-------------------:|----------:|----------:|----------:|----------:|
| lr_no_mcc          |              50 | mcc                   |                 0.77 |                       171 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_100 |             100 |                       |                 0.77 |                       172 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_200 |             200 |                       |                 0.77 |                       172 |      0.533238 |         0.404478 |      0.698454 |  0.512287 |  0.609811 |          0.0446667 |       271 |       399 |       117 |     14213 |
| lr_mcc_minfreq_50  |              50 |                       |                 0.76 |                       355 |      0.448793 |         0.359249 |      0.690722 |  0.472663 |  0.583116 |          0.0497333 |       268 |       478 |       120 |     14134 |

## 3. Cách đọc kết quả


- Nếu bỏ `mcc` làm PR-AUC/F2 giảm mạnh, nghĩa là `mcc` đang mang tín hiệu quan trọng.
- Nếu bỏ `mcc` mà kết quả gần tương đương, nên cân nhắc bỏ `mcc` để model dễ generalize hơn.
- Nếu min_frequency=100 hoặc 200 giữ hiệu năng gần bằng min_frequency=50 nhưng giảm số chiều, nên ưu tiên bản gọn hơn.
- `review_rate` cho biết tỷ lệ giao dịch bị đẩy sang kiểm tra thủ công.
- Test set chỉ dùng để đánh giá cuối cùng, không dùng để chọn threshold.


## 4. Khuyến nghị chọn model


Ưu tiên model có:
1. PR-AUC tốt.
2. F2 tốt vì fraud detection ưu tiên recall.
3. Review rate phù hợp năng lực kiểm tra nghiệp vụ.
4. Số chiều sau preprocessing không quá lớn.
5. Không phụ thuộc quá cực đoan vào một nhóm category hiếm.
