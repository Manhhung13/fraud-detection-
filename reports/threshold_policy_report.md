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
