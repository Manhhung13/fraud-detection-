# Tree / GPU Boosting Model Experiment Comparison Report

## 1. Mục tiêu


So sánh các mô hình mở rộng với Logistic Regression baseline:

- Random Forest.
- LightGBM.
- XGBoost.
- CatBoost.

Tất cả model dùng cùng train/validation/test split, cùng metric, cùng threshold search trên validation.


## 2. Bảng so sánh

| experiment_name                         | model_name   | dropped_categorical   | drop_interaction_features   |   min_frequency |   selected_threshold |   n_preprocessed_features |   test_pr_auc |   test_precision |   test_recall |   test_f1 |   test_f2 |   test_review_rate |   test_tp |   test_fp |   test_fn |   test_tn |   elapsed_seconds |
|:----------------------------------------|:-------------|:----------------------|:----------------------------|----------------:|---------------------:|--------------------------:|--------------:|-----------------:|--------------:|----------:|----------:|-------------------:|----------:|----------:|----------:|----------:|------------------:|
| cat_tune_depth7_l2_8_lr003_iter700      | catboost     |                       | True                        |             100 |                 0.66 |                       164 |      0.702748 |         0.500832 |      0.775773 |  0.608696 |  0.699025 |          0.0400667 |       301 |       300 |        87 |     14312 |           9.6647  |
| cat_tune_depth6_l2_10_lr003_iter700     | catboost     |                       | True                        |             100 |                 0.74 |                       164 |      0.695589 |         0.524412 |      0.747423 |  0.616366 |  0.688836 |          0.0368667 |       290 |       263 |        98 |     14349 |          13.7882  |
| cat_tune_base_depth6_l2_5_lr003_iter700 | catboost     |                       | True                        |             100 |                 0.74 |                       164 |      0.696704 |         0.526412 |      0.744845 |  0.616862 |  0.687768 |          0.0366    |       289 |       260 |        99 |     14352 |           7.56788 |
| cat_tune_depth5_l2_8_lr003_iter700      | catboost     |                       | True                        |             100 |                 0.74 |                       164 |      0.682039 |         0.475962 |      0.765464 |  0.586957 |  0.682445 |          0.0416    |       297 |       327 |        91 |     14285 |           6.75438 |
| cat_tune_depth6_l2_5_lr002_iter1000     | catboost     |                       | True                        |             100 |                 0.71 |                       164 |      0.692939 |         0.476651 |      0.762887 |  0.58672  |  0.681086 |          0.0414    |       296 |       325 |        92 |     14287 |          10.5444  |

## 3. Cách chọn model


Ưu tiên model có:
1. PR-AUC cao.
2. F2 cao vì fraud detection ưu tiên recall.
3. Review rate phù hợp năng lực kiểm tra.
4. Precision không quá thấp.
5. Số chiều không quá lớn.
6. Không tăng độ phức tạp quá nhiều nếu hiệu năng chỉ cải thiện rất nhỏ.
