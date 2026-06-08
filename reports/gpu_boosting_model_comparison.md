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
| cat_gpu_mcc_minfreq_100_no_interactions | catboost     |                       | True                        |             100 |                 0.74 |                       164 |      0.696704 |         0.526412 |      0.744845 |  0.616862 |  0.687768 |          0.0366    |       289 |       260 |        99 |     14352 |          11.3432  |
| cat_gpu_mcc_minfreq_100                 | catboost     |                       | False                       |             100 |                 0.73 |                       172 |      0.694449 |         0.513228 |      0.75     |  0.609424 |  0.686645 |          0.0378    |       291 |       276 |        97 |     14336 |          11.2949  |
| lgbm_gpu_no_mcc                         | lightgbm     | mcc                   | False                       |             100 |                 0.62 |                       171 |      0.679225 |         0.501742 |      0.742268 |  0.598753 |  0.677328 |          0.0382667 |       288 |       286 |       100 |     14326 |          18.728   |
| lgbm_gpu_mcc_minfreq_100                | lightgbm     |                       | False                       |             100 |                 0.62 |                       172 |      0.679193 |         0.501742 |      0.742268 |  0.598753 |  0.677328 |          0.0382667 |       288 |       286 |       100 |     14326 |          24.5722  |
| cat_gpu_no_mcc                          | catboost     | mcc                   | False                       |             100 |                 0.76 |                       171 |      0.692177 |         0.542969 |      0.716495 |  0.617778 |  0.67345  |          0.0341333 |       278 |       234 |       110 |     14378 |          11.9403  |
| xgb_gpu_mcc_minfreq_100                 | xgboost      |                       | False                       |             100 |                 0.68 |                       172 |      0.684834 |         0.492281 |      0.739691 |  0.591143 |  0.672131 |          0.0388667 |       287 |       296 |       101 |     14316 |           6.13283 |
| xgb_gpu_no_mcc                          | xgboost      | mcc                   | False                       |             100 |                 0.63 |                       171 |      0.680247 |         0.447806 |      0.762887 |  0.564347 |  0.668775 |          0.0440667 |       296 |       365 |        92 |     14247 |           5.9426  |

## 3. Cách chọn model


Ưu tiên model có:
1. PR-AUC cao.
2. F2 cao vì fraud detection ưu tiên recall.
3. Review rate phù hợp năng lực kiểm tra.
4. Precision không quá thấp.
5. Số chiều không quá lớn.
6. Không tăng độ phức tạp quá nhiều nếu hiệu năng chỉ cải thiện rất nhỏ.
