# Fraud Detection Project

Dự án xây dựng pipeline phát hiện gian lận giao dịch tài chính bằng Machine Learning. Bài toán có target mất cân bằng mạnh, vì vậy dự án tập trung vào các metric phù hợp hơn accuracy: `PR-AUC`, `F2-score`, `Precision`, `Recall` và `Review Rate`.

Model cuối cùng hiện tại là CatBoost, được promote vào `models/final_model/` và được dùng bởi FastAPI/Streamlit để demo inference.

## 1. Cấu trúc thư mục

```text
fraud-detection-project/
|-- app/                  # FastAPI API và Streamlit UI
|   |-- api/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/              # Dữ liệu gốc: fraud2.csv, fraud_metadata.xlsx
|   |-- interim/          # Dữ liệu đã clean tạm thời
|   |-- processed/        # Train/validation/test và feature parquet
|   `-- reports/          # CSV output từ EDA, mining, training
|-- models/               # Model artifacts và metadata
|-- notebooks/            # Notebook sinh từ pipeline/báo cáo
|-- reports/              # Báo cáo Markdown tổng hợp
|-- scripts/              # Entrypoint chạy từng giai đoạn
|-- src/                  # Source code pipeline nội bộ
|-- tests/                # Unit tests
|-- requirements.txt
`-- README.md
```

## 2. Yêu cầu môi trường

- Python 3.9 trở lên. Dự án hiện đã có `.venv` local, nhưng nên tạo mới nếu clone sang máy khác.
- Windows PowerShell, macOS Terminal hoặc Linux shell đều chạy được.
- Dữ liệu raw cần đặt trong `data/raw/`:
  - `fraud2.csv`
  - `fraud_metadata.xlsx`

Lưu ý: trong workspace hiện tại file metadata đang có tên `fraud_metadata (2).xlsx`, trong khi script `01_run_data_overview.py` tìm `fraud_metadata.xlsx`. Nếu chạy từ đầu, hãy đổi tên file thành:

```powershell
Rename-Item "data\raw\fraud_metadata (2).xlsx" "fraud_metadata.xlsx"
```

## 3. Cài đặt

Từ thư mục gốc dự án:

```powershell
cd E:\fraud-detection-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng macOS/Linux:

```bash
cd fraud-detection-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn activate script, chạy PowerShell bằng quyền phù hợp và set execution policy cho user hiện tại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Chạy pipeline từng giai đoạn

Tất cả lệnh bên dưới chạy từ thư mục gốc `fraud-detection-project` và sau khi đã activate virtual environment.

### Giai đoạn 1: Data understanding

Đọc `data/raw/fraud2.csv`, kiểm tra cột target, tạo các báo cáo tổng quan.

```powershell
python scripts/01_run_data_overview.py
```

Output chính:

- `data/reports/*.csv`
- `reports/eda_report.md`

### Giai đoạn 2: EDA

Chạy phân tích exploratory data analysis sau bước overview.

```powershell
python scripts/02_run_eda.py
```

Output chính nằm trong `data/reports/` và `reports/`.

### Giai đoạn 3: Insight mining và chia tập dữ liệu

Khai phá pattern/rule, tạo insight report và tạo các file split train/validation/test.

```powershell
python scripts/03_run_insight_mining.py
```

Output chính:

- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/test.parquet`
- `data/reports/insights/*.csv`
- `reports/insight_stability_report.md`

Nếu cần sinh thêm visualization cho insight mining:

```powershell
python scripts/03_generate_mining_visualizations.py
```

### Giai đoạn 4: Feature engineering

Tạo feature đầy đủ và bộ feature sẵn sàng đưa vào model.

```powershell
python scripts/04_build_features.py
```

Output chính:

- `data/processed/train_features.parquet`
- `data/processed/validation_features.parquet`
- `data/processed/test_features.parquet`
- `data/processed/model_train_features.parquet`
- `data/processed/model_validation_features.parquet`
- `data/processed/model_test_features.parquet`
- `data/processed/feature_metadata.json`
- `reports/feature_engineering_report.md`

### Giai đoạn 5: Train baseline Logistic Regression

Train model baseline và tìm threshold theo validation set.

```powershell
python scripts/05_train_model.py
```

Output chính:

- `models/logistic_regression_pipeline.joblib`
- `models/logistic_regression_metadata.json`
- `data/reports/model/*.csv`

Sinh báo cáo baseline:

```powershell
python scripts/06_generate_baseline_report.py
```

Output:

- `data/reports/model_report.md`

### Giai đoạn 6: Chạy các experiment Logistic Regression

So sánh các cấu hình baseline khác nhau, đặc biệt là việc dùng/không dùng `mcc` và rare category grouping.

```powershell
python scripts/05_run_experiments.py
```

Output chính:

- `models/experiments/*/pipeline.joblib`
- `data/reports/model/experiments/*`
- `data/reports/model/experiment_comparison.csv`
- `reports/model_experiment_comparison.md`

### Giai đoạn 7: Train model boosting

Chạy LightGBM, XGBoost và CatBoost. Script này cấu hình GPU cho các boosting model.

```powershell
python scripts/08_train_gpu_boosting_models.py
```

Output chính:

- `models/gpu_boosting_experiments/*/pipeline.joblib`
- `data/reports/gpu_boosting_models/*`
- `reports/gpu_boosting_model_comparison.md`

Nếu máy không có GPU/CUDA/OpenCL phù hợp, bước này có thể lỗi. Khi đó cần chỉnh tham số trong `scripts/08_train_gpu_boosting_models.py` hoặc model wrapper trong `src/models/` sang CPU.

### Giai đoạn 8: Báo cáo so sánh model cuối

Tổng hợp so sánh model sau các experiment.

```powershell
python scripts/09_generate_final_model_report.py
```

Output:

- `reports/final_model_comparison_report.md`

### Giai đoạn 9: Tune CatBoost

Chạy tuning các cấu hình CatBoost.

```powershell
python scripts/10_tune_catboost.py
```

Output chính:

- `models/catboost_tuning_experiments/*/pipeline.joblib`
- `data/reports/catboost_tuning/*`
- `data/reports/catboost_tuning/catboost_tuning_comparison.csv`

### Giai đoạn 10: Promote final model

Copy model tốt nhất từ tuning experiments sang `models/final_model/` để app inference sử dụng.

```powershell
python scripts/11_promote_final_model.py
```

Output:

- `models/final_model/fraud_catboost_final.joblib`
- `models/final_model/final_model_metadata.json`
- `models/final_model/final_feature_importance.csv`

Hiện script promote đang chọn experiment:

```text
cat_tune_depth7_l2_8_lr003_iter700
```

### Giai đoạn 11: Error analysis

Phân tích false positive, false negative và các nhóm lỗi của final model.

```powershell
python scripts/12_run_error_analysis.py
```

Output:

- `data/reports/final_error_analysis/*`
- Báo cáo liên quan trong `reports/`

### Giai đoạn 12: Threshold policy

Đánh giá chính sách threshold theo năng lực review thủ công.

```powershell
python scripts/13_generate_threshold_policy.py
```

Output:

- `data/reports/threshold_policy/threshold_policy_review_capacity.csv`

### Giai đoạn 13: Tổng hợp final project report

Sinh báo cáo tổng kết cuối cùng cho dự án.

```powershell
python scripts/14_generate_final_project_report.py
```

Output:

- `reports/FINAL_DS_PROJECT_REPORT.md`

### Tùy chọn: Sinh notebook

Nếu muốn tạo notebook từ project scripts/reports:

```powershell
python scripts/00_generate_project_notebooks.py
```

## 5. Chạy nhanh từ đầu đến cuối

Nếu dữ liệu raw đã đúng tên và môi trường đã cài xong, có thể chạy lần lượt:

```powershell
python scripts/01_run_data_overview.py
python scripts/02_run_eda.py
python scripts/03_run_insight_mining.py
python scripts/04_build_features.py
python scripts/05_train_model.py
python scripts/06_generate_baseline_report.py
python scripts/05_run_experiments.py
python scripts/08_train_gpu_boosting_models.py
python scripts/09_generate_final_model_report.py
python scripts/10_tune_catboost.py
python scripts/11_promote_final_model.py
python scripts/12_run_error_analysis.py
python scripts/13_generate_threshold_policy.py
python scripts/14_generate_final_project_report.py
```

## 6. Chạy API inference

API hiện tại nhận input dạng feature-ready, tức là các cột/feature đã qua xử lý từ pipeline feature engineering.

Chạy server:

```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở các endpoint:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Ví dụ request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":{"mcc_entropy_30d":0.4,"night_ratio_30d":0.1,"log_amount_to_max_30d":0.8,"spending_trend":0.6}}'
```

Lưu ý: vì model pipeline cần đầy đủ các feature đã train, request thực tế nên đưa bộ feature đầy đủ giống schema của `data/processed/model_train_features.parquet` trừ cột `fraud`.

## 7. Chạy Streamlit UI

Streamlit demo dùng final model trong `models/final_model/`.

```powershell
streamlit run app/streamlit_app.py
```

Mặc định UI mở tại:

```text
http://localhost:8501
```

Form Streamlit hiện nhập feature-ready values và trả về:

- fraud probability
- prediction
- risk level
- decision
- threshold

## 8. Chính sách risk hiện tại

Theo metadata của final model:

| Khoảng probability | Risk level | Decision |
| --- | --- | --- |
| `< 0.30` | LOW | APPROVE |
| `0.30` đến `< selected_threshold` | MEDIUM | MANUAL_REVIEW |
| `>= selected_threshold` | HIGH | FRAUD_ALERT |

Threshold của final model được lưu trong:

```text
models/final_model/final_model_metadata.json
```

## 9. Các file báo cáo quan trọng

- `reports/FINAL_DS_PROJECT_REPORT.md`
- `reports/final_model_comparison_report.md`
- `reports/insight_stability_report.md`
- `reports/gpu_boosting_model_comparison.md`
- `reports/model_experiment_comparison.md`
- `reports/feature_engineering_report.md`

## 10. Lỗi thường gặp

### Không tìm thấy `fraud_metadata.xlsx`

Đổi tên file metadata trong `data/raw/`:

```powershell
Rename-Item "data\raw\fraud_metadata (2).xlsx" "fraud_metadata.xlsx"
```

### Không tìm thấy final model khi chạy API/UI

Chạy promote model:

```powershell
python scripts/11_promote_final_model.py
```

Nếu chưa có tuning experiment source, cần chạy trước:

```powershell
python scripts/10_tune_catboost.py
```

### Lỗi import CatBoost/LightGBM/XGBoost

Cài thêm package model nâng cao:

```powershell
pip install catboost lightgbm xgboost
```

### Lỗi GPU khi train boosting

Chạy bước baseline trước để có model/report, sau đó sửa cấu hình boosting sang CPU nếu máy không có GPU phù hợp. Các file liên quan:

- `scripts/08_train_gpu_boosting_models.py`
- `src/models/catboost_model.py`
- `src/models/lightgbm_model.py`
- `src/models/xgboost_model.py`

### API/UI nhận feature-ready input

Dự án hiện chưa có realtime raw-transaction feature builder trong API. Nếu muốn predict từ giao dịch raw, cần thêm một inference feature pipeline dùng cùng logic với `src/features/build_features.py`.
