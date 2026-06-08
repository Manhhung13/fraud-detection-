# Fraud Detection Project

Du an xay dung pipeline phat hien gian lan giao dich tai chinh bang Machine Learning. Bai toan co target mat can bang manh, vi vay du an tap trung vao cac metric phu hop hon accuracy: `PR-AUC`, `F2-score`, `Precision`, `Recall` va `Review Rate`.

Model cuoi cung hien tai la CatBoost, duoc promote vao `models/final_model/` va duoc dung boi FastAPI/Streamlit de demo inference.

## 1. Cau truc thu muc

```text
fraud-detection-project/
|-- app/                  # FastAPI API va Streamlit UI
|   |-- api/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/              # Du lieu goc: fraud2.csv, fraud_metadata.xlsx
|   |-- interim/          # Du lieu da clean tam thoi
|   |-- processed/        # Train/validation/test va feature parquet
|   `-- reports/          # CSV output tu EDA, mining, training
|-- models/               # Model artifacts va metadata
|-- notebooks/            # Notebook sinh tu pipeline/bao cao
|-- reports/              # Bao cao Markdown tong hop
|-- scripts/              # Entrypoint chay tung giai doan
|-- src/                  # Source code pipeline noi bo
|-- tests/                # Unit tests
|-- requirements.txt
`-- README.md
```

## 2. Yeu cau moi truong

- Python 3.9 tro len. Du an hien da co `.venv` local, nhung nen tao moi neu clone sang may khac.
- Windows PowerShell, macOS Terminal hoac Linux shell deu chay duoc.
- Du lieu raw can dat trong `data/raw/`:
  - `fraud2.csv`
  - `fraud_metadata.xlsx`

Luu y: trong workspace hien tai file metadata dang co ten `fraud_metadata (2).xlsx`, trong khi script `01_run_data_overview.py` tim `fraud_metadata.xlsx`. Neu chay tu dau, hay doi ten file thanh:

```powershell
Rename-Item "data\raw\fraud_metadata (2).xlsx" "fraud_metadata.xlsx"
```

## 3. Cai dat

Tu thu muc goc du an:

```powershell
cd E:\fraud-detection-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu dung macOS/Linux:

```bash
cd fraud-detection-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu PowerShell chan activate script, chay PowerShell bang quyen phu hop va set execution policy cho user hien tai:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Chay pipeline tung giai doan

Tat ca lenh ben duoi chay tu thu muc goc `fraud-detection-project` va sau khi da activate virtual environment.

### Giai doan 1: Data understanding

Doc `data/raw/fraud2.csv`, kiem tra cot target, tao cac bao cao tong quan.

```powershell
python scripts/01_run_data_overview.py
```

Output chinh:

- `data/reports/*.csv`
- `reports/eda_report.md`

### Giai doan 2: EDA

Chay phan tich exploratory data analysis sau buoc overview.

```powershell
python scripts/02_run_eda.py
```

Output chinh nam trong `data/reports/` va `reports/`.

### Giai doan 3: Insight mining va chia tap du lieu

Khai pha pattern/rule, tao insight report va tao cac file split train/validation/test.

```powershell
python scripts/03_run_insight_mining.py
```

Output chinh:

- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/test.parquet`
- `data/reports/insights/*.csv`
- `reports/insight_stability_report.md`

Neu can sinh them visualization cho insight mining:

```powershell
python scripts/03_generate_mining_visualizations.py
```

### Giai doan 4: Feature engineering

Tao feature day du va bo feature san sang dua vao model.

```powershell
python scripts/04_build_features.py
```

Output chinh:

- `data/processed/train_features.parquet`
- `data/processed/validation_features.parquet`
- `data/processed/test_features.parquet`
- `data/processed/model_train_features.parquet`
- `data/processed/model_validation_features.parquet`
- `data/processed/model_test_features.parquet`
- `data/processed/feature_metadata.json`
- `reports/feature_engineering_report.md`

### Giai doan 5: Train baseline Logistic Regression

Train model baseline va tim threshold theo validation set.

```powershell
python scripts/05_train_model.py
```

Output chinh:

- `models/logistic_regression_pipeline.joblib`
- `models/logistic_regression_metadata.json`
- `data/reports/model/*.csv`

Sinh bao cao baseline:

```powershell
python scripts/06_generate_baseline_report.py
```

Output:

- `data/reports/model_report.md`

### Giai doan 6: Chay cac experiment Logistic Regression

So sanh cac cau hinh baseline khac nhau, dac biet la viec dung/khong dung `mcc` va rare category grouping.

```powershell
python scripts/05_run_experiments.py
```

Output chinh:

- `models/experiments/*/pipeline.joblib`
- `data/reports/model/experiments/*`
- `data/reports/model/experiment_comparison.csv`
- `reports/model_experiment_comparison.md`

### Giai doan 7: Train model boosting

Chay LightGBM, XGBoost va CatBoost. Script nay cau hinh GPU cho cac boosting model.

```powershell
python scripts/08_train_gpu_boosting_models.py
```

Output chinh:

- `models/gpu_boosting_experiments/*/pipeline.joblib`
- `data/reports/gpu_boosting_models/*`
- `reports/gpu_boosting_model_comparison.md`

Neu may khong co GPU/CUDA/OpenCL phu hop, buoc nay co the loi. Khi do can chinh tham so trong `scripts/08_train_gpu_boosting_models.py` hoac model wrapper trong `src/models/` sang CPU.

### Giai doan 8: Bao cao so sanh model cuoi

Tong hop so sanh model sau cac experiment.

```powershell
python scripts/09_generate_final_model_report.py
```

Output:

- `reports/final_model_comparison_report.md`

### Giai doan 9: Tune CatBoost

Chay tuning cac cau hinh CatBoost.

```powershell
python scripts/10_tune_catboost.py
```

Output chinh:

- `models/catboost_tuning_experiments/*/pipeline.joblib`
- `data/reports/catboost_tuning/*`
- `data/reports/catboost_tuning/catboost_tuning_comparison.csv`

### Giai doan 10: Promote final model

Copy model tot nhat tu tuning experiments sang `models/final_model/` de app inference su dung.

```powershell
python scripts/11_promote_final_model.py
```

Output:

- `models/final_model/fraud_catboost_final.joblib`
- `models/final_model/final_model_metadata.json`
- `models/final_model/final_feature_importance.csv`

Hien script promote dang chon experiment:

```text
cat_tune_depth7_l2_8_lr003_iter700
```

### Giai doan 11: Error analysis

Phan tich false positive, false negative va cac nhom loi cua final model.

```powershell
python scripts/12_run_error_analysis.py
```

Output:

- `data/reports/final_error_analysis/*`
- Bao cao lien quan trong `reports/`

### Giai doan 12: Threshold policy

Danh gia chinh sach threshold theo nang luc review thu cong.

```powershell
python scripts/13_generate_threshold_policy.py
```

Output:

- `data/reports/threshold_policy/threshold_policy_review_capacity.csv`

### Giai doan 13: Tong hop final project report

Sinh bao cao tong ket cuoi cung cho du an.

```powershell
python scripts/14_generate_final_project_report.py
```

Output:

- `reports/FINAL_DS_PROJECT_REPORT.md`

### Tuy chon: Sinh notebook

Neu muon tao notebook tu project scripts/reports:

```powershell
python scripts/00_generate_project_notebooks.py
```

## 5. Chay nhanh tu dau den cuoi

Neu du lieu raw da dung ten va moi truong da cai xong, co the chay lan luot:

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

## 6. Chay API inference

API hien tai nhan input dang feature-ready, tuc la cac cot/feature da qua xu ly tu pipeline feature engineering.

Chay server:

```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Mo cac endpoint:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Vi du request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":{"mcc_entropy_30d":0.4,"night_ratio_30d":0.1,"log_amount_to_max_30d":0.8,"spending_trend":0.6}}'
```

Luu y: vi model pipeline can day du cac feature da train, request thuc te nen dua bo feature day du giong schema cua `data/processed/model_train_features.parquet` tru cot `fraud`.

## 7. Chay Streamlit UI

Streamlit demo dung final model trong `models/final_model/`.

```powershell
streamlit run app/streamlit_app.py
```

Mac dinh UI mo tai:

```text
http://localhost:8501
```

Form Streamlit hien nhap feature-ready values va tra ve:

- fraud probability
- prediction
- risk level
- decision
- threshold

## 8. Chinh sach risk hien tai

Theo metadata cua final model:

| Khoang probability | Risk level | Decision |
| --- | --- | --- |
| `< 0.30` | LOW | APPROVE |
| `0.30` den `< selected_threshold` | MEDIUM | MANUAL_REVIEW |
| `>= selected_threshold` | HIGH | FRAUD_ALERT |

Threshold cua final model duoc luu trong:

```text
models/final_model/final_model_metadata.json
```

## 9. Cac file bao cao quan trong

- `reports/FINAL_DS_PROJECT_REPORT.md`
- `reports/final_model_comparison_report.md`
- `reports/insight_stability_report.md`
- `reports/gpu_boosting_model_comparison.md`
- `reports/model_experiment_comparison.md`
- `reports/feature_engineering_report.md`

## 10. Loi thuong gap

### Khong tim thay `fraud_metadata.xlsx`

Doi ten file metadata trong `data/raw/`:

```powershell
Rename-Item "data\raw\fraud_metadata (2).xlsx" "fraud_metadata.xlsx"
```

### Khong tim thay final model khi chay API/UI

Chay promote model:

```powershell
python scripts/11_promote_final_model.py
```

Neu chua co tuning experiment source, can chay truoc:

```powershell
python scripts/10_tune_catboost.py
```

### Loi import CatBoost/LightGBM/XGBoost

Cai them package model nang cao:

```powershell
pip install catboost lightgbm xgboost
```

### Loi GPU khi train boosting

Chay buoc baseline truoc de co model/report, sau do sua cau hinh boosting sang CPU neu may khong co GPU phu hop. Cac file lien quan:

- `scripts/08_train_gpu_boosting_models.py`
- `src/models/catboost_model.py`
- `src/models/lightgbm_model.py`
- `src/models/xgboost_model.py`

### API/UI nhan feature-ready input

Du an hien chua co realtime raw-transaction feature builder trong API. Neu muon predict tu giao dich raw, can them mot inference feature pipeline dung cung logic voi `src/features/build_features.py`.
