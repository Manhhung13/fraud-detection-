from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return f"_Không tìm thấy file: `{path}`_"
    return path.read_text(encoding="utf-8")


def generate_final_project_report(project_root: str | Path) -> None:
    project_root = Path(project_root)

    model_report = read_text_if_exists(project_root / "reports" / "final_model_comparison_report.md")
    error_report = read_text_if_exists(project_root / "reports" / "error_analysis_report.md")
    threshold_report = read_text_if_exists(project_root / "reports" / "threshold_policy_report.md")

    output_path = project_root / "reports" / "FINAL_DS_PROJECT_REPORT.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Final Data Science Project Report - Fraud Detection\n")

    lines.append("## 1. Business problem\n")
    lines.append(
        """
Mục tiêu của project là xây dựng mô hình phát hiện gian lận giao dịch tài chính.
Do dữ liệu mất cân bằng mạnh, accuracy không được dùng làm metric chính. 
Các metric chính gồm PR-AUC, Precision, Recall, F1, F2, Confusion Matrix và Review Rate.
"""
    )

    lines.append("\n## 2. Data science workflow\n")
    lines.append(
        """
Quy trình thực hiện:

1. Khảo sát dữ liệu và metadata.
2. EDA và kiểm tra mất cân bằng nhãn.
3. Insight mining trên train để tránh leakage.
4. Feature engineering dựa trên insight.
5. Train Logistic Regression baseline.
6. Train advanced boosting models: LightGBM, XGBoost, CatBoost.
7. Tuning CatBoost.
8. Chọn final model.
9. Error analysis.
10. Threshold policy theo năng lực review.
11. Inference pipeline, FastAPI, Streamlit demo.
"""
    )

    lines.append("\n---\n")
    lines.append(model_report)

    lines.append("\n---\n")
    lines.append(error_report)

    lines.append("\n---\n")
    lines.append(threshold_report)

    lines.append("\n## Final conclusion\n")
    lines.append(
        """
Final model được chọn là CatBoost tuned `cat_tune_depth7_l2_8_lr003_iter700`.

Model này được chọn vì có PR-AUC/F2 tốt nhất, recall cao, review rate vẫn dưới ngưỡng vận hành 5%, 
và feature importance vẫn khớp với các insight nghiệp vụ chính như mcc entropy, night behavior, amount anomaly, spending trend và country diversity.

Logistic Regression vẫn được giữ làm baseline giải thích được, còn CatBoost là model advanced dùng cho hiệu năng vận hành.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved final DS project report: {output_path}")