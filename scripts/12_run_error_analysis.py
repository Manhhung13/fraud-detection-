from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.analysis.error_analysis import run_error_analysis


def main():
    run_error_analysis(
        project_root=PROJECT_ROOT,
        experiment_name="cat_tune_depth7_l2_8_lr003_iter700",
    )


if __name__ == "__main__":
    main()