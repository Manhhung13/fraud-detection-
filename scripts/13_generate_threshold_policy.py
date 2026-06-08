from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.analysis.threshold_policy import generate_threshold_policy


def main():
    generate_threshold_policy(
        project_root=PROJECT_ROOT,
        experiment_name="cat_tune_depth7_l2_8_lr003_iter700",
        capacities=[0.03, 0.05, 0.10],
    )


if __name__ == "__main__":
    main()