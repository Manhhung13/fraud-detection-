from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.finalization.promote_model import promote_final_model


def main():
    promote_final_model(
        project_root=PROJECT_ROOT,
        final_experiment_name="cat_tune_depth7_l2_8_lr003_iter700",
    )


if __name__ == "__main__":
    main()