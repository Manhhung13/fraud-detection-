from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.reporting.model_insight_report import generate_all_final_reports


def main():
    parser = argparse.ArgumentParser(
        description="Generate final baseline model report with insight explanations."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="lr_no_mcc",
        help="Experiment name to use as final baseline. Default: lr_no_mcc",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GENERATE FINAL BASELINE MODEL REPORT")
    print("=" * 80)

    generate_all_final_reports(
        project_root=PROJECT_ROOT,
        selected_experiment=args.experiment,
    )

    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()