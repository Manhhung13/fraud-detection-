from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.reporting.final_model_report import generate_final_model_report


def main():
    print("=" * 80)
    print("GENERATE FINAL MODEL REPORT")
    print("=" * 80)

    generate_final_model_report(
        project_root=PROJECT_ROOT,
        output_path=PROJECT_ROOT / "reports" / "final_model_comparison_report.md",
    )

    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()