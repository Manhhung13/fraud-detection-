from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.reporting.final_project_report import generate_final_project_report


def main():
    generate_final_project_report(PROJECT_ROOT)


if __name__ == "__main__":
    main()