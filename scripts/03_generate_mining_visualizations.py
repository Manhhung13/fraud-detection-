from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.mining.visualization import (  # noqa: E402
    generate_data_mining_visualizations,
    write_visualization_gallery,
)


INSIGHT_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "insights"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "data_mining"
GALLERY_REPORT_PATH = PROJECT_ROOT / "reports" / "data_mining_visualizations.md"


def main() -> None:
    print("=" * 80)
    print("GENERATE DATA MINING VISUALIZATIONS")
    print("=" * 80)

    if not INSIGHT_REPORT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy insight reports: {INSIGHT_REPORT_DIR}\n"
            "Hãy chạy trước: python scripts/03_run_insight_mining.py"
        )

    image_paths = generate_data_mining_visualizations(
        insight_dir=INSIGHT_REPORT_DIR,
        figures_dir=FIGURES_DIR,
    )

    write_visualization_gallery(
        image_paths=image_paths,
        output_path=GALLERY_REPORT_PATH,
        project_root=PROJECT_ROOT,
    )

    print(f"Generated {len(image_paths)} figures:")
    for path in image_paths:
        print(f"- {path}")

    print(f"\nGallery report: {GALLERY_REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
