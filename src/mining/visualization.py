from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#2563EB",
    "teal": "#0F766E",
    "orange": "#EA580C",
    "red": "#DC2626",
    "green": "#16A34A",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "dark": "#111827",
    "light": "#E5E7EB",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _wrap(value: object, width: int = 34) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _save_current(figures_dir: Path, filename: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_split_distribution(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "split_summary.csv")
    if df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))

    bars = ax1.bar(x, df["rows"], color=COLORS["blue"], width=0.55, label="Rows")
    ax1.set_ylabel("Rows")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["split"].str.title())
    ax1.grid(axis="y", color=COLORS["light"], linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["fraud_rate"] * 100,
        marker="o",
        linewidth=2.4,
        color=COLORS["red"],
        label="Fraud rate",
    )
    ax2.set_ylabel("Fraud rate (%)")
    ax2.set_ylim(0, max(df["fraud_rate"].max() * 100 * 1.8, 4))

    for bar, row in zip(bars, df.to_dict("records")):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + df["rows"].max() * 0.025,
            f"{int(row['rows']):,}\nfraud {row['fraud_rate'] * 100:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax1.set_title("Data mining split: time-based train/validation/test")
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc="upper right")

    return _save_current(figures_dir, "01_split_distribution.png")


def plot_stable_segment_lift(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "stable_segments.csv")
    required = {"segment_name", "lift_train", "lift_valid", "fraud_rate_valid"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values("lift_valid", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(df))
    h = 0.36

    ax.barh(y - h / 2, df["lift_train"], height=h, color=COLORS["teal"], label="Train lift")
    ax.barh(
        y + h / 2,
        df["lift_valid"],
        height=h,
        color=COLORS["orange"],
        label="Validation lift",
    )
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 28) for x in df["segment_name"]])
    ax.set_xlabel("Lift vs global fraud rate")
    ax.set_title("Stable business segments from data mining")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    ax.legend(loc="lower right")

    for i, row in enumerate(df.to_dict("records")):
        ax.text(
            row["lift_valid"] + 0.06,
            i + h / 2,
            f"{row['lift_valid']:.2f}x | fraud {row['fraud_rate_valid'] * 100:.1f}%",
            va="center",
            fontsize=8,
        )

    return _save_current(figures_dir, "02_stable_segment_lift.png")


def plot_stable_numeric_patterns(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "stable_numeric_patterns.csv")
    required = {"feature", "bin", "lift_train", "lift_valid", "fraud_rate_valid"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values("lift_valid", ascending=True).tail(12)
    labels = [f"{row.feature}\n{row.bin}" for row in df.itertuples()]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(df))
    ax.barh(y, df["lift_valid"], color=COLORS["green"], label="Validation lift")
    ax.scatter(df["lift_train"], y, color=COLORS["purple"], s=42, label="Train lift", zorder=3)
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 38) for x in labels])
    ax.set_xlabel("Lift")
    ax.set_title("Stable numeric bins with high fraud lift")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    ax.legend(loc="lower right")

    for i, row in enumerate(df.to_dict("records")):
        ax.text(
            row["lift_valid"] + 0.05,
            i,
            f"{row['lift_valid']:.2f}x | fraud {row['fraud_rate_valid'] * 100:.1f}%",
            va="center",
            fontsize=8,
        )

    return _save_current(figures_dir, "03_stable_numeric_patterns.png")


def plot_stable_categorical_patterns(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "stable_categorical_patterns.csv")
    required = {"feature", "value", "lift_train", "lift_valid", "fraud_rate_valid"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values("lift_valid", ascending=True).tail(12)
    labels = [f"{row.feature}={row.value}" for row in df.itertuples()]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(df))
    ax.barh(y, df["lift_valid"], color=COLORS["blue"], label="Validation lift")
    ax.scatter(df["lift_train"], y, color=COLORS["orange"], s=42, label="Train lift", zorder=3)
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 38) for x in labels])
    ax.set_xlabel("Lift")
    ax.set_title("Stable categorical patterns from data mining")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    ax.legend(loc="lower right")

    for i, row in enumerate(df.to_dict("records")):
        ax.text(
            row["lift_valid"] + 0.05,
            i,
            f"{row['lift_valid']:.2f}x | fraud {row['fraud_rate_valid'] * 100:.1f}%",
            va="center",
            fontsize=8,
        )

    return _save_current(figures_dir, "04_stable_categorical_patterns.png")


def plot_feature_recommendation_rules(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "feature_recommendations_train.csv")
    required = {"rule", "lift", "fraud_rate", "support"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values("lift", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(df))
    colors = np.where(df["fraud_rate"] >= 0.20, COLORS["red"], COLORS["orange"])
    ax.barh(y, df["lift"], color=colors)
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 42) for x in df["rule"]])
    ax.set_xlabel("Lift")
    ax.set_title("Actionable high-lift rules recommended for feature engineering")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)

    for i, row in enumerate(df.to_dict("records")):
        ax.text(
            row["lift"] + 0.3,
            i,
            f"{row['lift']:.1f}x | fraud {row['fraud_rate'] * 100:.1f}% | n={int(row['support']):,}",
            va="center",
            fontsize=8,
        )

    return _save_current(figures_dir, "05_feature_recommendation_rules.png")


def plot_priority_interaction_stability(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "stable_priority_interactions.csv")
    required = {"rule", "lift_train", "lift_valid", "fraud_rate_valid", "support_valid"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.sort_values("lift_valid", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(df))
    ax.barh(y, df["lift_valid"], color=COLORS["teal"], label="Validation lift")
    ax.scatter(df["lift_train"], y, color=COLORS["red"], s=42, label="Train lift", zorder=3)
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 42) for x in df["rule"]])
    ax.set_xlabel("Lift")
    ax.set_title("Stable priority interactions")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)
    ax.legend(loc="lower right")

    for i, row in enumerate(df.to_dict("records")):
        ax.text(
            row["lift_valid"] + 0.08,
            i,
            f"{row['lift_valid']:.1f}x | fraud {row['fraud_rate_valid'] * 100:.1f}% | n={int(row['support_valid']):,}",
            va="center",
            fontsize=8,
        )

    return _save_current(figures_dir, "06_priority_interaction_stability.png")


def plot_thresholds(insight_dir: Path, figures_dir: Path) -> Path | None:
    df = _read_csv(insight_dir / "train_behavior_thresholds.csv")
    if df.empty:
        return None

    row = df.iloc[0].dropna()
    if row.empty:
        return None

    display = pd.DataFrame({"threshold": row.index, "value": row.values})
    display["value"] = pd.to_numeric(display["value"], errors="coerce")
    display = display.dropna()
    display = display[display["value"] > 0]
    if display.empty:
        return None

    display = display.sort_values("value")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(display))
    ax.barh(y, display["value"], color=COLORS["purple"])
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(x, 34) for x in display["threshold"]])
    ax.set_xscale("log")
    ax.set_xlabel("Threshold value fitted on train (log scale)")
    ax.set_title("Behavior thresholds fitted on train only")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)

    for i, row in enumerate(display.to_dict("records")):
        value = float(row["value"])
        label = f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.0f}"
        ax.text(value * 1.08, i, label, va="center", fontsize=8)

    return _save_current(figures_dir, "07_train_behavior_thresholds.png")


def generate_data_mining_visualizations(
    insight_dir: str | Path,
    figures_dir: str | Path,
) -> list[Path]:
    insight_dir = Path(insight_dir)
    figures_dir = Path(figures_dir)
    _setup_style()

    plotters = [
        plot_split_distribution,
        plot_stable_segment_lift,
        plot_stable_numeric_patterns,
        plot_stable_categorical_patterns,
        plot_feature_recommendation_rules,
        plot_priority_interaction_stability,
        plot_thresholds,
    ]

    outputs: list[Path] = []
    for plotter in plotters:
        output = plotter(insight_dir, figures_dir)
        if output is not None:
            outputs.append(output)

    return outputs


def write_visualization_gallery(
    image_paths: list[Path],
    output_path: str | Path,
    project_root: str | Path,
) -> None:
    output_path = Path(output_path)
    project_root = Path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Mining Visualization Gallery",
        "",
        "Các hình này được sinh từ output CSV của `scripts/03_run_insight_mining.py`.",
        "",
    ]

    captions = {
        "01_split_distribution.png": "Split theo thời gian và fraud rate từng tập.",
        "02_stable_segment_lift.png": "Các business segment có lift ổn định trên train/validation.",
        "03_stable_numeric_patterns.png": "Các numeric bin có fraud lift cao và ổn định.",
        "04_stable_categorical_patterns.png": "Các categorical value có fraud lift cao và ổn định.",
        "05_feature_recommendation_rules.png": "Rule lift cao được khuyến nghị chuyển thành feature.",
        "06_priority_interaction_stability.png": "Interaction ưu tiên ổn định giữa train và validation.",
        "07_train_behavior_thresholds.png": "Các behavior thresholds được fit từ train.",
    }

    for image_path in image_paths:
        rel = image_path.relative_to(output_path.parent).as_posix()
        caption = captions.get(image_path.name, image_path.stem)
        lines.extend([f"## {caption}", "", f"![{caption}]({rel})", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
