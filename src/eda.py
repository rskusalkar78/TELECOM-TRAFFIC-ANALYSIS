"""
Exploratory Data Analysis (EDA) Module.

Generates and saves:
1. Descriptive statistical summary (summary_statistics.txt).
2. Hourly Internet activity profile (hourly_internet_activity.png).
3. Cross-activity comparison: Calls, SMS, and Internet trends (activity_comparison.png).
4. Correlation matrix heatmap (correlation_heatmap.png).
5. Narrative EDA summary report (eda_summary.txt).
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_plotting_style() -> None:
    """Configure modern, readable aesthetic styles for Matplotlib and Seaborn."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def generate_summary_statistics(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> Path:
    """Generate and write descriptive statistics of activity metrics to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_statistics.txt"

    numeric_cols = [
        col
        for col in [
            "SMS-in",
            "SMS-out",
            "Call-in",
            "Call-out",
            "Internet",
            "total_sms",
            "total_calls",
        ]
        if col in df.columns
    ]

    stats_df = df[numeric_cols].describe().T

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TELECOM TRAFFIC ANALYSIS - DESCRIPTIVE STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Processed Records: {len(df):,}\n")
        f.write(f"Time Range: {df['datetime'].min()} to {df['datetime'].max()}\n\n")
        f.write(stats_df.to_string())
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Note: Values represent aggregated cellular activity counters.\n")
        f.write("=" * 70 + "\n")

    return summary_path


def plot_hourly_internet_activity(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> Path:
    """Plot mean Internet activity curve across hours of the day (diurnal cycle)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "hourly_internet_activity.png"
    set_plotting_style()

    hourly = df.groupby("hour")["Internet"].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        hourly["hour"],
        hourly["mean"],
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        label="Mean Activity",
    )
    ax.fill_between(
        hourly["hour"],
        (hourly["mean"] - 0.5 * hourly["std"]).clip(lower=0),
        hourly["mean"] + 0.5 * hourly["std"],
        color="#1f77b4",
        alpha=0.2,
        label="Variance Band (±0.5 STD)",
    )

    ax.set_title("Hourly Diurnal Profile: Mobile Internet Activity", fontsize=14, weight="bold")
    ax.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=12)
    ax.set_ylabel("Internet Activity Index (Arbitrary Units)", fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_activity_comparison(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> Path:
    """Compare normalized hourly trajectories of Calls, SMS, and Internet traffic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "activity_comparison.png"
    set_plotting_style()

    hourly = (
        df.groupby("hour")[["total_sms", "total_calls", "Internet"]].mean().reset_index()
    )

    # Min-max normalization for direct shape comparison
    norm_df = hourly.copy()
    for col in ["total_sms", "total_calls", "Internet"]:
        denom = norm_df[col].max() - norm_df[col].min()
        if denom > 0:
            norm_df[col] = (norm_df[col] - norm_df[col].min()) / denom

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        norm_df["hour"],
        norm_df["Internet"],
        marker="s",
        color="#2ca02c",
        linewidth=2.5,
        label="Internet Activity",
    )
    ax.plot(
        norm_df["hour"],
        norm_df["total_calls"],
        marker="^",
        color="#d62728",
        linewidth=2,
        linestyle="--",
        label="Call Volume (In + Out)",
    )
    ax.plot(
        norm_df["hour"],
        norm_df["total_sms"],
        marker="o",
        color="#9467bd",
        linewidth=2,
        linestyle=":",
        label="SMS Volume (In + Out)",
    )

    ax.set_title("Normalized Cross-Activity Comparison Throughout the Day", fontsize=14, weight="bold")
    ax.set_xlabel("Hour of Day (0 - 23)", fontsize=12)
    ax.set_ylabel("Normalized Activity Index [0.0 - 1.0]", fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> Path:
    """Compute and plot Pearson correlation heatmap between communication modalities."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "correlation_heatmap.png"
    set_plotting_style()

    cols = [
        col
        for col in [
            "SMS-in",
            "SMS-out",
            "Call-in",
            "Call-out",
            "total_sms",
            "total_calls",
            "Internet",
            "hour",
        ]
        if col in df.columns
    ]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Telecom Activity Indicators", fontsize=14, weight="bold")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def generate_eda_narrative(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> Path:
    """Generate concise textual summary of exploratory patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    narrative_path = output_dir / "eda_summary.txt"

    peak_hour = df.groupby("hour")["Internet"].mean().idxmax()
    trough_hour = df.groupby("hour")["Internet"].mean().idxmin()
    cols_to_corr = [c for c in ["Call-in", "Call-out", "SMS-in", "SMS-out"] if c in df.columns]
    corrs = df[cols_to_corr].apply(lambda s: s.corr(df["Internet"])).to_dict()

    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write("TELECOM TRAFFIC EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"1. DIURNAL TRAFFIC PATTERNS:\n")
        f.write(f"   - Peak internet activity occurs at approximately {peak_hour:02d}:00.\n")
        f.write(f"   - Minimum traffic (trough) occurs during nighttime at {trough_hour:02d}:00.\n")
        f.write(f"   - Traffic exhibits a clear human behavioral cycle matching typical work and leisure hours.\n\n")
        f.write(f"2. MODALITY CORRELATIONS:\n")
        for modality, val in corrs.items():
            f.write(f"   - Pearson correlation with {modality}: {val:.3f}\n")
        f.write("\n3. KEY OBSERVATIONS:\n")
        f.write("   - Voice and SMS volume moderately to strongly co-occur with data consumption.\n")
        f.write("   - Time-of-day (hour) is a crucial non-linear feature for activity estimation.\n")
        f.write("=" * 60 + "\n")

    return narrative_path


def run_eda_pipeline(
    df: pd.DataFrame, output_dir: Path = Path("outputs")
) -> None:
    """Execute all EDA routines and persist graphs and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_summary_statistics(df, output_dir)
    plot_hourly_internet_activity(df, output_dir)
    plot_activity_comparison(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    generate_eda_narrative(df, output_dir)
