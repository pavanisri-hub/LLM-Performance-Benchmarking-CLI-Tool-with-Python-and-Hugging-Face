from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


PALETTE = "viridis"
STYLE = "darkgrid"
FIG_SIZE = (10, 6)
DPI = 150


def generate_all_charts(df: pd.DataFrame, output_dir: str) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style=STYLE, palette=PALETTE)

    saved: list[str] = []

    saved.append(_bar_chart(
        df=df,
        metric_col="Latency_Seconds",
        agg="mean",
        title="Average Inference Latency per Model",
        ylabel="Latency (seconds)",
        filename="avg_latency_comparison.png",
        output_dir=out,
        color_start="#4C72B0",
        color_end="#55A868",
    ))

    saved.append(_bar_chart(
        df=df,
        metric_col="Peak_RAM_MB",
        agg="max",
        title="Peak RAM Usage per Model",
        ylabel="Peak RAM (MB)",
        filename="peak_ram_comparison.png",
        output_dir=out,
        color_start="#C44E52",
        color_end="#DD8452",
    ))

    saved.append(_bar_chart(
        df=df,
        metric_col="Throughput_TPS",
        agg="mean",
        title="Average Throughput per Model (Tokens / Second)",
        ylabel="Throughput (TPS)",
        filename="avg_throughput_comparison.png",
        output_dir=out,
        color_start="#8172B2",
        color_end="#937860",
    ))

    saved.append(_bar_chart(
        df=df,
        metric_col="Quality_Metric_Score",
        agg="mean",
        title="Average Quality Score (TTR) per Model",
        ylabel="Type-Token Ratio (0-1)",
        filename="quality_score_comparison.png",
        output_dir=out,
        color_start="#64B5CD",
        color_end="#6DBC71",
    ))

    return saved


def _bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    agg: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    color_start: str,
    color_end: str,
) -> str:
    grouped = df.groupby("Model_Name", sort=False)[metric_col]
    if agg == "mean":
        summary = grouped.mean().reset_index()
    else:
        summary = grouped.max().reset_index()

    summary.columns = ["Model_Name", metric_col]
    summary = summary.sort_values(metric_col, ascending=False)

    n = len(summary)
    colors = _interpolate_colors(color_start, color_end, n)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    bars = ax.bar(
        summary["Model_Name"],
        summary[metric_col],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Model", fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", rotation=15)

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _interpolate_colors(hex_start: str, hex_end: str, n: int) -> list[str]:
    if n == 1:
        return [hex_start]

    def _parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r0, g0, b0 = _parse(hex_start)
    r1, g1, b1 = _parse(hex_end)
    result = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        r = int(r0 + t * (r1 - r0))
        g = int(g0 + t * (g1 - g0))
        b = int(b0 + t * (b1 - b0))
        result.append(f"#{r:02x}{g:02x}{b:02x}")
    return result
