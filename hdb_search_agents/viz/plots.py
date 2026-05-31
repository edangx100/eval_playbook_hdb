"""
Matplotlib visualization helpers for price distribution charts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from hdb_search_agents.agent.models import Stats


def plot_price_hist(
    prices: list[int],
    stats: Stats,
    town: str | None = None,
    flat_type: str | None = None,
    months_back: int | None = None,
) -> matplotlib.figure.Figure:
    """
    Generate a histogram of resale prices with median and quartile overlays.

    Args:
        prices: List of resale prices (in SGD)
        stats: Stats object with median, p25, p75, min, max, count
        town: HDB town name for title
        flat_type: Flat type for title (e.g., "4 ROOM")
        months_back: Lookback window in months for title

    Returns:
        matplotlib Figure object ready for display/saving
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate histogram with auto bins
    if len(prices) > 0:
        # Use "auto" binning strategy, but cap at 30 bins for readability
        n_bins = min(30, max(10, len(prices) // 10))
        ax.hist(prices, bins=n_bins, color="#3498db", alpha=0.7, edgecolor="black")

        # Add vertical lines for median and quartiles
        ax.axvline(
            stats.median,
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            label=f"Median: ${stats.median:,.0f}",
        )
        ax.axvline(
            stats.p25,
            color="#f39c12",
            linestyle=":",
            linewidth=1.5,
            label=f"P25: ${stats.p25:,.0f}",
        )
        ax.axvline(
            stats.p75,
            color="#f39c12",
            linestyle=":",
            linewidth=1.5,
            label=f"P75: ${stats.p75:,.0f}",
        )

        # Add legend
        ax.legend(loc="upper right", fontsize=10)

    # Build title with available context
    title_parts = ["Resale Price Distribution"]
    if town:
        title_parts.append(f"| {town}")
    if flat_type:
        title_parts.append(f"| {flat_type}")
    if months_back:
        title_parts.append(f"| Last {months_back} months")
    title_parts.append(f"| n={stats.count}")

    title = " ".join(title_parts)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Labels and formatting
    ax.set_xlabel("Resale Price (SGD)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Format x-axis with thousands separator
    ax.ticklabel_format(style="plain", axis="x")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )

    # Tight layout for better spacing
    fig.tight_layout()

    return fig
