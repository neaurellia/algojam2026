"""Plot every price history as small multiples, one PNG, months on the x-axis.

Nine instruments spanning $1.87 to $150,789 cannot share a y-axis, so each gets
its own panel and its own scale. One series per panel means the title names it
and no legend is needed.

    python plot_prices.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from simulation import positionLimits

DATA_DIR = Path("data/")
OUTPUT = Path("price_history.png")

# Day index on which each month starts, for a 365-day non-leap year beginning
# 1 January. Months are uneven, so these are cumulative day counts rather than
# a flat every-30 step -- which is what makes the labels actually line up with
# the months they name.
MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Chart chrome. Ink and gridlines stay recessive so the data line is the only
# thing with any visual weight.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"


def load_prices():
    prices = {}
    for file in sorted(DATA_DIR.iterdir()):
        if file.name.endswith("_price_history.csv"):
            instrument = file.name.split("_price_history")[0]
            if instrument in positionLimits:
                prices[instrument] = pd.read_csv(file)["Price"].astype(float)
    return pd.DataFrame(prices)


def tick_formatter(axis):
    """Build a y-tick formatter with just enough decimals to stay unambiguous.

    Precision has to come from the TICK SPACING, not the magnitude: UQ Dollar
    sits near $100 with ticks half a dollar apart, so rounding to whole dollars
    prints "$100" twice. Pick the fewest decimals that keep adjacent ticks
    distinct, and switch to K notation once the numbers get long.
    """
    ticks = axis.get_yticks()
    spacing = min((b - a for a, b in zip(ticks, ticks[1:])), default=1.0)
    thousands = max(abs(t) for t in ticks) >= 10_000

    decimals = 0
    if not thousands:
        while decimals < 2 and round(spacing, decimals) == 0:
            decimals += 1

    def format_tick(value, _position):
        if thousands:
            return f"${value / 1000:,.0f}K"
        return f"${value:,.{decimals}f}"

    return plt.FuncFormatter(format_tick)


def format_range(value):
    """Compact price for the per-panel range caption."""
    if abs(value) >= 10_000:
        return f"${value / 1000:,.0f}K"
    if abs(value) >= 100:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def main():
    prices = load_prices()
    names = list(prices.columns)
    columns = 3
    rows = -(-len(names) // columns)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
    })
    figure, axes = plt.subplots(rows, columns, figsize=(16, 9.5),
                                facecolor=SURFACE)
    # Generous top and hspace: each panel carries a two-line heading, so the
    # rows need clearance the default grid does not give them.
    figure.subplots_adjust(hspace=0.62, wspace=0.24,
                           left=0.05, right=0.98, top=0.86, bottom=0.06)

    for index, axis in enumerate(axes.flat):
        if index >= len(names):
            axis.axis("off")
            continue

        name = names[index]
        series = prices[name]
        axis.set_facecolor(SURFACE)
        axis.plot(series.index, series.values, color=SERIES, linewidth=2,
                  solid_capstyle="round", zorder=3)

        # Recessive chrome: hairline horizontal grid only, no top/right spines.
        axis.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(BASELINE)
        axis.tick_params(colors=INK_MUTED, labelsize=7, length=3, color=BASELINE)

        axis.set_xticks(MONTH_STARTS)
        axis.set_xticklabels(MONTH_LABELS, fontsize=7)
        axis.set_xlim(0, len(series) - 1)
        # Formatter is built from the ticks, so it has to come after the data
        # and limits are in place.
        axis.yaxis.set_major_formatter(tick_formatter(axis))

        axis.set_title(name, fontsize=10, fontweight="bold",
                       color=INK_PRIMARY, loc="left", pad=20)
        # Range and limit as a muted caption rather than another chart element.
        axis.text(0, 1.04,
                  f"{format_range(series.min())} – {format_range(series.max())}"
                  f"   ·   limit {positionLimits[name]:,}",
                  transform=axis.transAxes, fontsize=7, color=INK_MUTED,
                  va="bottom")

    figure.suptitle("AlgoJam 2026 — Round 1 price histories",
                    fontsize=14, fontweight="bold", color=INK_PRIMARY,
                    x=0.05, y=0.975, ha="left")
    figure.text(0.05, 0.945,
                f"{len(prices)} days, January to December. Each panel has its "
                f"own price scale.",
                fontsize=8.5, color=INK_SECONDARY, ha="left")

    figure.savefig(OUTPUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {OUTPUT.resolve()}  ({len(names)} instruments, {len(prices)} days)")


if __name__ == "__main__":
    main()
