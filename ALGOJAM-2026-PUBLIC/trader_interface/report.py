"""Evaluate the strategy book and write a timestamped report.

Replays the real Algorithm class day by day, exactly as simulation.py does, so
the numbers here always describe the code that would actually trade. Nothing
about the signals is reimplemented in this file.

Every run saves the full report to reports/<YYYY-MM-DD_HHMM>.txt and appends a
row to reports/history.csv, then prints a one-line summary short enough to use
as a commit message.

Usage:
    python report.py                 # full report -> file, one-liner -> stdout
    python report.py --quiet         # only the one-liner
    python report.py --no-save       # print, do not write files
    python report.py --steps A/A,B   # custom incremental steps, '/'-separated
"""
import argparse
import contextlib
import csv
import datetime
import importlib
import io
from pathlib import Path

import pandas as pd

from simulation import (positionLimits, totalDailyBudget,
                        NO_LOCAL_PNL_INSTRUMENTS, Algorithm)

# Algorithm above is deliberately imported FROM simulation rather than from a
# module named here: whatever simulation.py runs is what this scores, so the
# two can never quietly describe different strategies. --algo overrides it to
# compare the per-author files against each other.

# Shown for classes that predate SPEC, i.e. the imported per-author files.
UNSPECIFIED = ("external", "see module source", None)

# Day index that splits the fitted half from the held-out half.
SPLIT_DAY = 182

# Where timestamped reports and the running history are kept.
REPORT_DIR = Path("reports")
HISTORY_CSV = REPORT_DIR / "history.csv"


def load_prices(dataFolder="data/"):
    prices = {}
    for file in Path(dataFolder).iterdir():
        if file.name.endswith("_price_history.csv"):
            instrument = file.name.split("_price_history")[0]
            if instrument in positionLimits:
                prices[instrument] = pd.read_csv(file)["Price"].astype(float)
    return pd.DataFrame(prices)


def replay(prices, enabled):
    """Run the Algorithm over every day and return the positions it asked for.

    Mirrors TradingEngine.run_algorithms: on each day the algorithm sees prices
    up to and including that day, and the position it returns is held into the
    next day. The algorithm's own logging is swallowed so the report stays clean.
    """
    algo = Algorithm({instrument: 0 for instrument in positionLimits})
    # Only classes that declare ENABLED can be restricted to a subset; the
    # per-author files always trade their own fixed set.
    if hasattr(algo, "ENABLED") and enabled is not None:
        algo.ENABLED = list(enabled)
    positions = {instrument: 0 for instrument in positionLimits}
    rows = []
    for day in range(len(prices)):
        algo.day = day
        algo.data = {name: prices[name][:day + 1].tolist() for name in prices.columns}
        algo.positions = positions
        algo.positionLimits = positionLimits
        with contextlib.redirect_stdout(io.StringIO()):
            positions = algo.get_positions()
        rows.append(dict(positions))
    return pd.DataFrame(rows, index=prices.index).fillna(0)


def measure(prices, positions, instruments):
    """P&L, capture and budget use for a set of instruments.

    A position held on day t is paid the move from day t to day t+1, so the
    final row earns nothing and is excluded. Capture is realised P&L over what
    perfect next-day foresight would have earned at the same position limits.
    """
    moves = prices.shift(-1) - prices
    pnl = pd.Series(0.0, index=prices.index)
    perfect = pd.Series(0.0, index=prices.index)
    exposure = pd.Series(0.0, index=prices.index)
    for name in instruments:
        # Exposure is counted for EVERY instrument, including ones that pay no
        # local P&L: notWithinBudget charges them against the daily cap all the
        # same, and Liferaft at one unit is ~$150K of it.
        exposure += (positions[name].abs() * prices[name]).fillna(0)
        if name in NO_LOCAL_PNL_INSTRUMENTS:
            continue
        pnl += (positions[name] * moves[name]).fillna(0)
        perfect += (moves[name].abs() * positionLimits[name]).fillna(0)
    fit, test = slice(0, SPLIT_DAY), slice(SPLIT_DAY, len(prices) - 1)
    return dict(
        fit=pnl[fit].sum(), test=pnl[test].sum(), total=pnl.sum(),
        fit_cap=pnl[fit].sum() / perfect[fit].sum(),
        test_cap=pnl[test].sum() / perfect[test].sum(),
        peak_budget=exposure.max(), mean_budget=exposure.mean(),
    )


def money(value, width=11):
    return f"${value:,.0f}".rjust(width)


def spec_for(name):
    """Signal description for one instrument, or a placeholder if undeclared."""
    return getattr(Algorithm, "SPEC", {}).get(name, UNSPECIFIED)


def one_liner(enabled, final):
    """Short settings + profit summary, sized for a commit subject line."""
    parts = [f"{name.lower()} {spec_for(name)[0].replace(' = ', '=')}"
             for name in enabled]
    return f"{', '.join(parts)} | profit ${final['total']:,.0f}"


def settings_lines(enabled):
    lines = ["SETTINGS",
             f"  {'Instrument':<20}{'Setting':<12}{'Signal':<30}{'Limit':>8}{'Params':>8}",
             "  " + "-" * 76]
    total_params = 0
    for name in enabled:
        setting, signal, params = spec_for(name)
        total_params += params or 0
        lines.append(f"  {name:<20}{setting:<12}{signal:<30}"
                     f"{positionLimits[name]:>8,}"
                     f"{'-' if params is None else params:>8}")
    for name in positionLimits:
        if name not in enabled:
            lines.append(f"  {name:<20}{'flat':<12}{'0':<30}{'-':>8}{'-':>8}")
    lines += ["  " + "-" * 76,
              f"  {'TOTAL FITTED PARAMS':<70}{total_params:>8}"]
    return lines


def results_lines(rows):
    lines = ["RESULTS  (FIT = days 0-181, TEST = days 182-363)",
             f"  {'Step':<26}{'FIT':>12}{'TEST':>12}{'Total':>12}"
             f"{'FitCap':>9}{'TestCap':>9}{'Gap':>9}{'PeakBudget':>13}",
             "  " + "-" * 102]
    for label, m in rows:
        gap = (m["fit_cap"] - m["test_cap"]) * 100
        lines.append(f"  {label:<26}{money(m['fit'], 12)}{money(m['test'], 12)}"
                     f"{money(m['total'], 12)}{m['fit_cap'] * 100:>8.1f}%"
                     f"{m['test_cap'] * 100:>8.1f}%{gap:>7.1f}pp"
                     f"{money(m['peak_budget'], 13)}")
    return lines


def per_instrument_lines(prices, positions, enabled):
    lines = ["PER-INSTRUMENT",
             f"  {'Instrument':<20}{'FIT':>12}{'TEST':>12}{'Total':>12}"
             f"{'TestCap':>9}{'Flips':>8}{'MeanExp':>12}",
             "  " + "-" * 85]
    for name in enabled:
        m = measure(prices, positions, [name])
        flips = int((positions[name].diff().fillna(0) != 0).sum())
        lines.append(f"  {name:<20}{money(m['fit'], 12)}{money(m['test'], 12)}"
                     f"{money(m['total'], 12)}{m['test_cap'] * 100:>8.1f}%{flips:>8}"
                     f"{money(m['mean_budget'], 12)}")
    return lines


def build_report(prices, enabled, results, stamp):
    final = results[-1][1]
    lines = ["=" * 104,
             f"ALGOJAM 2026 STRATEGY REPORT   {stamp:%Y-%m-%d %H:%M}",
             f"scoring {Algorithm.__module__}.py",
             "=" * 104, ""]
    lines += settings_lines(enabled) + [""]
    lines += results_lines(results) + [""]
    lines += per_instrument_lines(prices, replay(prices, enabled), enabled) + [""]
    lines += [f"  Budget: peak ${final['peak_budget']:,.0f} / ${totalDailyBudget:,} cap "
              f"({final['peak_budget'] / totalDailyBudget:.1%}), "
              f"${totalDailyBudget - final['peak_budget']:,.0f} headroom",
              "", "  " + one_liner(enabled, final), "=" * 104]
    return "\n".join(lines)


def append_history(stamp, enabled, final, filename):
    """One row per run, so progress across experiments stays comparable."""
    is_new = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["timestamp", "report", "book", "params", "fit", "test",
                             "total", "fit_cap", "test_cap", "gap_pp", "peak_budget"])
        writer.writerow([
            f"{stamp:%Y-%m-%d %H:%M}", filename, "; ".join(enabled),
            sum(spec_for(n)[2] or 0 for n in enabled),
            round(final["fit"], 2), round(final["test"], 2), round(final["total"], 2),
            round(final["fit_cap"] * 100, 1), round(final["test_cap"] * 100, 1),
            round((final["fit_cap"] - final["test_cap"]) * 100, 1),
            round(final["peak_budget"], 2),
        ])


def build_steps(spec, enabled):
    """Decide which incremental books to report.

    Without --steps, report the default build-up truncated to whatever is
    enabled. With it, each '/'-separated chunk is one step, labelled by the
    instruments it adds so the column stays readable however long the lists get.
    """
    if not spec:
        # Build the book up one instrument at a time in ENABLED order, so the
        # labels always name what was actually added rather than a fixed list
        # that goes stale whenever the book changes.
        steps = []
        for count in range(1, len(enabled) + 1):
            names = enabled[:count]
            label = f"{count}. " + ("+ " + names[-1] if count > 1 else names[0])
            steps.append((label[:26], names))
        return steps

    steps, previous = [], set()
    for chunk in spec.split("/"):
        names = [n.strip() for n in chunk.split(",") if n.strip()]
        added = [n for n in names if n not in previous]
        label = ("+ " + " + ".join(added)) if previous else ", ".join(names)
        steps.append((f"{len(steps) + 1}. {label}"[:26], names))
        previous = set(names)
    return steps


def traded_instruments(prices):
    """Which instruments the loaded class actually takes a position on.

    Classes that declare ENABLED say so directly. For the per-author files,
    which predate that convention, replay once and keep whatever they touch.
    """
    declared = getattr(Algorithm, "ENABLED", None)
    if declared:
        return list(declared)
    positions = replay(prices, None)
    return [name for name in prices.columns if (positions[name] != 0).any()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default=None,
                        help="module to score, e.g. algorithm_catherine "
                             "(default: whatever simulation.py imports)")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the one-line summary")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write the report or history files")
    parser.add_argument("--steps",
                        help="'/'-separated steps, each a comma-separated instrument list")
    args = parser.parse_args()

    if args.algo:
        global Algorithm
        Algorithm = importlib.import_module(args.algo).Algorithm

    prices = load_prices()
    enabled = traded_instruments(prices)

    steps = build_steps(args.steps, enabled)
    enabled = steps[-1][1]

    results = [(label, measure(prices, replay(prices, names), names))
               for label, names in steps]
    final = results[-1][1]

    stamp = datetime.datetime.now()
    report = build_report(prices, enabled, results, stamp)

    if not args.quiet:
        print(report)

    if not args.no_save:
        REPORT_DIR.mkdir(exist_ok=True)
        filename = f"{stamp:%Y-%m-%d_%H%M}.txt"
        with open(REPORT_DIR / filename, "w") as handle:
            handle.write(report + "\n")
        append_history(stamp, enabled, final, filename)
        if not args.quiet:
            print(f"\nSaved {REPORT_DIR / filename} "
                  f"and appended to {HISTORY_CSV}")

    if args.quiet:
        print(one_liner(enabled, final))


if __name__ == "__main__":
    main()
