"""
Calculate win rates and confidence intervals
Compare incumbent and non-incumbent win rates and make a bar chart.

Usage:
    python signed_margin_analysis/scripts/calculate_win_rates_and_confidence_intervals.py input.csv
    python signed_margin_analysis/scripts/calculate_win_rates_and_confidence_intervals.py input.csv --output summary.csv

The analysis keeps poll_type == 0 and excludes rows without a known incumbent
value. It does not apply a deposit_lost filter; results reflect whichever rows
are present in the input file. Pipeline files may already exclude deposit-lost
candidates, so check the selected input file's scope before interpreting rates.
SciPy supplies the critical value used to calculate the confidence intervals.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
try:
    from scipy.stats import norm
except ImportError:
    norm = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / "data_files" / "extended.csv"


def convert_to_boolean(values):
    """Read common true/false spellings from the incumbent column."""
    values = values.astype("string").str.strip().str.lower()
    return values.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )


def wilson_interval(wins, total, z):
    """Estimate a confidence range for one group's win rate."""
    if total == 0:
        return (float("nan"), float("nan"))

    # The observed win rate is the share of candidates who won.
    proportion = wins / total
    z_squared = z**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z
        * ((proportion * (1 - proportion) / total + z_squared / (4 * total**2)) ** 0.5)
        / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def newcombe_difference_interval(p1, interval1, p2, interval2):
    """Estimate a confidence range for the difference between two win rates."""
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    difference = p1 - p2
    lower = difference - ((p1 - lower1) ** 2 + (upper2 - p2) ** 2) ** 0.5
    upper = difference + ((upper1 - p1) ** 2 + (p2 - lower2) ** 2) ** 0.5
    return (lower, upper)


def save_win_rate_plot(
    incumbent,
    non_incumbent,
    difference,
    difference_ci,
    confidence,
    output_path,
):
    """Save a bar chart; the error bars show each estimate's confidence range."""
    labels = [
        "Incumbent win rate",
        "Non-incumbent win rate",
        "Win-rate difference",
    ]
    estimates = [
        incumbent["win_rate"] * 100,
        non_incumbent["win_rate"] * 100,
        difference * 100,
    ]
    lower_bounds = [
        incumbent["ci_lower"] * 100,
        non_incumbent["ci_lower"] * 100,
        difference_ci[0] * 100,
    ]
    upper_bounds = [
        incumbent["ci_upper"] * 100,
        non_incumbent["ci_upper"] * 100,
        difference_ci[1] * 100,
    ]
    colors = ["#2166ac", "#b2182b", "#4d9221"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = range(len(labels))
    # Each bar is the observed rate or the difference; whiskers show its CI.
    ax.bar(
        positions,
        estimates,
        color=colors,
        width=0.62,
        yerr=[
            [estimate - lower for estimate, lower in zip(estimates, lower_bounds)],
            [upper - estimate for estimate, upper in zip(estimates, upper_bounds)],
        ],
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%) / difference (percentage points)")
    ax.set_title(f"Incumbent and non-incumbent results ({confidence:.0%} CI)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for position, estimate in zip(positions, estimates):
        ax.text(position, estimate + 2, f"{estimate:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare incumbent and non-incumbent candidate win rates."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Candidate-level election CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a one-row CSV summary",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PROJECT_DIR / "visualisations" / "incumbent_win_rates_with_confidence_intervals.png",
        help="Path for the win-rate chart",
    )
    args = parser.parse_args()

    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")

    input_file = args.input_csv.resolve()
    candidates = pd.read_csv(input_file, low_memory=False)
    required = {"poll_type", "position", "incumbent", "margin_percentage"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    candidates["poll_type"] = pd.to_numeric(candidates["poll_type"], errors="coerce")
    candidates["position"] = pd.to_numeric(candidates["position"], errors="coerce")
    candidates["margin_percentage"] = pd.to_numeric(
        candidates["margin_percentage"], errors="coerce"
    )
    candidates["is_incumbent"] = convert_to_boolean(candidates["incumbent"])
    missing_incumbent = int(candidates["is_incumbent"].isna().sum())

    # Keep regular general elections and rows where incumbent status is known.
    candidates = candidates.loc[
        (candidates["poll_type"] == 0) & candidates["is_incumbent"].notna()
    ].copy()
    missing_margin_by_group = {
        label: int(
            (
                candidates["is_incumbent"].eq(status)
                & candidates["margin_percentage"].isna()
            ).sum()
        )
        for label, status in (("incumbent", True), ("non_incumbent", False))
    }
    candidates = candidates.dropna(subset=["position", "margin_percentage"])
    candidates["won"] = candidates["position"] == 1

    critical_value = (
        float(norm.ppf(1 - (1 - args.confidence) / 2))
        if norm is not None
        else 1.959963984540054
    )
    results = {}
    for label, is_incumbent in [("incumbent", True), ("non_incumbent", False)]:
        # Count candidates and winners separately for each group.
        group = candidates.loc[candidates["is_incumbent"] == is_incumbent]
        candidate_count = len(group)
        win_count = int(group["won"].sum())
        rate = win_count / candidate_count if candidate_count else float("nan")
        interval = wilson_interval(win_count, candidate_count, critical_value)
        results[label] = {
            "candidate_entries": candidate_count,
            "wins": win_count,
            "missing_margin_excluded": missing_margin_by_group[label],
            "win_rate": rate,
            "ci_lower": interval[0],
            "ci_upper": interval[1],
        }

    incumbent = results["incumbent"]
    non_incumbent = results["non_incumbent"]
    difference = incumbent["win_rate"] - non_incumbent["win_rate"]
    difference_ci = newcombe_difference_interval(
        incumbent["win_rate"],
        (incumbent["ci_lower"], incumbent["ci_upper"]),
        non_incumbent["win_rate"],
        (non_incumbent["ci_lower"], non_incumbent["ci_upper"]),
    )
    save_win_rate_plot(
        incumbent,
        non_incumbent,
        difference,
        difference_ci,
        args.confidence,
        args.plot_output,
    )
    summary = {
        "confidence_level": args.confidence,
        "poll_type_included": 0,
        "margin_percentage_nonmissing_required": True,
        "incumbent_candidate_entries": incumbent["candidate_entries"],
        "incumbent_wins": incumbent["wins"],
        "incumbent_missing_margin_exclusions": incumbent["missing_margin_excluded"],
        "incumbent_win_rate": incumbent["win_rate"],
        "incumbent_ci_lower": incumbent["ci_lower"],
        "incumbent_ci_upper": incumbent["ci_upper"],
        "non_incumbent_candidate_entries": non_incumbent["candidate_entries"],
        "non_incumbent_wins": non_incumbent["wins"],
        "non_incumbent_missing_margin_exclusions": non_incumbent["missing_margin_excluded"],
        "non_incumbent_win_rate": non_incumbent["win_rate"],
        "non_incumbent_ci_lower": non_incumbent["ci_lower"],
        "non_incumbent_ci_upper": non_incumbent["ci_upper"],
        "incumbent_minus_non_incumbent_difference": difference,
        "difference_ci_lower": difference_ci[0],
        "difference_ci_upper": difference_ci[1],
        "rows_excluded_for_missing_incumbent_status": missing_incumbent,
    }

    print(f"Input file: {input_file}")
    print("Common complete-case rule: poll_type=0, known incumbent status, numeric position, nonblank margin_percentage")
    print(f"Confidence level: {args.confidence:.0%}")
    for label, title in [
        ("incumbent", "Incumbent win rate"),
        ("non_incumbent", "Non-incumbent win rate"),
    ]:
        result = results[label]
        print(
            f"{title}: {result['wins']:,}/{result['candidate_entries']:,} = "
            f"{result['win_rate']:.2%} "
            f"(95% CI: {result['ci_lower']:.2%} to {result['ci_upper']:.2%}); "
            f"candidate sample n={result['candidate_entries']:,}; "
            f"missing-margin exclusions={result['missing_margin_excluded']:,}"
        )
    print(
        "Win-rate difference (incumbent minus non-incumbent): "
        f"{difference:.2%} "
        f"(95% CI: {difference_ci[0]:.2%} to {difference_ci[1]:.2%})"
    )
    print(f"Chart saved to: {args.plot_output}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.output, index=False)
        print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
