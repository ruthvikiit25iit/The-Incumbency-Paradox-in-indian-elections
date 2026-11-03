"""Compare candidates after narrowly winning or losing.

Run from the repository root:
    python scripts/close_election_analysis.py

Append the printed analysis to results.txt in PowerShell:
    python scripts/close_election_analysis.py | Add-Content -Path results.txt -Encoding utf8

The script uses data_files/extended.csv to identify the main election year for
each assembly and sources/general_elections.csv to reconstruct prior margins
and follow-up participation (including candidates who forfeited their deposit).
It saves two figures under visualisations/.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.stats import norm
except ImportError:  # Keep the script runnable where SciPy is not installed.
    norm = None


ROOT = Path(__file__).resolve().parents[1]
EXTENDED_FILE = ROOT / "data_files" / "extended.csv"
SOURCE_FILE = ROOT / "sources" / "general_elections.csv"
PLOT_DIR = ROOT / "visualisations"
OUTCOME_PLOT = PLOT_DIR / "subsequent_outcome_distribution.png"
WIN_RATE_PLOT = PLOT_DIR / "subsequent_win_rates_ci.png"
THRESHOLDS = (2.0, 5.0)
CONFIDENCE = 0.95
OUTCOMES = ("Won", "Lost", "Did not contest")


def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def newcombe_difference_interval(
    p1: float,
    ci1: tuple[float, float],
    p2: float,
    ci2: tuple[float, float],
) -> tuple[float, float]:
    """Newcombe interval for the difference of two independent proportions."""
    lower = (p1 - p2) - math.sqrt((p1 - ci1[0]) ** 2 + (ci2[1] - p2) ** 2)
    upper = (p1 - p2) + math.sqrt((ci1[1] - p1) ** 2 + (p2 - ci2[0]) ** 2)
    return lower, upper


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EXTENDED_FILE.exists():
        raise FileNotFoundError(f"Extended data file not found: {EXTENDED_FILE}")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Raw source file not found: {SOURCE_FILE}")

    extended = pd.read_csv(EXTENDED_FILE, low_memory=False)
    required_extended = {
        "year",
        "assembly_number",
    }
    missing = required_extended - set(extended.columns)
    if missing:
        raise ValueError(f"extended.csv is missing columns: {', '.join(sorted(missing))}")

    source = pd.read_csv(
        SOURCE_FILE,
        usecols=[
            "year", "assembly_number", "poll_type", "candidate_id", "position",
            "state_acronym", "constituency_number", "delimitation_id",
            "vote_share_percentage", "margin_percentage", "votes", "valid_votes",
        ],
        low_memory=False,
    )
    return extended, source


def prepare_data(
    extended: pd.DataFrame, source: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, int], int]:
    extended["assembly_number"] = pd.to_numeric(extended["assembly_number"], errors="coerce")
    extended["year"] = pd.to_numeric(extended["year"], errors="coerce")
    for column in [
        "year", "assembly_number", "poll_type", "position",
        "vote_share_percentage", "margin_percentage", "votes", "valid_votes",
    ]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["candidate_id"] = source["candidate_id"].astype("string").str.strip()
    source.loc[source["candidate_id"].eq(""), "candidate_id"] = pd.NA
    source["poll_type"] = pd.to_numeric(source["poll_type"], errors="coerce")

    # A Lok Sabha term can contain a small number of by-election records in
    # later years. Use the year with the most retained candidate entries as
    # that assembly's primary election year.
    yearly_counts = (
        extended.dropna(subset=["assembly_number", "year"])
        .groupby(["assembly_number", "year"])
        .size()
        .rename("entries")
        .reset_index()
        .sort_values(
            ["assembly_number", "entries", "year"],
            ascending=[True, False, True],
        )
    )
    canonical_year = (
        yearly_counts.drop_duplicates("assembly_number")
        .set_index("assembly_number")["year"]
        .astype(int)
        .to_dict()
    )

    # Rebuild signed margins on the raw source so prior-election candidates
    # omitted from extended.csv because of deposit forfeiture remain eligible.
    regular = source.loc[source["poll_type"].eq(0)].copy()
    share = regular["vote_share_percentage"]
    fallback_share = (
        regular["votes"]
        .div(regular["valid_votes"].where(regular["valid_votes"].gt(0)))
        .mul(100)
    )
    regular["_share"] = share.fillna(fallback_share)
    seat_keys = [
        "year", "assembly_number", "state_acronym",
        "constituency_number", "delimitation_id", "poll_type",
    ]
    winner_rows = regular.loc[regular["position"].eq(1)]
    winners = winner_rows.groupby(seat_keys, dropna=False).agg(
        _winner_share=("_share", "max"),
        _winner_margin=("margin_percentage", "max"),
    )
    runner_up = (
        regular.loc[~regular["position"].eq(1)]
        .groupby(seat_keys, dropna=False)["_share"]
        .max()
        .rename("_runner_up_share")
    )
    winners = winners.join(runner_up)
    regular = regular.join(winners, on=seat_keys, validate="many_to_one")
    regular["_signed_margin"] = regular["_share"] - regular["_winner_share"]
    is_winner = regular["position"].eq(1)
    winner_margin_fallback = regular["_winner_share"] - regular["_runner_up_share"]
    regular.loc[is_winner, "_signed_margin"] = regular.loc[is_winner, "_winner_margin"].fillna(
        winner_margin_fallback.loc[is_winner]
    )
    source["_signed_margin"] = np.nan
    source.loc[regular.index, "_signed_margin"] = regular["_signed_margin"]
    return source, canonical_year, int(len(source))


def make_cohort(
    source: pd.DataFrame,
    assembly: int,
    election_year: int,
    threshold: float,
) -> tuple[pd.DataFrame, int, int]:
    rows = source.loc[
        source["assembly_number"].eq(assembly)
        & source["year"].eq(election_year)
        & source["poll_type"].eq(0)
        & source["_signed_margin"].abs().le(threshold)
    ].copy()
    close_without_id = int(rows["candidate_id"].isna().sum())
    rows = rows.dropna(subset=["candidate_id", "position"])
    rows["prior_group"] = np.where(rows["position"].eq(1), "Previous narrow winners", "Previous narrow losers")

    # Some candidates ran in multiple constituencies in one election. Count
    # each candidate once per assembly, and omit cases where the close races
    # put the same candidate in both prior-outcome groups.
    summary = rows.groupby("candidate_id", as_index=False).agg(
        prior_group_count=("prior_group", "nunique"),
        prior_group=("prior_group", "first"),
    )
    ambiguous = int(summary["prior_group_count"].gt(1).sum())
    summary = summary.loc[summary["prior_group_count"].eq(1), ["candidate_id", "prior_group"]]
    return summary, close_without_id, ambiguous


def next_assembly_lookup(source: pd.DataFrame, assembly: int, year: int) -> pd.DataFrame:
    following = source.loc[
        source["assembly_number"].eq(assembly)
        & source["year"].eq(year)
        & source["poll_type"].eq(0)
    ].dropna(subset=["candidate_id"])
    if following.empty:
        return pd.DataFrame(columns=["candidate_id", "next_outcome"])
    status = following.groupby("candidate_id")["position"].agg(
        contested="size",
        won=lambda values: values.eq(1).any(),
    )
    status["next_outcome"] = np.where(
        status["won"], "Won", np.where(status["contested"].gt(0), "Lost", "Did not contest")
    )
    return status.reset_index()[["candidate_id", "next_outcome"]]


def analyze_transition(
    source: pd.DataFrame,
    assembly: int,
    election_year: int,
    next_assembly: int,
    next_year: int,
    threshold: float,
    z: float,
) -> dict:
    cohort, missing_id, ambiguous = make_cohort(
        source, assembly, election_year, threshold
    )
    future = next_assembly_lookup(source, next_assembly, next_year)
    cohort = cohort.merge(future, on="candidate_id", how="left", validate="one_to_one")
    cohort["next_outcome"] = cohort["next_outcome"].fillna("Did not contest")

    groups = {}
    for label in ("Previous narrow winners", "Previous narrow losers"):
        group = cohort.loc[cohort["prior_group"].eq(label)]
        counts = group["next_outcome"].value_counts().reindex(OUTCOMES, fill_value=0).astype(int)
        total = int(counts.sum())
        wins = int(counts["Won"])
        contested = int(counts["Won"] + counts["Lost"])
        all_rate = wins / total if total else math.nan
        conditional_rate = wins / contested if contested else math.nan
        groups[label] = {
            "counts": counts,
            "n": total,
            "wins": wins,
            "contested": contested,
            "win_rate": all_rate,
            "win_ci": wilson_interval(wins, total, z),
            "recontest_rate": contested / total if total else math.nan,
            "recontest_ci": wilson_interval(contested, total, z),
            "conditional_win_rate": conditional_rate,
            "conditional_win_ci": wilson_interval(wins, contested, z),
        }

    winners = groups["Previous narrow winners"]
    losers = groups["Previous narrow losers"]
    difference = winners["win_rate"] - losers["win_rate"]
    difference_ci = newcombe_difference_interval(
        winners["win_rate"], winners["win_ci"],
        losers["win_rate"], losers["win_ci"],
    )
    return {
        "threshold": threshold,
        "assembly": assembly,
        "election_year": election_year,
        "next_assembly": next_assembly,
        "next_year": next_year,
        "groups": groups,
        "difference": difference,
        "difference_ci": difference_ci,
        "ambiguous_candidates_excluded": ambiguous,
        "close_entries_missing_id": missing_id,
    }


def aggregate_results(transition_results: list[dict], z: float) -> list[dict]:
    """Pool counts across observed assembly transitions for each cutoff."""
    pooled = []
    for threshold in THRESHOLDS:
        subset = [r for r in transition_results if r["threshold"] == threshold]
        groups = {}
        for label in ("Previous narrow winners", "Previous narrow losers"):
            counts = sum(
                (r["groups"][label]["counts"] for r in subset),
                start=pd.Series(0, index=OUTCOMES, dtype="int64"),
            ).reindex(OUTCOMES, fill_value=0).astype(int)
            total = int(counts.sum())
            wins = int(counts["Won"])
            contested = int(counts["Won"] + counts["Lost"])
            groups[label] = {
                "counts": counts,
                "n": total,
                "wins": wins,
                "contested": contested,
                "win_rate": wins / total if total else math.nan,
                "win_ci": wilson_interval(wins, total, z),
                "recontest_rate": contested / total if total else math.nan,
                "recontest_ci": wilson_interval(contested, total, z),
                "conditional_win_rate": wins / contested if contested else math.nan,
                "conditional_win_ci": wilson_interval(wins, contested, z),
            }
        winners = groups["Previous narrow winners"]
        losers = groups["Previous narrow losers"]
        difference = winners["win_rate"] - losers["win_rate"]
        difference_ci = newcombe_difference_interval(
            winners["win_rate"], winners["win_ci"],
            losers["win_rate"], losers["win_ci"],
        )
        pooled.append(
            {
                "threshold": threshold,
                "groups": groups,
                "difference": difference,
                "difference_ci": difference_ci,
                "transition_count": len(subset),
                "transitions": subset,
                "ambiguous_candidates_excluded": sum(r["ambiguous_candidates_excluded"] for r in subset),
                "close_entries_missing_id": sum(r["close_entries_missing_id"] for r in subset),
            }
        )
    return pooled


def percentage(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def render_report(results: list[dict]) -> str:
    lines = [
        "CLOSE-ELECTION ANALYSIS",
        "Cutoff | Prior result | N | Won | Lost | Did not contest | Win rate (95% Wilson CI) | Recontested | Recontest rate | Win among recontesters",
    ]
    for result in results:
        threshold = result["threshold"]
        for label, short in [
            ("Previous narrow winners", "Narrow winner"),
            ("Previous narrow losers", "Narrow loser"),
        ]:
            group = result["groups"][label]
            counts = group["counts"]
            lines.append(
                f"+/-{threshold:.0f}% | {short} | {group['n']} | {counts['Won']} | "
                f"{counts['Lost']} | {counts['Did not contest']} | "
                f"{percentage(group['win_rate'])} ({percentage(group['win_ci'][0])}-"
                f"{percentage(group['win_ci'][1])}) | {group['contested']} | "
                f"{percentage(group['recontest_rate'])} | "
                f"{percentage(group['conditional_win_rate'])}"
            )
        lines.append(
            f"+/-{threshold:.0f}% | Difference (winner - loser, pp) | "
            f"{result['difference'] * 100:.2f} | Newcombe 95% CI "
            f"{result['difference_ci'][0] * 100:.2f} to "
            f"{result['difference_ci'][1] * 100:.2f}"
        )
    return "\n".join(lines)


def save_plots(results: list[dict]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    records = []
    for result in results:
        for group_label in ("Previous narrow winners", "Previous narrow losers"):
            group = result["groups"][group_label]
            labels.append(f"±{result['threshold']:.0f}%\n{group_label.replace('Previous ', '')}")
            records.append(group)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(records))
    colors = {"Won": "#3b8c6e", "Lost": "#d97941", "Did not contest": "#9da7b1"}
    for outcome in OUTCOMES:
        heights = np.array(
            [r["counts"][outcome] / r["n"] * 100 if r["n"] else 0 for r in records]
        )
        ax.bar(labels, heights, bottom=bottoms, label=outcome, color=colors[outcome], width=0.68)
        for index, (height, bottom) in enumerate(zip(heights, bottoms)):
            if height >= 7:
                ax.text(index, bottom + height / 2, f"{height:.1f}%", ha="center", va="center", color="white", fontsize=9)
        bottoms += heights
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of prior candidate group (%)")
    ax.set_title("Subsequent outcomes after a close prior election")
    ax.legend(title="Next assembly outcome", loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
    fig.savefig(OUTCOME_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(results))
    offsets = (-0.12, 0.12)
    colors_by_group = ("#2166ac", "#b2182b")
    for index, (label, color, offset) in enumerate(
        zip(("Previous narrow winners", "Previous narrow losers"), colors_by_group, offsets)
    ):
        rates = [result["groups"][label]["win_rate"] * 100 for result in results]
        lower = [
            (result["groups"][label]["win_rate"] - result["groups"][label]["win_ci"][0]) * 100
            for result in results
        ]
        upper = [
            (result["groups"][label]["win_ci"][1] - result["groups"][label]["win_rate"]) * 100
            for result in results
        ]
        ax.errorbar(
            x_positions + offset,
            rates,
            yerr=np.array([lower, upper]),
            fmt="o",
            capsize=5,
            markersize=8,
            color=color,
            label=label,
        )
        for x, rate in zip(x_positions + offset, rates):
            ax.annotate(f"{rate:.1f}%", (x, rate), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x_positions, [f"±{r['threshold']:.0f}%" for r in results])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Subsequent wins / all prior candidates (%)")
    ax.set_xlabel("Previous election margin cutoff")
    ax.set_title("Subsequent winning rates with 95% Wilson intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
    fig.savefig(WIN_RATE_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    extended, source = load_inputs()
    source, canonical_year, source_rows = prepare_data(extended, source)
    assemblies = sorted(canonical_year)
    if len(assemblies) < 2:
        raise ValueError("At least two Lok Sabha assemblies are required for follow-up analysis.")

    z = (
        float(norm.ppf(1 - (1 - CONFIDENCE) / 2))
        if norm is not None
        else 1.959963984540054
    )
    transition_results = []
    for assembly, next_assembly in zip(assemblies[:-1], assemblies[1:]):
        prior_year = canonical_year[assembly]
        next_year = canonical_year[next_assembly]
        for threshold in THRESHOLDS:
            transition_results.append(
                analyze_transition(
                    source,
                    assembly,
                    prior_year,
                    next_assembly,
                    next_year,
                    threshold,
                    z,
                )
            )

    results = aggregate_results(transition_results, z)
    save_plots(results)
    print(render_report(results))


if __name__ == "__main__":
    main()
