"""Relate candidates' previous signed margin to their next signed margin.

Run from the repository folder:
    .venv/bin/python scripts/previous_margin_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


INPUT_FILE = Path("data_files/extended.csv")
PLOT_FILE = Path(
    "visualisations/previous_vs_subsequent_signed_margin_regression.png"
)

PREVIOUS_MARGIN = "previous_contest_signed_victory_margin_percentage"
CURRENT_MARGIN = "signed_victory_margin_percentage"


def main():
    candidates = pd.read_csv(INPUT_FILE, low_memory=False)
    required_columns = {
        "poll_type",
        "previous_contest_poll_type",
        PREVIOUS_MARGIN,
        CURRENT_MARGIN,
    }
    missing_columns = required_columns - set(candidates.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    # Use general elections for both the current result and previous margin.
    candidates["poll_type"] = pd.to_numeric(candidates["poll_type"], errors="coerce")
    candidates["previous_contest_poll_type"] = pd.to_numeric(
        candidates["previous_contest_poll_type"], errors="coerce"
    )
    candidates[PREVIOUS_MARGIN] = pd.to_numeric(
        candidates[PREVIOUS_MARGIN], errors="coerce"
    )
    candidates[CURRENT_MARGIN] = pd.to_numeric(
        candidates[CURRENT_MARGIN], errors="coerce"
    )

    analysis_rows = candidates.loc[
        (candidates["poll_type"] == 0)
        & (candidates["previous_contest_poll_type"] == 0),
        [PREVIOUS_MARGIN, CURRENT_MARGIN],
    ].dropna()

    if len(analysis_rows) < 2:
        raise ValueError("At least two matched candidates are needed for regression.")

    previous_margin = analysis_rows[PREVIOUS_MARGIN]
    subsequent_margin = analysis_rows[CURRENT_MARGIN]
    result = linregress(previous_margin, subsequent_margin)
    r_squared = result.rvalue**2

    print("Simple linear regression")
    print("X = previous general-election signed victory margin (%)")
    print("Y = subsequent general-election signed victory margin (%)")
    print(f"Candidates included: {len(analysis_rows):,}")
    print("Model: Y = beta_0 + beta_1 * X")
    print(
        "Fitted equation: Y = "
        f"{result.intercept:.4f} + ({result.slope:.4f} * X)"
    )
    print(f"beta_0 (intercept): {result.intercept:.4f}")
    print(f"beta_1 (slope): {result.slope:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"p-value for slope: {result.pvalue:.4g}")

    x_line = pd.Series([previous_margin.min(), previous_margin.max()])
    y_line = result.intercept + result.slope * x_line

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        previous_margin,
        subsequent_margin,
        alpha=0.3,
        s=18,
        color="#2166ac",
        edgecolors="none",
        label="Candidate elections",
    )
    ax.plot(x_line, y_line, color="#b2182b", linewidth=2, label="Regression line")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Previous signed victory margin (%)")
    ax.set_ylabel("Subsequent signed victory margin (%)")
    ax.set_title("Previous and subsequent electoral performance")
    ax.legend()
    ax.grid(alpha=0.2)

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Regression graph saved to: {PLOT_FILE}")


if __name__ == "__main__":
    main()
