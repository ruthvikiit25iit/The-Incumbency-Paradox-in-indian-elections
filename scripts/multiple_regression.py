"""Check how incumbency and previous performance relate to the next margin.

Run from the repository folder:
    .venv/bin/python scripts/multiple_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import inv, lstsq
from scipy.stats import t


INPUT_FILE = Path("data_files/extended.csv")
PLOT_FILE = Path("visualisations/multiple_regression_observed_vs_predicted.png")

Y_COLUMN = "signed_victory_margin_percentage"
M_COLUMN = "previous_contest_signed_victory_margin_percentage"
P_COLUMN = "previous_party_vote_share_percentage"


def read_analysis_rows():
    """Keep candidates with all the values needed for this comparison."""
    candidates = pd.read_csv(INPUT_FILE, low_memory=False)
    required_columns = {
        "poll_type",
        "incumbent",
        Y_COLUMN,
        M_COLUMN,
        P_COLUMN,
    }
    missing = required_columns - set(candidates.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    # Keep regular elections as outcome records. Previous contest and party election poll types are unrestricted.
    numeric_columns = [
        "poll_type",
        Y_COLUMN,
        M_COLUMN,
        P_COLUMN,
    ]
    for column in numeric_columns:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    candidates["I"] = (
        candidates["incumbent"].astype("string").str.strip().str.lower().map(
            {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
        )
    )

    rows = candidates.loc[
        (candidates["poll_type"] == 0),
        ["I", Y_COLUMN, M_COLUMN, P_COLUMN],
    ].dropna()
    return rows


def fit_model(rows):
    """Fit Y = beta_0 + beta_1*I + beta_2*M + beta_3*P."""
    y_values = rows[Y_COLUMN].to_numpy(dtype=float)
    predictors = rows[["I", M_COLUMN, P_COLUMN]].to_numpy(dtype=float)

    # The intercept lets the model have a starting value when predictors are zero.
    design_frame = rows[["I", M_COLUMN, P_COLUMN]].copy()
    design_frame.insert(0, "intercept", 1.0)
    design = design_frame.to_numpy(dtype=float)

    coefficients, _, rank, _ = lstsq(design, y_values)
    if rank < design.shape[1]:
        raise ValueError(
            "The regression predictors are not independent enough to estimate all coefficients."
        )

    predicted = design @ coefficients
    residuals = y_values - predicted
    degrees_of_freedom = len(y_values) - design.shape[1]
    if degrees_of_freedom <= 0:
        raise ValueError("There are too few complete rows to fit this model.")

    # Use the residuals to estimate uncertainty around each coefficient.
    residual_variance = (residuals @ residuals) / degrees_of_freedom
    coefficient_variance = residual_variance * inv(design.T @ design).diagonal()
    standard_errors = coefficient_variance.clip(0) ** 0.5
    t_values = coefficients / standard_errors
    p_values = 2 * t.sf(abs(t_values), degrees_of_freedom)
    critical_value = t.ppf(0.975, degrees_of_freedom)
    lower_bounds = coefficients - critical_value * standard_errors
    upper_bounds = coefficients + critical_value * standard_errors

    total_variation = ((y_values - y_values.mean()) ** 2).sum()
    r_squared = 1 - (residuals @ residuals) / total_variation

    return {
        "coefficients": coefficients,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "p_values": p_values,
        "r_squared": r_squared,
        "actual": y_values,
        "predicted": predicted,
        "incumbent": rows["I"].to_numpy(dtype=int),
    }


def save_plot(model):
    """Plot observed margins against margins predicted by the model."""
    actual = model["actual"]
    predicted = model["predicted"]
    incumbent = model["incumbent"]
    beta_0, beta_1, beta_2, beta_3 = model["coefficients"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for value, label, color in [
        (0, "Non-incumbents", "#b2182b"),
        (1, "Incumbents", "#2166ac"),
    ]:
        group = incumbent == value
        ax.scatter(
            predicted[group],
            actual[group],
            label=label,
            color=color,
            alpha=0.35,
            s=18,
            edgecolors="none",
        )

    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(
        limits,
        limits,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )
    ax.set_xlabel("Model-predicted next-election margin (percentage points)")
    ax.set_ylabel("Actual next-election margin (percentage points)")
    ax.set_title("How closely do predictions match election results?")
    equation = (
        "Fitted model:\n"
        f"Y = {beta_0:.2f} {beta_1:+.2f}I "
        f"{beta_2:+.2f}M {beta_3:+.2f}P\n"
        "I = incumbent; M = previous candidate margin;\n"
        "P = previous party vote share"
    )
    ax.text(
        0.03,
        0.97,
        equation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    ax.legend()
    ax.grid(alpha=0.2)

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = read_analysis_rows()
    if len(rows) < 5:
        raise ValueError("Too few complete rows to estimate the regression model.")

    model = fit_model(rows)
    beta_0, beta_1, beta_2, beta_3 = model["coefficients"]

    separator = "=" * 110
    print(separator)
    print("MULTIPLE LINEAR REGRESSION")
    print(f"Current election poll_type: 0")
    print("Previous contest poll_type: all values")
    print("Previous party poll_type: all values")
    print(f"Complete model records: {len(rows):,}")
    print("Model: Y = beta_0 + beta_1*I + beta_2*M + beta_3*P")
    print(
        "Fitted equation: Y = "
        f"{beta_0:.4f} {beta_1:+.4f}I "
        f"{beta_2:+.4f}M {beta_3:+.4f}P"
    )
    print("Coefficients (estimate, 95% CI):")
    for name, estimate, lower, upper in zip(
        ["beta_0", "beta_1", "beta_2", "beta_3"],
        model["coefficients"],
        model["lower_bounds"],
        model["upper_bounds"],
    ):
        print(f"  {name}: {estimate:.4f} ({lower:.4f}, {upper:.4f})")
    print(f"Incumbency coefficient p-value: {model['p_values'][1]:.6f}")
    print(f"R-squared: {model['r_squared']:.4f}")
    print(separator)

    save_plot(model)


if __name__ == "__main__":
    main()
