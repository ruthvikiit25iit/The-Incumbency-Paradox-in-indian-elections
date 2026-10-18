"""Plot winning candidates by margin-percentage bracket.

Run from any directory with:
    python plot_margin_variance.py path/to/incumbents.csv path/to/non_incumbents.csv

The script reads filtered_data/incumbents.csv and
filtered_data/non_incumbents.csv, keeps winning candidates (position == 1),
and saves a separate count histogram for each group.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data_visualisations"
OUTPUTS = {
    "Incumbent winners": OUTPUT_DIR / "incumbent_winners_by_margin_bracket.png",
    "Non-incumbent winners": OUTPUT_DIR / "non_incumbent_winners_by_margin_bracket.png",
}
BIN_EDGES = list(range(0, 101, 5))


def winning_margins(path: Path, label: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path, low_memory=False)
    required = {"position", "margin_percentage"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")

    # Convert before filtering so both numeric and text-coded CSV values work.
    position = pd.to_numeric(frame["position"], errors="coerce")
    margin = pd.to_numeric(frame["margin_percentage"], errors="coerce")
    values = margin.loc[position.eq(1)].dropna()
    values.name = label
    if values.empty:
        raise ValueError(f"No winning candidates with valid margin percentages in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot winning candidates by margin-percentage bracket."
    )
    parser.add_argument("incumbents_csv", type=Path, help="CSV containing incumbent candidates")
    parser.add_argument("non_incumbents_csv", type=Path, help="CSV containing non-incumbent candidates")
    args = parser.parse_args()

    groups = {
        "Incumbent winners": winning_margins(args.incumbents_csv, "Incumbent winners"),
        "Non-incumbent winners": winning_margins(args.non_incumbents_csv, "Non-incumbent winners"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, margins in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        counts, edges, bars = ax.hist(
            margins,
            bins=BIN_EDGES,
            color="#377eb8" if label == "Incumbent winners" else "#e68632",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(f"{label}: winners by margin-percentage bracket")
        ax.set_xlabel("Margin percentage bracket")
        ax.set_ylabel("Number of winning candidates")
        ax.set_xticks(BIN_EDGES)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, labels=[f"{int(count)}" if count else "" for count in counts], padding=2, fontsize=8)
        fig.tight_layout()
        output_path = OUTPUTS[label]
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"{label}: {margins.size:,} winners; plot saved to {output_path}")


if __name__ == "__main__":
    main()
