
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "filtered_data/corrected/general_election_incumbents_with_last_win_margin.csv"
)

OUTPUT_DIR = Path("filtered_data/corrected")
MARGIN_COLUMN = "last_won_margin_percentage"


df = pd.read_csv(INPUT_FILE)

if MARGIN_COLUMN not in df.columns:
    raise KeyError(f"Column '{MARGIN_COLUMN}' not found in the input CSV.")

df[MARGIN_COLUMN] = pd.to_numeric(df[MARGIN_COLUMN], errors="coerce")

valid_df = df[
    df[MARGIN_COLUMN].notna() &
    (df[MARGIN_COLUMN] >= 0)
].copy()

margin_less_than_2 = valid_df[
    valid_df[MARGIN_COLUMN] < 2
].copy()

margin_less_than_5 = valid_df[
    valid_df[MARGIN_COLUMN] < 5
].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

margin_less_than_2.to_csv(
    OUTPUT_DIR / "general_election_incumbents_last_win_margin_less_than_2.csv",
    index=False
)

margin_less_than_5.to_csv(
    OUTPUT_DIR / "general_election_incumbents_last_win_margin_less_than_5.csv",
    index=False
)

import matplotlib.pyplot as plt

PLOT_DIR = Path("data_visualisations")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

bins = list(range(0, 105, 5))
labels = [f"{i}-{i+5}" for i in range(0, 100, 5)]

datasets = [
    (margin_less_than_2, "Below 2%", "available-incumbent_current_winners_last_margin_below_2.png"),
    (margin_less_than_5, "Below 5%", "available-incumbent_current_winners_last_margin_below_5.png")
]

for data, title, filename in datasets:

    winners = data[data["position"] == 1].copy()

    winners["margin_percentage"] = pd.to_numeric(
        winners["margin_percentage"], errors="coerce"
    )

    winners = winners[
        winners["margin_percentage"].between(0, 100)
    ]

    counts = pd.cut(
        winners["margin_percentage"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    ).value_counts().sort_index()

    plt.figure(figsize=(12, 6))

    bars = plt.bar(labels, counts.values)

    plt.bar_label(bars, labels=[
        str(int(x)) for x in counts.values
    ], padding=3)

    plt.title(f"Current Winners with Last Won Margin {title}")
    plt.xlabel("Current winning margin percentage bracket")
    plt.ylabel("Number of winning candidates")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()
