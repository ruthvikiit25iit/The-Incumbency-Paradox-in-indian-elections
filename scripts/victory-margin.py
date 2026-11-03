import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
default_source = project_root / "data_files" / "extended.csv"
parser = argparse.ArgumentParser(description="Summarize victory margins for winning candidates.")
parser.add_argument(
    "source_file",
    nargs="?",
    type=Path,
    default=default_source,
    help=f"Candidate-level election CSV (default: {default_source})",
)
args = parser.parse_args()
source_file = args.source_file.resolve()

visualisation_folder = project_root / "visualisations"
visualisation_folder.mkdir(exist_ok=True)

data = pd.read_csv(source_file, low_memory=False)
data["poll_type"] = pd.to_numeric(data["poll_type"], errors="coerce")
data["position"] = pd.to_numeric(data["position"], errors="coerce")
data["margin_percentage"] = pd.to_numeric(
    data["margin_percentage"],
    errors="coerce"
)
status = data["incumbent"].astype("string").str.strip().str.lower()
data["is_incumbent"] = status.map(
    {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
)

base = data.loc[
    data["poll_type"].eq(0)
    & data["position"].notna()
    & data["is_incumbent"].notna()
].copy()
complete = base.dropna(subset=["margin_percentage"])
missing_margin_by_group = {
    label: int(
        (
            base["position"].eq(1)
            & base["is_incumbent"].eq(status_value)
            & base["margin_percentage"].isna()
        ).sum()
    )
    for label, status_value in (("incumbent", True), ("non_incumbent", False))
}
complete_winners = complete.loc[complete["position"].eq(1)].copy()

incumbents = complete_winners[complete_winners["is_incumbent"].eq(True)]
non_incumbents = complete_winners[complete_winners["is_incumbent"].eq(False)]

incumbent_mean = incumbents["margin_percentage"].mean()
non_incumbent_mean = non_incumbents["margin_percentage"].mean()

incumbent_median = incumbents["margin_percentage"].median()
non_incumbent_median = non_incumbents["margin_percentage"].median()

incumbent_std = incumbents["margin_percentage"].std()
non_incumbent_std = non_incumbents["margin_percentage"].std()

bins = list(range(0, 101, 5))
labels = [f"{start}-{start + 5}" for start in bins[:-1]]

incumbent_bins = pd.cut(
    incumbents["margin_percentage"],
    bins=bins,
    right=False,
    labels=labels
)

non_incumbent_bins = pd.cut(
    non_incumbents["margin_percentage"],
    bins=bins,
    right=False,
    labels=labels
)

incumbent_counts = incumbent_bins.value_counts().reindex(
    labels,
    fill_value=0
)

non_incumbent_counts = non_incumbent_bins.value_counts().reindex(
    labels,
    fill_value=0
)

positions = list(range(len(labels)))

plt.figure(figsize=(15, 7))

incumbent_bars = plt.bar(
    [x - 0.2 for x in positions],
    incumbent_counts,
    width=0.4,
    label="Incumbents"
)

non_incumbent_bars = plt.bar(
    [x + 0.2 for x in positions],
    non_incumbent_counts,
    width=0.4,
    label="Non-incumbents"
)

for bar in list(incumbent_bars) + list(non_incumbent_bars):
    if bar.get_height() > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=7
        )

largest_count = max(
    incumbent_counts.max(),
    non_incumbent_counts.max()
)

plt.ylim(0, largest_count * 1.1)
plt.xticks(positions, labels, rotation=45)
plt.title("Victory Margin Distribution of Winning Candidates")
plt.xlabel("Victory Margin Percentage Bin")
plt.ylabel("Number of Winners")
plt.grid(axis="y")
plt.legend()
plt.tight_layout()

plt.savefig(
    visualisation_folder / "victory_margin_pairwise_histogram.png"
)

plt.close()

plt.figure(figsize=(9, 7))

plt.boxplot(
    [
        incumbents["margin_percentage"],
        non_incumbents["margin_percentage"]
    ],
    showfliers=False
)

plt.scatter(
    [1] * len(incumbents),
    incumbents["margin_percentage"],
    color="blue",
    alpha=0.12,
    s=8
)

plt.scatter(
    [2] * len(non_incumbents),
    non_incumbents["margin_percentage"],
    color="orange",
    alpha=0.12,
    s=8
)

plt.xticks([1, 2], ["Incumbents", "Non-incumbents"])
plt.title("Victory Margin Distribution of Winning Candidates")
plt.ylabel("Victory Margin Percentage")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    visualisation_folder / "victory_margin_boxplot.png"
)

plt.close()

print()
print("VICTORY MARGIN ANALYSIS")
print("=" * 65)
print(f"Input file: {source_file}")
print("Common complete-case rule: poll_type=0, known incumbent status, numeric position, nonblank margin_percentage")
print(
    "Complete candidate sample: "
    f"incumbents={int(complete['is_incumbent'].sum()):,}; "
    f"non-incumbents={int(complete['is_incumbent'].eq(False).sum()):,}"
)
print(
    "Winning rows excluded for blank margin_percentage: "
    f"incumbents={missing_margin_by_group['incumbent']:,}; "
    f"non-incumbents={missing_margin_by_group['non_incumbent']:,}"
)
print(f"{'Statistic':<30}{'Incumbents':>15}{'Non-incumbents':>20}")
print("-" * 65)
print(f"{'Number of winners':<30}{len(incumbents):>15}{len(non_incumbents):>20}")
print(f"{'Mean victory margin (%)':<30}{incumbent_mean:>15.2f}{non_incumbent_mean:>20.2f}")
print(f"{'Median victory margin (%)':<30}{incumbent_median:>15.2f}{non_incumbent_median:>20.2f}")
print(f"{'Standard deviation (%)':<30}{incumbent_std:>15.2f}{non_incumbent_std:>20.2f}")
print("-" * 65)
print(f"Mean difference: {incumbent_mean - non_incumbent_mean:.2f} percentage points")
print("Positive difference means incumbents have the larger average margin.")
