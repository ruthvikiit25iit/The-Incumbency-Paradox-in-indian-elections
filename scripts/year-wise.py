import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

source_file = sys.argv[1]

data_folder = Path("data_files")
visualisation_folder = Path("visualisations")

data_folder.mkdir(exist_ok=True)
visualisation_folder.mkdir(exist_ok=True)

data = pd.read_csv(source_file, low_memory=False)
data = data[data["poll_type"] == 0]

data["incumbent"] = data["incumbent"].astype(str).str.lower()
data["won"] = data["position"] == 1

rows = []

for year in sorted(data["year"].unique()):
    year_data = data[data["year"] == year]

    incumbents = year_data[year_data["incumbent"] == "true"]
    non_incumbents = year_data[year_data["incumbent"] == "false"]
    unknown = year_data[~year_data["incumbent"].isin(["true", "false"])]

    incumbent_rate = incumbents["won"].mean() * 100
    non_incumbent_rate = non_incumbents["won"].mean() * 100

    rows.append([
        year,
        len(incumbents),
        len(non_incumbents),
        len(unknown),
        len(year_data),
        incumbents["won"].sum(),
        non_incumbents["won"].sum(),
        unknown["won"].sum(),
        incumbent_rate,
        non_incumbent_rate,
        incumbent_rate - non_incumbent_rate
    ])

result = pd.DataFrame(rows, columns=[
    "year",
    "no_of_incumbents",
    "no_of_non_incumbents",
    "no_of_unknown_incumbent_status",
    "total_no_participated",
    "no_of_incumbents_won",
    "no_of_non_incumbents_won",
    "no_of_unknown_status_winners",
    "incumbency_win_percentage",
    "non_incumbency_win_percentage",
    "diff_percentage_points"
])

result.round(2).to_csv(
    data_folder / "yearly_incumbency_win_rates.csv",
    index=False
)

plt.figure(figsize=(12, 7))

plt.plot(
    result["year"],
    result["incumbency_win_percentage"],
    marker="o",
    label="Incumbents"
)

plt.plot(
    result["year"],
    result["non_incumbency_win_percentage"],
    marker="o",
    label="Non-incumbents"
)

plt.title("Incumbent vs Non-incumbent Win Percentage")
plt.xlabel("Year")
plt.ylabel("Win Percentage")
plt.grid()
plt.legend()

plt.savefig(
    visualisation_folder / "yearly_incumbency_win_rate_comparison.png"
)

plt.close()

plt.figure(figsize=(12, 7))

plt.axhline(0, color="black")

plt.plot(
    result["year"],
    result["diff_percentage_points"],
    marker="o"
)

plt.title("Difference in Incumbent and Non-incumbent Win Percentage")
plt.xlabel("Year")
plt.ylabel("Percentage Points")
plt.grid()

plt.savefig(
    visualisation_folder / "yearly_incumbency_win_rate_difference.png"
)

plt.close()
