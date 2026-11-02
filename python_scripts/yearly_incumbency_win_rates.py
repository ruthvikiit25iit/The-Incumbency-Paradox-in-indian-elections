
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "filtered_data" / "corrected"
    / "updated_general_elections.csv"
)

CSV_OUTPUT = (
    PROJECT_DIR / "filtered_data" / "corrected"
    / "yearly_incumbency_win_rates.csv"
)

PLOT_DIR = PROJECT_DIR / "data_visualisations"



if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = {"year", "incumbent", "position"}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise KeyError(f"Missing columns: {sorted(missing_columns)}")




incumbent_values = (
    df["incumbent"].astype("string").str.strip().str.lower()
)

df["_incumbent_status"] = incumbent_values.map({
    "true": True,
    "1": True,
    "yes": True,
    "false": False,
    "0": False,
    "no": False
})

df["_year"] = pd.to_numeric(df["year"], errors="coerce")
df["_position"] = pd.to_numeric(df["position"], errors="coerce")

missing_years = int(df["_year"].isna().sum())

df = df[df["_year"].notna()].copy()



results = []

for year, year_data in df.groupby("_year", sort=True):

    incumbents = year_data[
        year_data["_incumbent_status"] == True
    ]

    non_incumbents = year_data[
        year_data["_incumbent_status"] == False
    ]

    unknown = year_data[
        year_data["_incumbent_status"].isna()
    ]

    incumbent_wins = int((incumbents["_position"] == 1).sum())

    non_incumbent_wins = int(
        (non_incumbents["_position"] == 1).sum()
    )

    unknown_wins = int((unknown["_position"] == 1).sum())

    incumbent_rate = (
        incumbent_wins / len(incumbents) * 100
        if len(incumbents) > 0 else float("nan")
    )

    non_incumbent_rate = (
        non_incumbent_wins / len(non_incumbents) * 100
        if len(non_incumbents) > 0 else float("nan")
    )

    difference = (
        incumbent_rate - non_incumbent_rate
        if pd.notna(incumbent_rate)
        and pd.notna(non_incumbent_rate)
        else float("nan")
    )

    results.append({
        "year": int(year) if float(year).is_integer() else year,

        "no_of_incumbents": len(incumbents),
        "no_of_non_incumbents": len(non_incumbents),
        "no_of_unknown_incumbent_status": len(unknown),

        "total_no_participated": len(year_data),

        "no_of_incumbents_won": incumbent_wins,
        "no_of_non_incumbents_won": non_incumbent_wins,
        "no_of_unknown_status_winners": unknown_wins,

        "incumbency_win_percentage": round(incumbent_rate, 2),
        "non_incumbency_win_percentage": round(non_incumbent_rate, 2),

        "diff_percentage_points": round(difference, 2)
    })


result = pd.DataFrame(results)

if result.empty:
    raise ValueError("No valid election-year data found.")



CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(CSV_OUTPUT, index=False)




PLOT_DIR.mkdir(parents=True, exist_ok=True)

years = result["year"]

incumbent_rates = result["incumbency_win_percentage"]
non_incumbent_rates = result["non_incumbency_win_percentage"]

differences = result["diff_percentage_points"]

incumbent_counts = result["no_of_incumbents"]
non_incumbent_counts = result["no_of_non_incumbents"]



fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(
    years, incumbent_rates,
    marker="o", label="Incumbency win percentage"
)

ax.plot(
    years, non_incumbent_rates,
    marker="o", label="Non-incumbency win percentage"
)

ax.set_title("Incumbency vs Non-incumbency Win Percentage")
ax.set_xlabel("Year")
ax.set_ylabel("Win Percentage")

ax.grid(True, alpha=0.6)
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig(PLOT_DIR / "yearly_incumbency_win_rate_comparison.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(
    years, differences,
    marker="o", label="Diff percentage"
)

ax.set_title("Difference in Incumbency and Non-incumbency Win Percentage")
ax.set_xlabel("Year")
ax.set_ylabel("Percentage")

ax.grid(True, alpha=0.6)
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(PLOT_DIR / "yearly_incumbency_win_rate_difference.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    years,
    incumbent_counts,
    marker="o",
    linewidth=2,
    label="Incumbents"
)

ax.plot(
    years,
    non_incumbent_counts,
    marker="o",
    linewidth=2,
    label="Non-incumbents"
)

ax.set_title(
    "Year-wise Candidate Participation"
)

ax.set_xlabel("General Election Year")
ax.set_ylabel("Number of Candidates")

ax.grid(axis="y", alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "yearly_incumbency_candidate_counts.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)
