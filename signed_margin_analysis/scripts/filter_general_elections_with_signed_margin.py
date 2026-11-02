"""Filter regular elections and add a signed victory margin.

Run from the project folder:
    python signed_margin_analysis/filter_general_elections_with_signed_margin.py input.csv output.csv
"""

from pathlib import Path
import sys

import pandas as pd


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python signed_margin_analysis/filter_general_elections_with_signed_margin.py "
        "input.csv output.csv"
    )

input_file = Path(sys.argv[1])

output_file = Path(sys.argv[2])
df = pd.read_csv(input_file, low_memory=False)

required_columns = [
    "year",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "delimitation_id",
    "poll_type",
    "deposit_lost",
    "position",
    "vote_share_percentage",
    "margin_percentage",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

def to_bool(series):
    """Convert common CSV boolean values to True/False; leave others missing."""
    values = series.astype("string").str.strip().str.lower()
    return values.map({
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    })


df["poll_type"] = pd.to_numeric(df["poll_type"], errors="coerce")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df["vote_share_percentage"] = pd.to_numeric(
    df["vote_share_percentage"], errors="coerce"
)
df["margin_percentage"] = pd.to_numeric(df["margin_percentage"], errors="coerce")

# Keep regular elections. Find their winners before removing deposit-lost rows,
# so every retained candidate can still be compared with the winner.
general = df.loc[df["poll_type"] == 0].copy()
if {"votes", "valid_votes"}.issubset(general.columns):
    votes = pd.to_numeric(general["votes"], errors="coerce")
    valid_votes = pd.to_numeric(general["valid_votes"], errors="coerce")
    calculated_share = (votes / valid_votes.where(valid_votes > 0)) * 100
    general["vote_share_percentage"] = general["vote_share_percentage"].fillna(
        calculated_share
    )

group_columns = [
    "year",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "delimitation_id",
]
for column in ["month", "poll_type"]:
    if column in general.columns:
        group_columns.append(column)

winners = general.loc[
    general["position"] == 1,
    group_columns + ["vote_share_percentage"],
].copy()
winner_counts = winners.groupby(group_columns, dropna=False).size()
if winner_counts.empty or not winner_counts.eq(1).all():
    raise ValueError("Each regular-election constituency must have exactly one position 1 candidate.")

winners = winners.rename(
    columns={"vote_share_percentage": "_winner_vote_share_percentage"}
)
general = general.merge(
    winners,
    on=group_columns,
    how="left",
    validate="many_to_one",
)

# Exclude candidates whose election deposit was forfeited.
deposit_lost = to_bool(general["deposit_lost"])
result = general.loc[deposit_lost != True].copy()

# Losers receive a negative gap from the winner. Winners use the dataset's
# existing margin over the runner-up, as requested.
result["signed_victory_margin_percentage"] = (
    result["vote_share_percentage"] - result["_winner_vote_share_percentage"]
)
winner_rows = result["position"] == 1
result.loc[winner_rows, "signed_victory_margin_percentage"] = result.loc[
    winner_rows, "margin_percentage"
]

missing_loser_margins = int(
    (
        (result["position"] != 1)
        & result["_winner_vote_share_percentage"].isna()
    ).sum()
)
result = result.drop(columns=["_winner_vote_share_percentage"])

output_file.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_file, index=False)

print(f"Rows written: {len(result):,}")
print(f"Filtered CSV created: {output_file}")
if missing_loser_margins:
    print(
        f"Warning: {missing_loser_margins} losing candidate row(s) had no usable "
        "winner vote share, so their signed margins are blank."
    )
