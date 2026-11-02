"""Add a signed margin: winners' margin_percentage; losers' gap from the winner.

Run from the project folder:
    python add_signed_victory_margin.py input.csv output.csv
"""

import sys

import pandas as pd


input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, low_memory=False)

required_columns = [
    "year",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "delimitation_id",
    "position",
    "vote_share_percentage",
    "margin_percentage",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Include these fields when present so separate polls are kept separate.
group_columns = [
    "year",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "delimitation_id",
]
for column in ["month", "poll_type"]:
    if column in df.columns:
        group_columns.append(column)

df["position"] = pd.to_numeric(df["position"], errors="coerce")
df["vote_share_percentage"] = pd.to_numeric(
    df["vote_share_percentage"], errors="coerce"
)
df["margin_percentage"] = pd.to_numeric(df["margin_percentage"], errors="coerce")

# If a vote share is missing, calculate it from votes when valid votes are available.
if {"votes", "valid_votes"}.issubset(df.columns):
    votes = pd.to_numeric(df["votes"], errors="coerce")
    valid_votes = pd.to_numeric(df["valid_votes"], errors="coerce")
    calculated_share = (votes / valid_votes.where(valid_votes > 0)) * 100
    df["vote_share_percentage"] = df["vote_share_percentage"].fillna(calculated_share)

# Get the position 1 candidate's vote share for every constituency-election.
winners = df.loc[df["position"] == 1, group_columns + ["vote_share_percentage"]]
winner_counts = winners.groupby(group_columns, dropna=False).size()
if not winner_counts.eq(1).all():
    raise ValueError("Each constituency-election must have exactly one position 1 candidate.")
winners = winners.rename(
    columns={"vote_share_percentage": "_winner_vote_share_percentage"}
)
df = df.merge(winners, on=group_columns, how="left", validate="many_to_one")

df["signed_victory_margin_percentage"] = (
    df["vote_share_percentage"] - df["_winner_vote_share_percentage"]
)
# For winners, use their existing margin over the runner-up.
winner_rows = df["position"] == 1
df.loc[winner_rows, "signed_victory_margin_percentage"] = df.loc[
    winner_rows, "margin_percentage"
]
missing_loser_margins = df.loc[
    (df["position"] != 1) & df["_winner_vote_share_percentage"].isna()
].shape[0]
df = df.drop(columns=["_winner_vote_share_percentage"])

df.to_csv(output_file, index=False)
print(f"CSV with signed victory margins created: {output_file}")
if missing_loser_margins:
    print(
        f"Warning: {missing_loser_margins} losing candidate row(s) had no usable "
        "winner vote share, so their signed margins are blank."
    )
