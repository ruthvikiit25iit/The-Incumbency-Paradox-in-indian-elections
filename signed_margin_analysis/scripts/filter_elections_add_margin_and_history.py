"""Filter elections, calculate signed margins, and add candidate/party history.

Run from the project folder:
    python signed_margin_analysis/scripts/filter_elections_add_margin_and_history.py input.csv output.csv

The output keeps poll_type 0 rows and excludes deposit_lost == true.
Candidate participation and previous constituency-party vote share are searched
across all poll types, within a six-year gap. Party vote share is a percentage
read from the prior election's vote_share_percentage field.
"""

from pathlib import Path
import sys

import pandas as pd


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python signed_margin_analysis/filter_elections_add_margin_and_history.py "
        "input.csv output.csv"
    )

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
df = pd.read_csv(input_file, low_memory=False)

required_columns = [
    "candidate_id",
    "year",
    "poll_type",
    "position",
    "deposit_lost",
    "party",
    "party_id",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "constituency_name",
    "delimitation_id",
    "vote_share_percentage",
    "margin_percentage",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def to_bool(series):
    values = series.astype("string").str.strip().str.lower()
    return values.map({
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    })


df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["poll_type"] = pd.to_numeric(df["poll_type"], errors="coerce")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df["vote_share_percentage"] = pd.to_numeric(
    df["vote_share_percentage"], errors="coerce"
)
df["margin_percentage"] = pd.to_numeric(df["margin_percentage"], errors="coerce")
df["_vote_share_for_calculation"] = df["vote_share_percentage"]
if "month" in df.columns:
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

# Fill missing vote shares from votes / valid_votes when possible.
if {"votes", "valid_votes"}.issubset(df.columns):
    votes = pd.to_numeric(df["votes"], errors="coerce")
    valid_votes = pd.to_numeric(df["valid_votes"], errors="coerce")
    calculated_share = (votes / valid_votes.where(valid_votes > 0)) * 100
    df["_vote_share_for_calculation"] = df["_vote_share_for_calculation"].fillna(
        calculated_share
    )

# Match each candidate to the winner in that same constituency and election.
seat_columns = [
    "year",
    "state_acronym",
    "assembly_number",
    "constituency_number",
    "delimitation_id",
]
for column in ["month", "poll_type"]:
    if column in df.columns:
        seat_columns.append(column)

winners = df.loc[
    df["position"] == 1,
    seat_columns + ["_vote_share_for_calculation"],
].copy()
winner_counts = winners.groupby(seat_columns, dropna=False).size()
if winner_counts.empty or not winner_counts.eq(1).all():
    raise ValueError("Each constituency-election must have exactly one position 1 candidate.")

winners = winners.rename(
    columns={"_vote_share_for_calculation": "_winner_vote_share_percentage"}
)
df = df.merge(winners, on=seat_columns, how="left", validate="many_to_one")

df["signed_victory_margin_percentage"] = (
    df["_vote_share_for_calculation"] - df["_winner_vote_share_percentage"]
)
winner_rows = df["position"] == 1
df.loc[winner_rows, "signed_victory_margin_percentage"] = df.loc[
    winner_rows, "margin_percentage"
]

# Give every election in each seat a key so the prior seat election can be found.
constituency_columns = [
    "state_acronym",
    "constituency_number",
    "constituency_name",
]
event_columns = constituency_columns + [
    "year",
    "assembly_number",
    "poll_type",
]
if "month" in df.columns:
    event_columns.append("month")
df["_event_id"] = df.groupby(
    event_columns, dropna=False, sort=False
).ngroup()

# Keep regular-election records and remove forfeited-deposit candidates.
deposit_lost = to_bool(df["deposit_lost"])
output = df.loc[(df["poll_type"] == 0) & (deposit_lost != True)].copy()

# Prepare previous-contest fields for every candidate, not just incumbents.
history_fields = [
    "previous_contest_year",
    "previous_contest_poll_type",
    "previous_contest_party",
    "previous_contest_state_acronym",
    "previous_contest_constituency_name",
    "previous_contest_signed_victory_margin_percentage",
    "previous_contest_match_status",
]
for column in history_fields:
    output[column] = pd.NA

candidate_history = df.loc[df["candidate_id"].notna()].copy()
history_by_candidate = {
    candidate_id: group
    for candidate_id, group in candidate_history.groupby("candidate_id", sort=False)
}

for index, current in output.iterrows():
    candidate_id = current["candidate_id"]
    current_year = current["year"]

    if pd.isna(candidate_id):
        output.at[index, "previous_contest_match_status"] = "MISSING_CANDIDATE_ID"
        continue
    if pd.isna(current_year):
        output.at[index, "previous_contest_match_status"] = "MISSING_CURRENT_YEAR"
        continue

    candidate_records = history_by_candidate.get(candidate_id)
    if candidate_records is None:
        output.at[index, "previous_contest_match_status"] = (
            "NO_PREVIOUS_CONTEST_WITHIN_6_YEARS"
        )
        continue

    # Prior years must be earlier and no more than six years before this row.
    earlier = candidate_records[
        (candidate_records["year"] < current_year)
        & ((current_year - candidate_records["year"]) <= 6)
    ].copy()

    # If month data is available, also include earlier contests in the same year.
    if "month" in df.columns and pd.notna(current.get("month")):
        same_year_earlier = candidate_records[
            (candidate_records["year"] == current_year)
            & candidate_records["month"].notna()
            & (
                (candidate_records["month"] < current["month"])
                | (
                    (candidate_records["month"] == current["month"])
                    & (candidate_records["poll_type"] < current["poll_type"])
                )
            )
        ]
        earlier = pd.concat([earlier, same_year_earlier]).drop_duplicates()

    if earlier.empty:
        output.at[index, "previous_contest_match_status"] = (
            "NO_PREVIOUS_CONTEST_WITHIN_6_YEARS"
        )
        continue

    sort_columns = ["year"]
    if "month" in earlier.columns:
        sort_columns.append("month")
    sort_columns.append("poll_type")
    earlier = earlier.sort_values(sort_columns, na_position="first")
    latest = earlier.iloc[-1]

    same_latest_event = earlier[
        (earlier["year"] == latest["year"])
        & (earlier["poll_type"] == latest["poll_type"])
    ]
    if "month" in earlier.columns:
        same_month = same_latest_event["month"].eq(latest["month"])
        if pd.isna(latest["month"]):
            same_month = same_latest_event["month"].isna()
        same_latest_event = same_latest_event[same_month]

    if len(same_latest_event) > 1:
        same_constituency = same_latest_event[
            (same_latest_event["state_acronym"] == current["state_acronym"])
            & (same_latest_event["constituency_number"] == current["constituency_number"])
            & (same_latest_event["delimitation_id"] == current["delimitation_id"])
        ]
        if len(same_constituency) == 1:
            latest = same_constituency.iloc[0]
        else:
            output.at[index, "previous_contest_match_status"] = (
                "AMBIGUOUS_MULTIPLE_CONTESTS_SAME_EVENT"
            )
            continue

    output.at[index, "previous_contest_year"] = latest["year"]
    output.at[index, "previous_contest_poll_type"] = latest["poll_type"]
    output.at[index, "previous_contest_party"] = latest["party"]
    output.at[index, "previous_contest_state_acronym"] = latest["state_acronym"]
    output.at[index, "previous_contest_constituency_name"] = latest["constituency_name"]
    output.at[index, "previous_contest_signed_victory_margin_percentage"] = latest[
        "signed_victory_margin_percentage"
    ]
    output.at[index, "previous_contest_match_status"] = (
        "MATCHED" if pd.notna(latest["signed_victory_margin_percentage"])
        else "MATCHED_MARGIN_MISSING"
    )

# Find the previous election held in the same constituency, regardless of who ran.
event_details = [
    "_event_id",
    "year",
    "poll_type",
    "assembly_number",
    *constituency_columns,
]
if "month" in df.columns:
    event_details.append("month")
events = df[event_details].drop_duplicates().copy()

event_sort_columns = constituency_columns + ["year"]
if "month" in events.columns:
    event_sort_columns.append("month")
event_sort_columns.extend(["poll_type", "assembly_number"])
events = events.sort_values(event_sort_columns, na_position="first")
for column in [
    "_previous_event_id",
    "_previous_event_year",
    "_previous_event_poll_type",
]:
    events[column] = pd.NA

# Same-year by-elections are not the previous election for a regular election.
# For each event, choose the latest event from a strictly earlier year, within
# the six-year limit. If that year has multiple events, prefer poll_type 0.
for _, seat_events in events.groupby(constituency_columns, dropna=False, sort=False):
    for current_index, current_event in seat_events.iterrows():
        if pd.isna(current_event["year"]):
            continue
        earlier_events = seat_events[
            (seat_events["year"] < current_event["year"])
            & ((current_event["year"] - seat_events["year"]) <= 6)
        ]
        if earlier_events.empty:
            continue

        previous_year = earlier_events["year"].max()
        latest_year_events = earlier_events[earlier_events["year"] == previous_year]
        regular_events = latest_year_events[latest_year_events["poll_type"] == 0]
        if not regular_events.empty:
            latest_year_events = regular_events
        previous_event = latest_year_events.sort_values(
            [column for column in ["month", "poll_type", "assembly_number"]
             if column in latest_year_events.columns],
            na_position="first",
        ).iloc[-1]

        events.at[current_index, "_previous_event_id"] = previous_event["_event_id"]
        events.at[current_index, "_previous_event_year"] = previous_event["year"]
        events.at[current_index, "_previous_event_poll_type"] = previous_event["poll_type"]

event_links = events[[
    "_event_id",
    "_previous_event_id",
    "_previous_event_year",
    "_previous_event_poll_type",
]]
output = output.merge(event_links, on="_event_id", how="left", validate="many_to_one")
output["previous_party_election_year"] = output["_previous_event_year"]
output["previous_party_election_poll_type"] = output["_previous_event_poll_type"]

# Use vote_share_percentage directly from the previous election's party rows.
party_vote_shares = (
    df.loc[df["party_id"].notna()]
    .groupby(["_event_id", "party_id"], dropna=False)
    .agg(
        _previous_party_vote_share=(
            "vote_share_percentage",
            lambda values: values.sum(min_count=1),
        ),
        _previous_party_record_count=("vote_share_percentage", "size"),
    )
    .reset_index()
    .rename(columns={"_event_id": "_party_vote_event_id"})
)
output = output.merge(
    party_vote_shares,
    left_on=["_previous_event_id", "party_id"],
    right_on=["_party_vote_event_id", "party_id"],
    how="left",
    validate="many_to_one",
)
output["previous_party_vote_share_percentage"] = output[
    "_previous_party_vote_share"
]
output["previous_party_match_status"] = "MATCHED"
output.loc[
    output["_previous_event_id"].isna(), "previous_party_match_status"
] = "NO_PREVIOUS_CONSTITUENCY_ELECTION_WITHIN_6_YEARS"
output.loc[
    output["_previous_event_id"].notna() & output["party_id"].isna(),
    "previous_party_match_status",
] = "MISSING_CURRENT_PARTY_ID"
output.loc[
    output["_previous_event_id"].notna()
    & output["party_id"].notna()
    & output["_previous_party_record_count"].isna(),
    "previous_party_match_status",
] = "PARTY_DID_NOT_CONTEST_PREVIOUS_ELECTION"
output.loc[
    output["_previous_party_record_count"].notna()
    & output["_previous_party_vote_share"].isna(),
    "previous_party_match_status",
] = "PARTY_VOTE_SHARE_MISSING"
output.loc[
    output["previous_party_match_status"] != "MATCHED",
    "previous_party_vote_share_percentage",
] = pd.NA

missing_loser_margins = int(
    (
        (output["position"] != 1)
        & output["_winner_vote_share_percentage"].isna()
    ).sum()
)
internal_columns = [
    "_vote_share_for_calculation",
    "_winner_vote_share_percentage",
    "_event_id",
    "_previous_event_id",
    "_previous_event_year",
    "_previous_event_poll_type",
    "_party_vote_event_id",
    "_previous_party_vote_share",
    "_previous_party_record_count",
]
output = output.drop(columns=internal_columns, errors="ignore")

output_file.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(output_file, index=False)
print(f"Rows written: {len(output):,}")
print(f"Filtered file with margins, previous-contest history, and party votes created: {output_file}")
if missing_loser_margins:
    print(
        f"Warning: {missing_loser_margins} losing candidate row(s) had no usable "
        "winner vote share, so their signed margins are blank."
    )
