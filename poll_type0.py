import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, low_memory=False)

df = df[df["poll_type"] == 0].copy()

winners = df[df["position"] == 1].copy()

df = df[df["deposit_lost"] != True].copy()

df["last_won_margin_percentage"] = pd.NA

for index, row in df[df["incumbent"] == True].iterrows():
    previous_win = winners[
        (winners["candidate_id"] == row["candidate_id"]) &
        (winners["year"] == row["year"] - 5)
    ]

    if not previous_win.empty:
        df.loc[index, "last_won_margin_percentage"] = \
            previous_win.iloc[0]["margin_percentage"]

remove_columns = [
    "month",
    "sub_region",
    "candidate_sex",
    "candidate_type",
    "education_level",
    "poll_type",
    "deposit_lost"
]

for column in df.columns:
    if "profession" in column.lower():
        remove_columns.append(column)

df = df.drop(columns=remove_columns)

df.to_csv(output_file, index=False)

