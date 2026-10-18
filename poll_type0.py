import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, low_memory=False)

df = df[df["poll_type"] == 0]

remove_columns = [
    "month",
    "sub_region",
    "candidate_sex",
    "candidate_type",
    "education_level",
    "poll_type"
]

for column in df.columns:
    if "profession" in column.lower():
        remove_columns.append(column)

df = df.drop(columns=remove_columns)

df.to_csv(output_file, index=False)

print("New CSV created:", output_file)
