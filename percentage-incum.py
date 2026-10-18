import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file, low_memory=False)

result = []

for year in sorted(df["year"].unique()):
    year_data = df[df["year"] == year]

    incumbents_true = year_data[year_data["incumbent"] == True]
    incumbents_false = year_data[year_data["incumbent"] == False]

    total_participated = len(year_data)

    incumbent_wins = len(incumbents_true[incumbents_true["position"] == 1])
    non_incumbent_wins = len(incumbents_false[incumbents_false["position"] == 1])

    if len(incumbents_true) > 0:
        incumbency_win_percentage = (incumbent_wins / len(incumbents_true)) * 100
    else:
        incumbency_win_percentage = 0

    if len(incumbents_false) > 0:
        non_incumbency_win_percentage = (
            non_incumbent_wins / len(incumbents_false)
        ) * 100
    else:
        non_incumbency_win_percentage = 0

    diff = incumbency_win_percentage - non_incumbency_win_percentage

    result.append([
        year,
        len(incumbents_true),
        len(incumbents_false),
        total_participated,
        incumbent_wins,
        non_incumbent_wins,
        incumbency_win_percentage,
        non_incumbency_win_percentage,
        diff
    ])

new_df = pd.DataFrame(result, columns=[
    "year",
    "no_of_incumbents_true",
    "no_of_incumbents_false",
    "total_no_participated",
    "no_of_incumbents_won",
    "no_of_non_incumbents_won",
    "incumbency_win_percentage",
    "non_incumbency_win_percentage",
    "diff"
])

new_df.to_csv(output_file, index=False)

print("New CSV created:", output_file)
