import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

input_file = sys.argv[1]

df = pd.read_csv(input_file, on_bad_lines="skip")

margin_column = df.columns[-1]

df[margin_column] = pd.to_numeric(df[margin_column], errors="coerce")

df = df.dropna(subset=[margin_column])

df_less_than_5 = df[df[margin_column] < 5]
df_less_than_2 = df[df[margin_column] < 2]

folder = os.path.dirname(input_file)
name = os.path.splitext(os.path.basename(input_file))[0]

csv_5 = os.path.join(folder, name + "_less_than_5.csv")
csv_2 = os.path.join(folder, name + "_less_than_2.csv")

df_less_than_5.to_csv(csv_5, index=False)
df_less_than_2.to_csv(csv_2, index=False)

current_winners_5 = df_less_than_5[df_less_than_5["position"] == 1]
current_winners_2 = df_less_than_2[df_less_than_2["position"] == 1]

labels = []
counts_5 = []
counts_2 = []

for start in range(0, 100, 5):
    end = start + 5

    labels.append(str(start) + "-" + str(end))

    count_5 = len(current_winners_5[
        (current_winners_5["margin_percentage"] >= start) &
        (current_winners_5["margin_percentage"] < end)
    ])

    count_2 = len(current_winners_2[
        (current_winners_2["margin_percentage"] >= start) &
        (current_winners_2["margin_percentage"] < end)
    ])

    counts_5.append(count_5)
    counts_2.append(count_2)

plt.figure(figsize=(12, 6))

bars = plt.bar(labels, counts_5)

for bar, count in zip(bars, counts_5):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count,
        str(count),
        ha="center",
        va="bottom"
    )

plt.xlabel("Current winning margin percentage bracket")
plt.ylabel("Number of winning candidates")
plt.title("Current Winners with Last Won Margin Below 5%")
plt.xticks(rotation=45)
plt.tight_layout()

graph_5 = os.path.join(
    folder,
    name + "_current_winners_last_margin_below_5.png"
)

plt.savefig(graph_5)
plt.show()

plt.figure(figsize=(12, 6))

bars = plt.bar(labels, counts_2)

for bar, count in zip(bars, counts_2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count,
        str(count),
        ha="center",
        va="bottom"
    )

plt.xlabel("Current winning margin percentage bracket")
plt.ylabel("Number of winning candidates")
plt.title("Current Winners with Last Won Margin Below 2%")
plt.xticks(rotation=45)
plt.tight_layout()

graph_2 = os.path.join(
    folder,
    name + "_current_winners_last_margin_below_2.png"
)

plt.savefig(graph_2)
plt.show()
