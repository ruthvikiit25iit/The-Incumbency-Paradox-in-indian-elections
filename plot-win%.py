import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

input_file = sys.argv[1]

df = pd.read_csv(input_file)
df = df.sort_values("year")

plt.plot(
    df["year"],
    df["incumbency_win_percentage"],
    marker="o",
    label="Incumbency win percentage"
)

plt.plot(
    df["year"],
    df["non_incumbency_win_percentage"],
    marker="o",
    label="Non-incumbency win percentage"
)

plt.xlabel("Year")
plt.ylabel("Win Percentage")
plt.title("Incumbency vs Non-incumbency Win Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_file = os.path.splitext(input_file)[0] + "_graph.png"

plt.savefig(output_file)
plt.show()

print("Graph created:", output_file)
