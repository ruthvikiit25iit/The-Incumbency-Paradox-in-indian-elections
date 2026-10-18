import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

input_file = sys.argv[1]

df = pd.read_csv(input_file)
df = df.sort_values("year")

plt.plot(
    df["year"],
    df["diff"],
    marker="o",
    label="Diff percentage"
)

plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title("Difference in Incumbency and Non-incumbency Win Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_file = os.path.splitext(input_file)[0] + "_diff_graph.png"

plt.savefig(output_file)
plt.show()

print("Graph created:", output_file)
