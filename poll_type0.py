import pandas as pd

input_file = "srcs/in_tcpd_elections_general_elections.csv"
output_file = "filtered_data/in_tcpd_elections_poll_type_0.csv"

df = pd.read_csv(input_file, low_memory=False)
df = df[df["poll_type"] == 0]
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows to {output_file}")
