import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

last_column = df.columns[-1]
df = df.dropna(subset=[last_column])

df.to_csv(output_file, index=False)

