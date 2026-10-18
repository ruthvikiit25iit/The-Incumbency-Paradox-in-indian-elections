import sys
import pandas as pd

source = sys.argv[1]
ls = pd.read_csv(source)

incumbents = ls[ls["incumbent"] == True].drop(columns=["incumbent"])
non_incumbents = ls[ls["incumbent"] == False].drop(columns=["incumbent"])

incumbents.to_csv("filtered_data/incumbents.csv", index=False)
non_incumbents.to_csv("filtered_data/non_incumbents.csv", index=False)
