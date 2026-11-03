
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])

pairs = df[
    (pd.to_numeric(df["poll_type"], errors="coerce") == 0) &
    (pd.to_numeric(df["previous_contest_poll_type"], errors="coerce") == 0) &
    (df["previous_contest_match_status"].astype(str).str.upper() == "MATCHED")
].copy()

pairs["X"] = pd.to_numeric(
    pairs["previous_contest_signed_victory_margin_percentage"],
    errors="coerce"
)

pairs["Y"] = pd.to_numeric(
    pairs["signed_victory_margin_percentage"],
    errors="coerce"
)

pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y"])

data = pairs[
    pairs["same_constituency"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
].copy()

X = data["X"].to_numpy()
Y = data["Y"].to_numpy()

x_mean = np.mean(X)
y_mean = np.mean(Y)

beta1 = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
beta0 = y_mean - beta1 * x_mean

Y_pred = beta0 + beta1 * X

r2 = 1 - (
    np.sum((Y - Y_pred) ** 2) /
    np.sum((Y - y_mean) ** 2)
)

sign = "+" if beta1 >= 0 else "-"

print("SAME CONSTITUENCY LINEAR REGRESSION")
print("-----------------------------------")
print(f"Observations: {len(data)}")
print(f"Intercept (beta_0): {beta0:.6f}")
print(f"Slope (beta_1): {beta1:.6f}")
print(f"Fitted line: Y_hat = {beta0:.4f} {sign} {abs(beta1):.4f}X")
print(f"R-squared: {r2:.6f}")

x_line = np.linspace(X.min(), X.max(), 200)
y_line = beta0 + beta1 * x_line

plt.figure(figsize=(9, 6))
plt.scatter(X, Y, alpha=0.22, s=16, label="Candidate observations")
plt.plot(
    x_line,
    y_line,
    color="red",
    linewidth=2,
    label=f"Y = {beta0:.3f} {sign} {abs(beta1):.3f}X"
)

plt.xlabel("Previous General Election Signed Margin (%)")
plt.ylabel("Current General Election Signed Margin (%)")
plt.title("Previous vs Current Electoral Margins")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

project_root = Path(__file__).resolve().parent.parent

plt.savefig(
    project_root / "visualisations" / "simple_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
