
from pathlib import Path
import matplotlib.pyplot as plt

from multiple_regression import read_analysis_rows, fit_model

rows = read_analysis_rows()
model = fit_model(rows)

predicted = model["predicted"]
residuals = model["actual"] - predicted

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(predicted, residuals, alpha=0.35, s=18)
ax.axhline(0, color="red", linestyle="--")

ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted Values")
ax.grid(alpha=0.2)

plt.tight_layout()

output = Path("visualisations/residuals_vs_fitted.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=200)
plt.show()

