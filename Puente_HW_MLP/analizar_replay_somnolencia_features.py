from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GT_CSV = Path("replay_groundtruth_somnolencia.csv")
PRED_CSV = Path("predictions_log_somnolencia_features.csv")

if not GT_CSV.exists():
    raise FileNotFoundError(f"No encuentro {GT_CSV.resolve()}")

if not PRED_CSV.exists():
    raise FileNotFoundError(f"No encuentro {PRED_CSV.resolve()}")

df_gt = pd.read_csv(GT_CSV)
df_pred = pd.read_csv(PRED_CSV)

n = min(len(df_gt), len(df_pred))
df_gt = df_gt.iloc[:n].copy()
df_pred = df_pred.iloc[:n].copy()

y_true = df_gt["perclos_true"].values.astype(float)
y_pred = df_pred["prediction_smooth"].values.astype(float)

rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
mae = float(np.mean(np.abs(y_pred - y_true)))

if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
    corr = np.nan
else:
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])

print(f"RMSE online pseudo-real: {rmse:.4f}")
print(f"MAE  online pseudo-real: {mae:.4f}")
print(f"CORR online pseudo-real: {corr:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(df_gt["time_sec"], y_true, label="PERCLOS real")
plt.plot(df_gt["time_sec"], y_pred, label="PERCLOS predicho")
plt.xlabel("Tiempo (s)")
plt.ylabel("PERCLOS")
plt.title("Somnolencia: replay de features reales vs predicción online")
plt.legend()
plt.tight_layout()
plt.savefig("replay_somnolencia_real_vs_pred.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("PERCLOS real")
plt.ylabel("PERCLOS predicho")
plt.title("Dispersión: real vs predicho (online pseudo-real)")
plt.tight_layout()
plt.savefig("replay_somnolencia_dispersion.png", dpi=200)
plt.show()