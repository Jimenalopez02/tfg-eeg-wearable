from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED_CSV = Path("predictions_log_concentracion.csv")
TIMELINE_CSV = Path("replay_timeline_concentracion.csv")
FS = 250

if not PRED_CSV.exists():
    raise FileNotFoundError(f"No encuentro {PRED_CSV.resolve()}")

if not TIMELINE_CSV.exists():
    raise FileNotFoundError(f"No encuentro {TIMELINE_CSV.resolve()}")

df_pred = pd.read_csv(PRED_CSV)
df_timeline = pd.read_csv(TIMELINE_CSV)

df_pred["time_sec"] = df_pred["sample_count"] / FS

expected = np.zeros(len(df_pred), dtype=int)

for _, row in df_timeline.iterrows():
    mask = (df_pred["time_sec"] >= row["start_sec"]) & (df_pred["time_sec"] < row["end_sec"])
    expected[mask] = int(row["label"])

df_pred["state_expected"] = expected

acc = np.mean(df_pred["pred_class_smooth"].values == df_pred["state_expected"].values)
print(f"Accuracy temporal frente al replay fisiológico esperado: {acc:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(df_pred["time_sec"], df_pred["prob_calculo_mental"], label="Probabilidad cálculo mental")
plt.step(df_pred["time_sec"], df_pred["state_expected"], where="post", linestyle="--", label="Estado esperado replay")
plt.xlabel("Tiempo (s)")
plt.ylabel("Valor")
plt.title("Probabilidad predicha vs replay fisiológico esperado")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("replay_probabilidad_vs_estado.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 5))
plt.step(df_pred["time_sec"], df_pred["pred_class_smooth"], where="post", label="Clase predicha suavizada")
plt.step(df_pred["time_sec"], df_pred["state_expected"], where="post", linestyle="--", label="Estado esperado replay")
plt.xlabel("Tiempo (s)")
plt.ylabel("Clase")
plt.title("Clase predicha vs replay fisiológico esperado")
plt.yticks([0, 1], ["Reposo", "Cálculo mental"])
plt.legend()
plt.tight_layout()
plt.savefig("replay_clase_vs_estado.png", dpi=200)
plt.show()