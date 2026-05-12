from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("predictions_log_concentracion.csv")
FS = 250
STATE_DURATION_SEC = 30  # mismo valor que en el emulador

if not CSV_PATH.exists():
    raise FileNotFoundError(f"No encuentro {CSV_PATH.resolve()}")

df = pd.read_csv(CSV_PATH)

if "sample_count" not in df.columns:
    raise ValueError("No encuentro la columna 'sample_count' en el CSV")

if "prob_calculo_mental" not in df.columns:
    raise ValueError("No encuentro la columna 'prob_calculo_mental' en el CSV")

if "pred_class_smooth" not in df.columns:
    raise ValueError("No encuentro la columna 'pred_class_smooth' en el CSV")

# tiempo en segundos aproximado
df["time_sec"] = df["sample_count"] / FS

# estado esperado del emulador:
# 0 = reposo, 1 = cálculo mental
df["state_expected"] = ((df["time_sec"] // STATE_DURATION_SEC).astype(int)) % 2

# gráfica 1: probabilidad de cálculo mental
plt.figure(figsize=(12, 5))
plt.plot(df["time_sec"], df["prob_calculo_mental"], label="Probabilidad cálculo mental")
plt.plot(df["time_sec"], df["state_expected"], label="Estado esperado emulador", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("Valor")
plt.title("Probabilidad predicha vs estado esperado")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("probabilidad_vs_estado_esperado.png", dpi=200)
plt.show()

# gráfica 2: clase suavizada predicha
plt.figure(figsize=(12, 5))
plt.step(df["time_sec"], df["pred_class_smooth"], where="post", label="Clase predicha suavizada")
plt.step(df["time_sec"], df["state_expected"], where="post", label="Estado esperado emulador", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("Clase")
plt.title("Clase predicha vs estado esperado")
plt.yticks([0, 1], ["Reposo", "Cálculo mental"])
plt.legend()
plt.tight_layout()
plt.savefig("clase_vs_estado_esperado.png", dpi=200)
plt.show()

# resumen numérico simple
acc = np.mean(df["pred_class_smooth"].values == df["state_expected"].values)
print(f"Accuracy temporal aproximada frente al estado esperado: {acc:.4f}")

