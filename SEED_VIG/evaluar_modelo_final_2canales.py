from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------
# 1) Rutas
# ---------------------------------
feature_dir = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
label_dir = Path(r"perclos_labels\perclos_labels")
out_dir = Path("resultados_finales")
out_dir.mkdir(exist_ok=True)

feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))
label_map = {f.name: f for f in label_files}

if not feature_files:
    raise FileNotFoundError("No encuentro archivos de features.")
if not label_files:
    raise FileNotFoundError("No encuentro archivos de labels.")

# ---------------------------------
# 2) Dataset final
# ---------------------------------
def build_dataset_2ch(feature_key="de_movingAve", channel_pair=(1, 2)):
    X_all = []
    y_all = []
    groups = []

    ch_a, ch_b = channel_pair

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"No encuentro label para {f_path.name}")

        l_path = label_map[f_path.name]
        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(l_path)

        X = mat_f[feature_key]      # (4, 885, 5)
        y = mat_l["perclos"]        # (885, 1)

        X = X[[ch_a, ch_b], :, :]         # (2, 885, 5)
        X = np.transpose(X, (1, 0, 2))    # (885, 2, 5)
        X = X.reshape(X.shape[0], -1)     # (885, 10)
        y = y.reshape(-1)                 # (885,)

        X_all.append(X)
        y_all.append(y)
        groups.extend([exp_id] * len(y))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    return X_all, y_all, groups


X_all, y_all, groups = build_dataset_2ch()

print("Shape X:", X_all.shape)
print("Shape y:", y_all.shape)

# ---------------------------------
# 3) Modelo final
# ---------------------------------
def make_final_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ))
    ])

# ---------------------------------
# 4) Validación cruzada y predicciones
# ---------------------------------
gkf = GroupKFold(n_splits=5)

y_pred_all = np.zeros_like(y_all, dtype=float)
fold_metrics = []

saved_first_fold = False

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    model = make_final_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_all[test_idx] = y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cor = 0.0 if (np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12) else np.corrcoef(y_test, y_pred)[0, 1]

    fold_metrics.append((fold, cor, rmse))
    print(f"Fold {fold}: COR={cor:.4f}, RMSE={rmse:.4f}")

    if not saved_first_fold:
        plt.figure(figsize=(12, 5))
        plt.plot(y_test, label="PERCLOS real")
        plt.plot(y_pred, label="PERCLOS predicho")
        plt.title(f"Comparación temporal - Fold {fold}")
        plt.xlabel("Muestras")
        plt.ylabel("PERCLOS")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "comparacion_temporal_fold1.png", dpi=200)
        plt.close()
        saved_first_fold = True

# ---------------------------------
# 5) Métricas globales
# ---------------------------------
rmse_global = np.sqrt(mean_squared_error(y_all, y_pred_all))
cor_global = 0.0 if (np.std(y_all) < 1e-12 or np.std(y_pred_all) < 1e-12) else np.corrcoef(y_all, y_pred_all)[0, 1]

print("\n===== RESULTADO GLOBAL =====")
print(f"COR global  = {cor_global:.4f}")
print(f"RMSE global = {rmse_global:.4f}")

# ---------------------------------
# 6) Gráfica dispersión real vs predicho
# ---------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_all, y_pred_all, s=8, alpha=0.4)
plt.xlabel("PERCLOS real")
plt.ylabel("PERCLOS predicho")
plt.title("Dispersión: real vs predicho")
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, linestyle="--")
plt.tight_layout()
plt.savefig(out_dir / "dispersion_real_vs_predicho.png", dpi=200)
plt.close()

# ---------------------------------
# 7) Histograma de errores
# ---------------------------------
errors = y_pred_all - y_all

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40)
plt.xlabel("Error (predicho - real)")
plt.ylabel("Frecuencia")
plt.title("Histograma de errores")
plt.tight_layout()
plt.savefig(out_dir / "histograma_errores.png", dpi=200)
plt.close()

# ---------------------------------
# 8) Guardar métricas a txt
# ---------------------------------
with open(out_dir / "metricas_finales.txt", "w", encoding="utf-8") as f:
    f.write("Modelo final 2 canales\n")
    f.write("Feature: de_movingAve\n")
    f.write("Canales: (1, 2)\n")
    f.write("MLP hidden_layer_sizes=(32,), alpha=0.001\n\n")

    for fold, cor, rmse in fold_metrics:
        f.write(f"Fold {fold}: COR={cor:.4f}, RMSE={rmse:.4f}\n")

    f.write("\n")
    f.write(f"COR global: {cor_global:.4f}\n")
    f.write(f"RMSE global: {rmse_global:.4f}\n")

print(f"\nArchivos guardados en: {out_dir.resolve()}")
print("Proceso completado.")

# ---------------------------------
# 9) Entrenamiento final con todos los datos
# ---------------------------------
print("Empezando entrenamiento final con todos los datos...")
final_model = make_final_model()
final_model.fit(X_all, y_all)
print("Entrenamiento final terminado.")

# ---------------------------------
# 10) Guardar pipeline completo
# ---------------------------------
pipeline_path = out_dir / "pipeline_mlp_2canales.pkl"
joblib.dump(final_model, pipeline_path)
print(f"Pipeline final guardado en: {pipeline_path.resolve()}")