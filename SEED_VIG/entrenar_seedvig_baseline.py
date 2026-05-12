from pathlib import Path
import numpy as np
import scipy.io as sio

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1) Rutas
# -----------------------------
feature_dir = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
label_dir = Path(r"perclos_labels\perclos_labels")

feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))

if len(feature_files) == 0:
    raise FileNotFoundError(f"No encuentro features en {feature_dir.resolve()}")

if len(label_files) == 0:
    raise FileNotFoundError(f"No encuentro labels en {label_dir.resolve()}")

# emparejamiento por nombre
label_map = {f.name: f for f in label_files}

# -----------------------------
# 2) Cargar todos los experimentos
# -----------------------------
X_all = []
y_all = []
groups = []

for exp_id, f_path in enumerate(feature_files):
    if f_path.name not in label_map:
        raise ValueError(f"No encuentro label para {f_path.name}")

    l_path = label_map[f_path.name]

    mat_f = sio.loadmat(f_path)
    mat_l = sio.loadmat(l_path)

    # Elegimos una representación para empezar
    X = mat_f["de_LDS"]     # shape original: (4, 885, 5)
    y = mat_l["perclos"]    # shape original: (885, 1)

    # Reordenar a (muestras, canales, bandas)
    X = np.transpose(X, (1, 0, 2))   # -> (885, 4, 5)

    # Aplanar cada muestra a un vector de 20 features
    X = X.reshape(X.shape[0], -1)    # -> (885, 20)

    # Vector de etiquetas
    y = y.reshape(-1)                # -> (885,)

    # Guardar
    X_all.append(X)
    y_all.append(y)
    groups.extend([exp_id] * len(y))

X_all = np.vstack(X_all)             # (23*885, 20)
y_all = np.concatenate(y_all)        # (23*885,)
groups = np.array(groups)

print("Shape total X:", X_all.shape)
print("Shape total y:", y_all.shape)
print("Número de experimentos:", len(feature_files))
print("Mínimo y:", np.min(y_all))
print("Máximo y:", np.max(y_all))
print("Media y:", np.mean(y_all))

# -----------------------------
# 3) Validación por grupos
# -----------------------------
# Recomendación: no mezclar muestras del mismo experimento en train y test
gkf = GroupKFold(n_splits=5)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Ridge(alpha=1.0))
])

cors = []
rmses = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # COR (coeficiente de correlación)
    if np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12:
        cor = 0.0
    else:
        cor = np.corrcoef(y_test, y_pred)[0, 1]

    cors.append(cor)
    rmses.append(rmse)

    print(f"\nFold {fold}")
    print(f"  COR  = {cor:.4f}")
    print(f"  RMSE = {rmse:.4f}")

print("\n===== RESULTADO FINAL =====")
print(f"COR medio  = {np.mean(cors):.4f} ± {np.std(cors):.4f}")
print(f"RMSE medio = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")