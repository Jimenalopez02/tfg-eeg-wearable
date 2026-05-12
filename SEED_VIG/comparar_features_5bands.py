from pathlib import Path
import numpy as np
import scipy.io as sio

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ---------------------------------
# 1) Rutas
# ---------------------------------
feature_dir = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
label_dir = Path(r"perclos_labels\perclos_labels")

feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))

if not feature_files:
    raise FileNotFoundError(f"No encuentro features en: {feature_dir.resolve()}")

if not label_files:
    raise FileNotFoundError(f"No encuentro labels en: {label_dir.resolve()}")

label_map = {f.name: f for f in label_files}

# ---------------------------------
# 2) Features a comparar
# ---------------------------------
feature_names = [
    "psd_movingAve",
    "psd_LDS",
    "de_movingAve",
    "de_LDS",
]

# ---------------------------------
# 3) Construcción del dataset
# ---------------------------------
def build_dataset(feature_key):
    X_all = []
    y_all = []
    groups = []

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"No encuentro label para {f_path.name}")

        l_path = label_map[f_path.name]

        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(l_path)

        if feature_key not in mat_f:
            raise KeyError(f"La variable '{feature_key}' no está en {f_path.name}")

        X = mat_f[feature_key]      # (4, 885, 5)
        y = mat_l["perclos"]        # (885, 1)

        # Reordenar a (muestras, canales, bandas)
        X = np.transpose(X, (1, 0, 2))   # -> (885, 4, 5)

        # Aplanar a (muestras, 20)
        X = X.reshape(X.shape[0], -1)    # -> (885, 20)

        # y a vector
        y = y.reshape(-1)                # -> (885,)

        X_all.append(X)
        y_all.append(y)
        groups.extend([exp_id] * len(y))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    return X_all, y_all, groups

# ---------------------------------
# 4) Evaluación
# ---------------------------------
def evaluate_feature(feature_key):
    X_all, y_all, groups = build_dataset(feature_key)

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

        if np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12:
            cor = 0.0
        else:
            cor = np.corrcoef(y_test, y_pred)[0, 1]

        cors.append(cor)
        rmses.append(rmse)

    return {
        "feature": feature_key,
        "cor_mean": float(np.mean(cors)),
        "cor_std": float(np.std(cors)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "n_samples": int(len(y_all)),
        "n_features": int(X_all.shape[1]),
    }

# ---------------------------------
# 5) Comparación final
# ---------------------------------
results = []

print("Comparando features de EEG_Feature_5Bands...\n")

for feat in feature_names:
    print(f"Evaluando: {feat}")
    res = evaluate_feature(feat)
    results.append(res)
    print(f"  COR medio  = {res['cor_mean']:.4f} ± {res['cor_std']:.4f}")
    print(f"  RMSE medio = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}\n")

results_sorted = sorted(results, key=lambda r: (-r["cor_mean"], r["rmse_mean"]))

print("\n===== RANKING FINAL =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. {r['feature']}: "
        f"COR={r['cor_mean']:.4f} ± {r['cor_std']:.4f}, "
        f"RMSE={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
    )