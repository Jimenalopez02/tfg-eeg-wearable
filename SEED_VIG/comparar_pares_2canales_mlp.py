from pathlib import Path
from itertools import combinations
import numpy as np
import scipy.io as sio

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

feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))

if not feature_files:
    raise FileNotFoundError(f"No encuentro features en: {feature_dir.resolve()}")

if not label_files:
    raise FileNotFoundError(f"No encuentro labels en: {label_dir.resolve()}")

label_map = {f.name: f for f in label_files}

# ---------------------------------
# 2) Construcción del dataset para un par de canales
# ---------------------------------
def build_dataset_2ch(feature_key="de_movingAve", channel_pair=(0, 1)):
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

        X = mat_f[feature_key]      # shape original: (4, 885, 5)
        y = mat_l["perclos"]        # shape original: (885, 1)

        # Seleccionar solo 2 canales
        X = X[[ch_a, ch_b], :, :]   # -> (2, 885, 5)

        # Reordenar a (muestras, canales, bandas)
        X = np.transpose(X, (1, 0, 2))   # -> (885, 2, 5)

        # Aplanar a (muestras, 10)
        X = X.reshape(X.shape[0], -1)    # -> (885, 10)

        y = y.reshape(-1)                # -> (885,)

        X_all.append(X)
        y_all.append(y)
        groups.extend([exp_id] * len(y))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    return X_all, y_all, groups

# ---------------------------------
# 3) Evaluación
# ---------------------------------
def evaluate_model(model, X_all, y_all, groups):
    gkf = GroupKFold(n_splits=5)

    cors = []
    rmses = []

    for train_idx, test_idx in gkf.split(X_all, y_all, groups=groups):
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
        "cor_mean": float(np.mean(cors)),
        "cor_std": float(np.std(cors)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
    }

# ---------------------------------
# 4) Modelo base para la comparación
# ---------------------------------
def make_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42
        ))
    ])

# ---------------------------------
# 5) Comparar todas las parejas de 2 canales
# ---------------------------------
channel_pairs = list(combinations(range(4), 2))
results = []

print("Comparando todas las combinaciones de 2 canales...\n")

for pair in channel_pairs:
    print(f"Evaluando par de canales: {pair}")

    X_all, y_all, groups = build_dataset_2ch(
        feature_key="de_movingAve",
        channel_pair=pair
    )

    model = make_mlp()
    res = evaluate_model(model, X_all, y_all, groups)

    results.append({
        "pair": pair,
        "n_features": X_all.shape[1],
        **res
    })

    print(f"  Shape X: {X_all.shape}")
    print(f"  COR medio  = {res['cor_mean']:.4f} ± {res['cor_std']:.4f}")
    print(f"  RMSE medio = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}\n")

# ---------------------------------
# 6) Ranking final
# ---------------------------------
results_sorted = sorted(results, key=lambda r: (-r["cor_mean"], r["rmse_mean"]))

print("\n===== RANKING FINAL 2 CANALES =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. canales={r['pair']}: "
        f"COR={r['cor_mean']:.4f} ± {r['cor_std']:.4f}, "
        f"RMSE={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
    )

best = results_sorted[0]
print("\n===== MEJOR PAR =====")
print(f"Par de canales: {best['pair']}")
print(f"COR medio  = {best['cor_mean']:.4f} ± {best['cor_std']:.4f}")
print(f"RMSE medio = {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}")