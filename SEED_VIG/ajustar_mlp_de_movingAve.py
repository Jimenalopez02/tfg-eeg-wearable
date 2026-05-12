from pathlib import Path
import itertools
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
# 2) Construcción del dataset
# ---------------------------------
def build_dataset(feature_key="de_movingAve"):
    X_all = []
    y_all = []
    groups = []

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"No encuentro label para {f_path.name}")

        l_path = label_map[f_path.name]

        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(l_path)

        X = mat_f[feature_key]      # (4, 885, 5)
        y = mat_l["perclos"]        # (885, 1)

        X = np.transpose(X, (1, 0, 2))   # (885, 4, 5)
        X = X.reshape(X.shape[0], -1)    # (885, 20)
        y = y.reshape(-1)                # (885,)

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
# 4) Dataset fijo
# ---------------------------------
X_all, y_all, groups = build_dataset(feature_key="de_movingAve")

print("Shape X:", X_all.shape)
print("Shape y:", y_all.shape)

# ---------------------------------
# 5) Búsqueda de hiperparámetros
# ---------------------------------
hidden_layer_options = [
    (32,),
    (64,),
    (64, 32),
    (64, 64),
    (128, 64),
]

alpha_options = [1e-4, 1e-3, 1e-2]

configs = list(itertools.product(hidden_layer_options, alpha_options))

results = []

for i, (hidden_layers, alpha) in enumerate(configs, start=1):
    print(f"\nConfiguración {i}/{len(configs)}")
    print(f"  hidden_layer_sizes = {hidden_layers}")
    print(f"  alpha = {alpha}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ))
    ])

    res = evaluate_model(model, X_all, y_all, groups)
    results.append({
        "hidden_layers": hidden_layers,
        "alpha": alpha,
        **res
    })

    print(f"  COR medio  = {res['cor_mean']:.4f} ± {res['cor_std']:.4f}")
    print(f"  RMSE medio = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")

# ---------------------------------
# 6) Ranking final
# ---------------------------------
results_sorted = sorted(results, key=lambda r: (-r["cor_mean"], r["rmse_mean"]))

print("\n===== RANKING FINAL MLP =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. hidden_layers={r['hidden_layers']}, alpha={r['alpha']}: "
        f"COR={r['cor_mean']:.4f} ± {r['cor_std']:.4f}, "
        f"RMSE={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
    )

best = results_sorted[0]
print("\n===== MEJOR CONFIGURACIÓN =====")
print(f"hidden_layer_sizes = {best['hidden_layers']}")
print(f"alpha = {best['alpha']}")
print(f"COR medio  = {best['cor_mean']:.4f} ± {best['cor_std']:.4f}")
print(f"RMSE medio = {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}")