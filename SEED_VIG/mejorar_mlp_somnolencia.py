from pathlib import Path
import numpy as np
import scipy.io as sio

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# 1) Rutas
# --------------------------------------------------
feature_dir = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
label_dir = Path(r"perclos_labels\perclos_labels")

feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))

if not feature_files:
    raise FileNotFoundError(f"No encuentro features en: {feature_dir.resolve()}")

if not label_files:
    raise FileNotFoundError(f"No encuentro labels en: {label_dir.resolve()}")

label_map = {f.name: f for f in label_files}


# --------------------------------------------------
# 2) Parámetros del modelo base
# --------------------------------------------------
FEATURE_KEY = "de_movingAve"
CHANNEL_PAIR = (1, 2)

HIDDEN_LAYER_SIZES = (32,)
ALPHA = 0.001

N_SPLITS = 5


# --------------------------------------------------
# 3) Funciones auxiliares
# --------------------------------------------------
def corrcoef_safe(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def moving_average_1d(x, k=5):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    for i in range(len(x)):
        start = max(0, i - k + 1)
        y[i] = np.mean(x[start:i + 1])

    return y


def build_dataset_2ch():
    X_all = []
    y_all = []
    groups = []

    ch_a, ch_b = CHANNEL_PAIR

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"No encuentro label para {f_path.name}")

        l_path = label_map[f_path.name]

        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(l_path)

        X = mat_f[FEATURE_KEY]       # (4, 885, 5)
        y = mat_l["perclos"]         # (885, 1)

        # Selección de 2 canales
        X = X[[ch_a, ch_b], :, :]    # (2, 885, 5)

        # Reordenar a (muestras, canales, bandas)
        X = np.transpose(X, (1, 0, 2))   # (885, 2, 5)

        # Aplanar a 10 features
        X = X.reshape(X.shape[0], -1)    # (885, 10)
        y = y.reshape(-1)                # (885,)

        X_all.append(X)
        y_all.append(y)
        groups.extend([exp_id] * len(y))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    return X_all, y_all, groups


def build_dataset_temporal_context(context=5):
    """
    Crea muestras con contexto temporal.
    Si context=5, cada muestra contiene:
    [features t-4, t-3, t-2, t-1, t]
    Entrada final: 5 * 10 = 50 features.

    Importante:
    El contexto se construye dentro de cada experimento, no mezclando experimentos.
    """
    X_all = []
    y_all = []
    groups = []

    ch_a, ch_b = CHANNEL_PAIR

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"No encuentro label para {f_path.name}")

        l_path = label_map[f_path.name]

        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(l_path)

        X = mat_f[FEATURE_KEY]       # (4, 885, 5)
        y = mat_l["perclos"].reshape(-1)

        X = X[[ch_a, ch_b], :, :]    # (2, 885, 5)
        X = np.transpose(X, (1, 0, 2))   # (885, 2, 5)
        X = X.reshape(X.shape[0], -1)    # (885, 10)

        X_ctx = []
        y_ctx = []

        for i in range(context - 1, len(X)):
            block = X[i - context + 1:i + 1].reshape(-1)
            X_ctx.append(block)
            y_ctx.append(y[i])

        X_ctx = np.array(X_ctx)
        y_ctx = np.array(y_ctx)

        X_all.append(X_ctx)
        y_all.append(y_ctx)
        groups.extend([exp_id] * len(y_ctx))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups = np.array(groups)

    return X_all, y_all, groups


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation="relu",
            solver="adam",
            alpha=ALPHA,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])


def evaluate_variant(X, y, groups, variant_name, postprocess=None):
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_results = []

    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if postprocess is not None:
            y_pred = postprocess(y_pred)

        fold_cor = corrcoef_safe(y_test, y_pred)
        fold_rmse = rmse(y_test, y_pred)

        fold_results.append((fold_cor, fold_rmse))

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        print(
            f"  Fold {fold}: "
            f"COR={fold_cor:.4f}, RMSE={fold_rmse:.4f}"
        )

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    cor_global = corrcoef_safe(y_true_all, y_pred_all)
    rmse_global = rmse(y_true_all, y_pred_all)

    cors = np.array([r[0] for r in fold_results])
    rmses = np.array([r[1] for r in fold_results])

    print(f"\nResultado {variant_name}")
    print(f"  COR medio folds  = {np.nanmean(cors):.4f} ± {np.nanstd(cors):.4f}")
    print(f"  RMSE medio folds = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  COR global       = {cor_global:.4f}")
    print(f"  RMSE global      = {rmse_global:.4f}")

    return {
        "variant": variant_name,
        "cor_mean": float(np.nanmean(cors)),
        "cor_std": float(np.nanstd(cors)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "cor_global": float(cor_global),
        "rmse_global": float(rmse_global),
    }


# --------------------------------------------------
# 4) Evaluación de variantes
# --------------------------------------------------
print("\nConstruyendo dataset base 2 canales...")
X_base, y_base, groups_base = build_dataset_2ch()

print("X base shape:", X_base.shape)
print("y base shape:", y_base.shape)
print("grupos:", np.unique(groups_base).shape[0])

results = []

print("\n===== VARIANTE 1: MLP base =====")
res = evaluate_variant(
    X_base,
    y_base,
    groups_base,
    "MLP base",
    postprocess=None
)
results.append(res)


print("\n===== VARIANTE 2: MLP + clipping [0,1] =====")
res = evaluate_variant(
    X_base,
    y_base,
    groups_base,
    "MLP + clipping",
    postprocess=lambda y_pred: np.clip(y_pred, 0, 1)
)
results.append(res)


print("\n===== VARIANTE 3: MLP + clipping + suavizado k=5 =====")
res = evaluate_variant(
    X_base,
    y_base,
    groups_base,
    "MLP + clipping + smoothing k=5",
    postprocess=lambda y_pred: moving_average_1d(np.clip(y_pred, 0, 1), k=5)
)
results.append(res)


print("\nConstruyendo dataset con contexto temporal...")
X_ctx, y_ctx, groups_ctx = build_dataset_temporal_context(context=5)

print("X contexto shape:", X_ctx.shape)
print("y contexto shape:", y_ctx.shape)
print("grupos:", np.unique(groups_ctx).shape[0])


print("\n===== VARIANTE 4: MLP con contexto temporal 5 ventanas =====")
res = evaluate_variant(
    X_ctx,
    y_ctx,
    groups_ctx,
    "MLP contexto temporal 5 ventanas",
    postprocess=lambda y_pred: np.clip(y_pred, 0, 1)
)
results.append(res)


print("\n===== RANKING FINAL DE VARIANTES =====")
results_sorted = sorted(results, key=lambda r: (-r["cor_global"], r["rmse_global"]))

for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. {r['variant']}: "
        f"COR_global={r['cor_global']:.4f}, "
        f"RMSE_global={r['rmse_global']:.4f}, "
        f"COR_folds={r['cor_mean']:.4f} ± {r['cor_std']:.4f}, "
        f"RMSE_folds={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
    )
    