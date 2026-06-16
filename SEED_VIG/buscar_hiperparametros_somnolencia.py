
from pathlib import Path
import itertools
import numpy as np
import scipy.io as sio
import warnings

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================
# CONFIGURACIÓN — debe coincidir con evaluar_somnolencia_mejorado.py v2
# ==============================================================
FEATURE_DIR = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
LABEL_DIR   = Path(r"perclos_labels\perclos_labels")
OUT_DIR     = Path("resultados_mejorados")

FEATURE_KEY  = "de_movingAve"
CHANNEL_PAIR = (1, 2)

USE_DELTA_FEATURES     = True
USE_ARTIFACT_REJECTION = True
USE_EMA_POSTPROCESS    = True
EMA_ALPHA              = 0.2
ARTIFACT_ZSCORE_THRESH = 3.5

# ==============================================================
# ESPACIO DE BÚSQUEDA
# ==============================================================
HIDDEN_OPTIONS = [
    (32,),          # baseline actual
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (128, 64, 32),  # 3 capas
]

ALPHA_OPTIONS = [1e-4, 1e-3, 1e-2]

LR_OPTIONS = [1e-3]   # fijo: adam es robusto a esto; añadir 5e-4 si quieres más fino

CONFIGS = list(itertools.product(HIDDEN_OPTIONS, ALPHA_OPTIONS, LR_OPTIONS))


# ==============================================================
# UTILIDADES (idénticas a v2)
# ==============================================================

def corrcoef_safe(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ema_smooth(y, alpha=0.2):
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out

def reject_artifact_windows(X, y, thresh=3.5):
    norms = np.linalg.norm(X, axis=1)
    z = (norms - np.mean(norms)) / (np.std(norms) + 1e-12)
    mask = np.abs(z) <= thresh
    return X[mask], y[mask], mask

def add_delta_and_asymmetry_features(X):
    delta = np.zeros_like(X)
    delta[1:] = X[1:] - X[:-1]
    ch_a_feats = X[:, 0:5]
    ch_b_feats = X[:, 5:10]
    asymmetry  = ch_a_feats - ch_b_feats
    return np.hstack([X, delta, asymmetry])


# ==============================================================
# CARGA DEL DATASET (una sola vez)
# ==============================================================

def build_full_dataset():
    feature_files = sorted(FEATURE_DIR.glob("*.mat"))
    label_files   = sorted(LABEL_DIR.glob("*.mat"))

    if not feature_files:
        raise FileNotFoundError(f"No hay features en: {FEATURE_DIR.resolve()}")
    if not label_files:
        raise FileNotFoundError(f"No hay labels en: {LABEL_DIR.resolve()}")

    label_map = {f.name: f for f in label_files}
    ch_a, ch_b = CHANNEL_PAIR

    X_list, y_list, groups_list = [], [], []
    n_rejected = 0
    n_total    = 0

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"Sin label para {f_path.name}")

        mat_f = sio.loadmat(f_path)
        mat_l = sio.loadmat(label_map[f_path.name])

        X = mat_f[FEATURE_KEY]
        y = mat_l["perclos"].reshape(-1)

        X = X[[ch_a, ch_b], :, :]
        X = np.transpose(X, (1, 0, 2))
        X = X.reshape(X.shape[0], -1)     # (N, 10)

        n_total += len(y)

        if USE_DELTA_FEATURES:
            X = add_delta_and_asymmetry_features(X)  # (N, 25)

        if USE_ARTIFACT_REJECTION:
            X, y, mask = reject_artifact_windows(X, y, thresh=ARTIFACT_ZSCORE_THRESH)
            n_rejected += np.sum(~mask)

        X_list.append(X)
        y_list.append(y)
        groups_list.extend([exp_id] * len(y))

    X_all  = np.vstack(X_list)
    y_all  = np.concatenate(y_list)
    groups = np.array(groups_list)

    pct = 100 * n_rejected / n_total
    print(f"  Dataset: {X_all.shape[0]} ventanas, {X_all.shape[1]} features, "
          f"{len(np.unique(groups))} sujetos")
    print(f"  Artefactos rechazados: {n_rejected}/{n_total} ({pct:.1f}%)")

    return X_all, y_all, groups


# ==============================================================
# EVALUACIÓN DE UNA CONFIGURACIÓN
# ==============================================================

def evaluate_config(X_all, y_all, groups, hidden, alpha, lr):
    logo = LeaveOneGroupOut()

    fold_cors  = []
    fold_rmses = []

    for train_idx, test_idx in logo.split(X_all, y_all, groups=groups):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if USE_EMA_POSTPROCESS:
            y_pred = ema_smooth(y_pred, alpha=EMA_ALPHA)

        y_pred = np.clip(y_pred, 0, 1)

        fold_cors.append(corrcoef_safe(y_test, y_pred))
        fold_rmses.append(rmse_score(y_test, y_pred))

    cors  = np.array(fold_cors)
    rmses = np.array(fold_rmses)

    return {
        "hidden": hidden,
        "alpha":  alpha,
        "lr":     lr,
        "cor_mean":  float(np.nanmean(cors)),
        "cor_std":   float(np.nanstd(cors)),
        "cor_min":   float(np.nanmin(cors)),
        "cor_max":   float(np.nanmax(cors)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std":  float(np.std(rmses)),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  BÚSQUEDA DE HIPERPARÁMETROS — MÓDULO SOMNOLENCIA v2")
    print("=" * 65)
    print(f"\n  Features: delta+asimetría={USE_DELTA_FEATURES} | "
          f"artefactos={USE_ARTIFACT_REJECTION} | EMA={USE_EMA_POSTPROCESS}")
    print(f"  Configuraciones a evaluar : {len(CONFIGS)}")
    print(f"  Folds por configuración   : 23 (LOSO)")
    print(f"  Total entrenamientos      : {len(CONFIGS) * 23}\n")

    # Carga única del dataset
    print("Cargando dataset...")
    X_all, y_all, groups = build_full_dataset()

    # Referencia: configuración actual del TFG
    print("\n  [REF] Configuración actual del TFG: hidden=(32,), alpha=0.001")
    print("  (aparecerá en el ranking para comparación directa)\n")

    results = []

    for i, (hidden, alpha, lr) in enumerate(CONFIGS, start=1):
        tag = f"hidden={hidden}, alpha={alpha:.0e}, lr={lr:.0e}"
        print(f"[{i:2d}/{len(CONFIGS)}] {tag}")

        res = evaluate_config(X_all, y_all, groups, hidden, alpha, lr)
        results.append(res)

        marker = " ◄ actual TFG" if hidden == (32,) and alpha == 0.001 else ""
        print(f"        COR  {res['cor_mean']:.4f} ± {res['cor_std']:.4f}  "
              f"[{res['cor_min']:.3f}, {res['cor_max']:.3f}]{marker}")
        print(f"        RMSE {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")

    # Ranking
    results_sorted = sorted(results,
                             key=lambda r: (-r["cor_mean"], r["rmse_mean"]))

    print("\n" + "=" * 65)
    print("  RANKING FINAL")
    print("=" * 65)
    print(f"  {'#':<3} {'Arquitectura':<22} {'alpha':<8} "
          f"{'COR medio':<20} {'RMSE medio':<18} {'COR rango'}")
    print("  " + "-" * 90)

    for rank, r in enumerate(results_sorted, start=1):
        marker = " ◄" if r["hidden"] == (32,) and r["alpha"] == 0.001 else ""
        print(
            f"  {rank:<3} {str(r['hidden']):<22} {r['alpha']:<8.0e} "
            f"{r['cor_mean']:.4f} ± {r['cor_std']:.4f}   "
            f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}   "
            f"[{r['cor_min']:.3f}, {r['cor_max']:.3f}]{marker}"
        )

    best = results_sorted[0]
    print(f"\n  MEJOR CONFIGURACIÓN:")
    print(f"    hidden_layer_sizes = {best['hidden']}")
    print(f"    alpha              = {best['alpha']}")
    print(f"    COR medio LOSO     = {best['cor_mean']:.4f} ± {best['cor_std']:.4f}")
    print(f"    RMSE medio LOSO    = {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}")

    # Ganancia sobre la config actual
    current = next(r for r in results if r["hidden"] == (32,) and r["alpha"] == 0.001)
    delta   = best["cor_mean"] - current["cor_mean"]
    print(f"\n  Ganancia sobre config actual: +{delta:.4f} COR")

    # Guardar resultados
    out_path = OUT_DIR / "busqueda_hiperparametros.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Búsqueda de hiperparámetros — módulo somnolencia v2\n")
        f.write("=" * 60 + "\n")
        f.write(f"Features: {X_all.shape[1]} (delta+asimetría={USE_DELTA_FEATURES})\n")
        f.write(f"Artefactos rechazados: thresh={ARTIFACT_ZSCORE_THRESH}\n")
        f.write(f"EMA: alpha={EMA_ALPHA}\n")
        f.write(f"Validación: LOSO ({len(np.unique(groups))} sujetos)\n\n")
        f.write(f"{'#':<3} {'Arquitectura':<22} {'alpha':<8} "
                f"{'COR medio':<14} {'COR std':<10} {'RMSE medio'}\n")
        f.write("-" * 75 + "\n")
        for rank, r in enumerate(results_sorted, start=1):
            marker = " *" if r["hidden"] == best["hidden"] and r["alpha"] == best["alpha"] else ""
            f.write(
                f"{rank:<3} {str(r['hidden']):<22} {r['alpha']:<8.0e} "
                f"{r['cor_mean']:<14.4f} {r['cor_std']:<10.4f} "
                f"{r['rmse_mean']:.4f}{marker}\n"
            )
        f.write(f"\nMejor config: hidden={best['hidden']}, alpha={best['alpha']}\n")
        f.write(f"COR medio = {best['cor_mean']:.4f} ± {best['cor_std']:.4f}\n")
        f.write(f"RMSE medio = {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}\n")

    print(f"\n  Resultados guardados en: {out_path.resolve()}")
    print("\n  → Copia los valores de MEJOR CONFIGURACIÓN en")
    print("    MLP_HIDDEN y MLP_ALPHA de evaluar_somnolencia_mejorado.py")
    print("    y vuelve a ejecutarlo para obtener las métricas y gráficas finales.\n")


if __name__ == "__main__":
    main()
