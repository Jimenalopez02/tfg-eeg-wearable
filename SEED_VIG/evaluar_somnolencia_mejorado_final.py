"""
evaluar_somnolencia_mejorado.py  — v2

Uso:
  python evaluar_somnolencia_mejorado.py  — v2

Requisitos:
  pip install numpy scipy scikit-learn matplotlib joblib
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================
# CONFIGURACIÓN — edita aquí si cambian tus rutas o hiperparámetros
# ==============================================================
FEATURE_DIR = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
LABEL_DIR   = Path(r"perclos_labels\perclos_labels")
OUT_DIR     = Path("resultados_mejorados")

FEATURE_KEY  = "de_movingAve"
CHANNEL_PAIR = (1, 2)        # par de canales frontales seleccionado

# Hiperparámetros del MLP (los tuyos, probados como mejor config)
MLP_HIDDEN   = (32,)
MLP_ALPHA    = 0.001

# Mejoras activables individualmente (True/False)
USE_SUBJECT_NORMALIZATION = False  # DESACTIVADO: en sujetos con PERCLOS poco variable
                                   # la desnormalización amplifica el error y satura en [0,1]
USE_DELTA_FEATURES        = False  # desactivado: mantiene 10 features del TFG
USE_ARTIFACT_REJECTION    = False  # desactivado: no altera los datos originales
USE_EMA_POSTPROCESS       = True   # mejora 4: suavizado exponencial
EMA_ALPHA                 = 0.2    # más suave que 0.3 → menos ruido en predicción
ARTIFACT_ZSCORE_THRESH    = 3.5    # ligeramente más agresivo que 4.0


# ==============================================================
# UTILIDADES
# ==============================================================

def corrcoef_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ema_smooth(y: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Suavizado exponencial (EMA). alpha pequeño → más suave."""
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def reject_artifact_windows(X: np.ndarray, y: np.ndarray,
                              thresh: float = 4.0):
    """
    Elimina ventanas cuya norma L2 de features supere thresh desviaciones estándar.
    Devuelve X_clean, y_clean y la máscara booleana de ventanas válidas.
    """
    norms = np.linalg.norm(X, axis=1)
    z = (norms - np.mean(norms)) / (np.std(norms) + 1e-12)
    mask = np.abs(z) <= thresh
    return X[mask], y[mask], mask


# ==============================================================
# CARGA Y CONSTRUCCIÓN DEL DATASET
# ==============================================================

def load_raw_experiment(f_path: Path, l_path: Path,
                         ch_a: int, ch_b: int) -> tuple:
    """Devuelve X (N,10) e y (N,) para un experimento."""
    mat_f = sio.loadmat(f_path)
    mat_l = sio.loadmat(l_path)

    X = mat_f[FEATURE_KEY]          # (4, 885, 5)  canal × muestra × banda
    y = mat_l["perclos"].reshape(-1)  # (885,)

    X = X[[ch_a, ch_b], :, :]          # (2, 885, 5)
    X = np.transpose(X, (1, 0, 2))     # (885, 2, 5)
    X = X.reshape(X.shape[0], -1)      # (885, 10)

    return X, y


def add_delta_and_asymmetry_features(X: np.ndarray) -> np.ndarray:
    """
    Añade a cada ventana t:
      - delta_DE: diferencia X[t] - X[t-1]  (10 features)
      - asimetría frontal: X[t, 0:5] - X[t, 5:10]  (5 features por banda)
    Total: 10 (original) + 10 (delta) + 5 (asimetría) = 25 features.

    La primera ventana usa delta=0.
    Las 5 features de asimetría son: canal_A - canal_B por cada banda.
    """
    N = X.shape[0]  # número de muestras

    # Delta temporal
    delta = np.zeros_like(X)
    delta[1:] = X[1:] - X[:-1]

    # Asimetría frontal (canal 0 vs canal 1, cada uno con 5 bandas)
    ch_a_feats = X[:, 0:5]   # DE canal A por banda
    ch_b_feats = X[:, 5:10]  # DE canal B por banda
    asymmetry = ch_a_feats - ch_b_feats  # (N, 5)

    return np.hstack([X, delta, asymmetry])  # (N, 25)


def build_full_dataset(ch_a: int, ch_b: int):
    """
    Construye X_all, y_all, groups con todas las mejoras activadas.
    Aplica mejoras por experimento (dentro del bucle) para evitar fuga de datos.
    """
    feature_files = sorted(FEATURE_DIR.glob("*.mat"))
    label_files   = sorted(LABEL_DIR.glob("*.mat"))

    if not feature_files:
        raise FileNotFoundError(f"No hay features en: {FEATURE_DIR.resolve()}")
    if not label_files:
        raise FileNotFoundError(f"No hay labels en: {LABEL_DIR.resolve()}")

    label_map = {f.name: f for f in label_files}

    X_list, y_list, groups_list = [], [], []
    n_rejected_total = 0
    n_total = 0

    for exp_id, f_path in enumerate(feature_files):
        if f_path.name not in label_map:
            raise ValueError(f"Sin label para {f_path.name}")

        X, y = load_raw_experiment(f_path, label_map[f_path.name], ch_a, ch_b)
        n_total += len(y)

        # Mejora 1: normalización z-score del target por sujeto
        # (solo desplaza la distribución; al evaluar se compara en escala original
        #  dentro de cada fold de test, por eso se desnormaliza en evaluate_loso)
        if USE_SUBJECT_NORMALIZATION:
            y_mean = np.mean(y)
            y_std  = np.std(y) + 1e-12
        else:
            y_mean, y_std = 0.0, 1.0

        y_norm = (y - y_mean) / y_std

        # Mejora 2: features extendidas (delta + asimetría)
        if USE_DELTA_FEATURES:
            X = add_delta_and_asymmetry_features(X)

        # Mejora 3: rechazo de artefactos
        if USE_ARTIFACT_REJECTION:
            X, y_norm, mask = reject_artifact_windows(
                X, y_norm, thresh=ARTIFACT_ZSCORE_THRESH
            )
            n_rejected = np.sum(~mask)
            n_rejected_total += n_rejected

        X_list.append(X)
        y_list.append(y_norm)
        groups_list.extend([exp_id] * len(y_norm))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups = np.array(groups_list)

    if USE_ARTIFACT_REJECTION:
        pct = 100 * n_rejected_total / n_total
        print(f"  Artefactos rechazados: {n_rejected_total}/{n_total} ({pct:.1f}%)")

    return X_all, y_all, groups, feature_files, label_map


# ==============================================================
# MODELO
# ==============================================================

def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN,
            activation="relu",
            solver="adam",
            alpha=MLP_ALPHA,
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ))
    ])


# ==============================================================
# EVALUACIÓN LOSO
# ==============================================================

def evaluate_loso(X_all, y_all, groups, feature_files, label_map,
                  ch_a, ch_b, use_baseline=True):
    """
    Leave-One-Subject-Out.
    Si USE_SUBJECT_NORMALIZATION=True, el modelo predice en espacio normalizado
    y se desnormaliza antes de calcular métricas, para que COR y RMSE sean
    comparables con los resultados del TFG.
    """
    logo = LeaveOneGroupOut()
    subject_ids = np.unique(groups)
    n_subjects  = len(subject_ids)

    fold_cors_model    = []
    fold_rmses_model   = []
    fold_cors_baseline = []
    fold_rmses_baseline = []

    y_pred_all  = np.full_like(y_all, np.nan)
    y_denorm_all = np.zeros_like(y_all)  # y en escala original para métricas globales

    print(f"\n  LOSO sobre {n_subjects} sujetos...\n")

    # Pre-calcular estadísticos de normalización por sujeto
    # (se recuperan del dataset original para desnormalizar)
    subject_stats = {}
    flist = sorted(FEATURE_DIR.glob("*.mat"))
    lmap  = {f.name: f for f in sorted(LABEL_DIR.glob("*.mat"))}

    for exp_id, f_path in enumerate(flist):
        mat_l = sio.loadmat(lmap[f_path.name])
        y_raw = mat_l["perclos"].reshape(-1)
        subject_stats[exp_id] = (np.mean(y_raw), np.std(y_raw) + 1e-12)

    for fold_idx, (train_idx, test_idx) in enumerate(
            logo.split(X_all, y_all, groups=groups)):

        test_subject = groups[test_idx[0]]

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # Modelo
        model = make_model()
        model.fit(X_train, y_train)
        y_pred_norm = model.predict(X_test)

        # Mejora 4: EMA en espacio normalizado antes de desnormalizar
        if USE_EMA_POSTPROCESS:
            y_pred_norm = ema_smooth(y_pred_norm, alpha=EMA_ALPHA)

        # Desnormalizar para métricas en escala PERCLOS original [0,1]
        if USE_SUBJECT_NORMALIZATION:
            mu, sigma = subject_stats[test_subject]
            y_test_orig = y_test  * sigma + mu
            y_pred_orig = np.clip(y_pred_norm * sigma + mu, 0, 1)
        else:
            y_test_orig = y_test
            y_pred_orig = np.clip(y_pred_norm, 0, 1)

        y_pred_all[test_idx]  = y_pred_orig
        y_denorm_all[test_idx] = y_test_orig

        cor  = corrcoef_safe(y_test_orig, y_pred_orig)
        rmse = rmse_score(y_test_orig, y_pred_orig)
        fold_cors_model.append(cor)
        fold_rmses_model.append(rmse)

        # Baseline: predicción de la media de train (desnormalizada)
        if use_baseline:
            if USE_SUBJECT_NORMALIZATION:
                mu_tr, sig_tr = np.mean(y_train), np.std(y_train) + 1e-12
                y_baseline = np.full_like(y_test_orig,
                             np.mean(y_train) * sigma + mu)
            else:
                y_baseline = np.full_like(y_test_orig, np.mean(y_train))
            y_baseline = np.clip(y_baseline, 0, 1)
            cor_bl  = corrcoef_safe(y_test_orig, y_baseline)
            rmse_bl = rmse_score(y_test_orig, y_baseline)
            fold_cors_baseline.append(cor_bl)
            fold_rmses_baseline.append(rmse_bl)

        print(
            f"  Sujeto {test_subject+1:2d}/{n_subjects}: "
            f"COR={cor:.4f}  RMSE={rmse:.4f}"
            + (f"  |  Baseline COR={cor_bl:.4f}" if use_baseline else "")
        )

    # Métricas globales
    valid = ~np.isnan(y_pred_all)
    cor_global  = corrcoef_safe(y_denorm_all[valid], y_pred_all[valid])
    rmse_global = rmse_score(y_denorm_all[valid], y_pred_all[valid])

    cors  = np.array(fold_cors_model)
    rmses = np.array(fold_rmses_model)

    return {
        "cor_mean":  float(np.nanmean(cors)),
        "cor_std":   float(np.nanstd(cors)),
        "cor_min":   float(np.nanmin(cors)),
        "cor_max":   float(np.nanmax(cors)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std":  float(np.std(rmses)),
        "cor_global":  cor_global,
        "rmse_global": rmse_global,
        "y_true": y_denorm_all[valid],
        "y_pred": y_pred_all[valid],
        "baseline_cor_mean":  float(np.nanmean(fold_cors_baseline)) if use_baseline else None,
        "baseline_rmse_mean": float(np.mean(fold_rmses_baseline))   if use_baseline else None,
    }


# ==============================================================
# GRÁFICAS
# ==============================================================

def plot_results(res: dict, out_dir: Path):
    y_true = res["y_true"]
    y_pred = res["y_pred"]

    # Dispersión real vs predicho
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color="steelblue")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("PERCLOS real")
    ax.set_ylabel("PERCLOS predicho")
    ax.set_title(
        f"Dispersión LOSO\nCOR={res['cor_global']:.4f}  RMSE={res['rmse_global']:.4f}"
    )
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    fig.savefig(out_dir / "dispersion_mejorado.png", dpi=200)
    plt.close(fig)

    # Histograma de errores
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (predicho − real)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de errores — modelo mejorado")
    plt.tight_layout()
    fig.savefig(out_dir / "histograma_errores_mejorado.png", dpi=200)
    plt.close(fig)

    print(f"  Gráficas guardadas en {out_dir.resolve()}")


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    ch_a, ch_b = CHANNEL_PAIR

    print("=" * 60)
    print("  MÓDULO SOMNOLENCIA — VERSIÓN MEJORADA")
    print("=" * 60)
    print(f"\nMejoras activas:")
    print(f"  1. Normalización por sujeto : {USE_SUBJECT_NORMALIZATION}")
    print(f"  2. Features delta+asimetría : {USE_DELTA_FEATURES}")
    print(f"  3. Rechazo de artefactos    : {USE_ARTIFACT_REJECTION} (thresh z={ARTIFACT_ZSCORE_THRESH})")
    print(f"  4. EMA postprocesado        : {USE_EMA_POSTPROCESS} (alpha={EMA_ALPHA})")
    print(f"  Protocolo validación        : LOSO (Leave-One-Subject-Out)")

    # ------- Dataset -------
    print("\nCargando dataset...")
    X_all, y_all, groups, feature_files, label_map = build_full_dataset(ch_a, ch_b)
    n_subjects = len(np.unique(groups))
    print(f"  Shape X: {X_all.shape}  |  Shape y: {y_all.shape}  |  Sujetos: {n_subjects}")

    # ------- Evaluación LOSO -------
    res = evaluate_loso(X_all, y_all, groups, feature_files, label_map,
                        ch_a, ch_b, use_baseline=True)

    # ------- Resumen -------
    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES")
    print("=" * 60)
    print(f"  COR  medio LOSO : {res['cor_mean']:.4f} ± {res['cor_std']:.4f}")
    print(f"  COR  rango      : [{res['cor_min']:.4f}, {res['cor_max']:.4f}]")
    print(f"  RMSE medio LOSO : {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print(f"  COR  global     : {res['cor_global']:.4f}")
    print(f"  RMSE global     : {res['rmse_global']:.4f}")
    if res["baseline_cor_mean"] is not None:
        print(f"\n  Baseline (media de entrenamiento):")
        print(f"    COR  medio LOSO : {res['baseline_cor_mean']:.4f}")
        print(f"    RMSE medio LOSO : {res['baseline_rmse_mean']:.4f}")
        delta_cor = res['cor_mean'] - res['baseline_cor_mean']
        print(f"    Ganancia modelo : +{delta_cor:.4f} COR sobre baseline")
    print("=" * 60)

    # ------- Gráficas -------
    print("\nGenerando gráficas...")
    plot_results(res, OUT_DIR)

    # ------- Guardar métricas txt -------
    metrics_path = OUT_DIR / "metricas_mejoradas.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Módulo somnolencia — versión mejorada\n")
        f.write("=" * 50 + "\n")
        f.write(f"Feature key  : {FEATURE_KEY}\n")
        f.write(f"Canales      : {CHANNEL_PAIR}\n")
        f.write(f"MLP hidden   : {MLP_HIDDEN}  alpha={MLP_ALPHA}\n")
        f.write(f"Normalización por sujeto : {USE_SUBJECT_NORMALIZATION}\n")
        f.write(f"Delta + asimetría        : {USE_DELTA_FEATURES}\n")
        f.write(f"Rechazo artefactos       : {USE_ARTIFACT_REJECTION} thresh={ARTIFACT_ZSCORE_THRESH}\n")
        f.write(f"EMA postprocesado        : {USE_EMA_POSTPROCESS} alpha={EMA_ALPHA}\n")
        f.write(f"Validación               : LOSO ({n_subjects} sujetos)\n\n")
        f.write(f"COR  medio  = {res['cor_mean']:.4f} ± {res['cor_std']:.4f}\n")
        f.write(f"COR  rango  = [{res['cor_min']:.4f}, {res['cor_max']:.4f}]\n")
        f.write(f"RMSE medio  = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}\n")
        f.write(f"COR  global = {res['cor_global']:.4f}\n")
        f.write(f"RMSE global = {res['rmse_global']:.4f}\n")
        if res["baseline_cor_mean"] is not None:
            f.write(f"\nBaseline COR  = {res['baseline_cor_mean']:.4f}\n")
            f.write(f"Baseline RMSE = {res['baseline_rmse_mean']:.4f}\n")
    print(f"  Métricas guardadas: {metrics_path.resolve()}")

    # ------- Entrenamiento final con todos los datos -------
    print("\nEntrenando modelo final con todos los datos...")
    final_model = make_model()
    final_model.fit(X_all, y_all)
    model_path = OUT_DIR / "pipeline_somnolencia_mejorado.pkl"
    joblib.dump(final_model, model_path)
    print(f"  Modelo guardado: {model_path.resolve()}")

    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
