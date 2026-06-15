"""
buscar_hiperparametros_concentracion.py
=========================================
Búsqueda de hiperparámetros del RF sobre features extendidas (25 features).

Mejoras sobre el pipeline original:
  1. Features extendidas: delta temporal + asimetría frontal (10 → 25 features)
     - delta_DE : diferencia entre ventanas consecutivas (10 features)
     - asimetría: Fp1_banda - Fp2_banda por cada banda (5 features)
  2. Búsqueda de hiperparámetros del Random Forest:
     - n_estimators, max_depth, min_samples_split, max_features
  3. LOSO real (un fold por sujeto)
  4. Ranking final con comparación directa contra el baseline original

Al terminar indica la mejor configuración y la ganancia exacta sobre el original.
Guarda los resultados en resultados_concentracion_mejorado/busqueda_rf.txt

Uso:
  Coloca este script en la misma carpeta que los archivos SubjectXX_Y.edf
  python buscar_hiperparametros_concentracion.py

Tiempo estimado: ~15-25 min (24 configuraciones × 10 folds LOSO)
"""

from pathlib import Path
import re
import itertools
import numpy as np
import mne
import warnings
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURACIÓN
# ==============================================================
DATA_DIR = Path(".")
OUT_DIR  = Path("resultados_concentracion_mejorado")

CHANNELS   = ["EEG Fp1", "EEG Fp2"]
WINDOW_SEC = 4.0
STEP_SEC   = 1.0
MOVING_K   = 5

BANDS = [
    ("delta", 1,  4),
    ("theta", 4,  8),
    ("alpha", 8,  13),
    ("beta",  13, 30),
    ("gamma", 30, 45),
]

# Resultado original del TFG para comparación directa
BASELINE_ACC = 0.681
BASELINE_F1  = 0.67

# ==============================================================
# ESPACIO DE BÚSQUEDA
# ==============================================================
N_ESTIMATORS_OPTIONS   = [200, 300, 500]
MAX_DEPTH_OPTIONS      = [None, 10, 20]       # None = sin límite
MIN_SAMPLES_SPLIT_OPT  = [2, 5]
MAX_FEATURES_OPTIONS   = ["sqrt", 0.5]        # sqrt es el default de RF

CONFIGS = list(itertools.product(
    N_ESTIMATORS_OPTIONS,
    MAX_DEPTH_OPTIONS,
    MIN_SAMPLES_SPLIT_OPT,
    MAX_FEATURES_OPTIONS,
))


# ==============================================================
# UTILIDADES DE SEÑAL
# ==============================================================

def make_windows(data: np.ndarray, sfreq: float) -> np.ndarray:
    win  = int(WINDOW_SEC * sfreq)
    step = int(STEP_SEC   * sfreq)
    return np.stack(
        [data[:, s:s + win] for s in range(0, data.shape[1] - win + 1, step)],
        axis=0
    )

def bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

def differential_entropy(sig: np.ndarray) -> float:
    return 0.5 * np.log(2 * np.pi * np.e * max(np.var(sig), 1e-12))

def extract_de_features(X_windows: np.ndarray, sfreq: float) -> np.ndarray:
    feats = []
    for win in X_windows:
        row = []
        for ch in range(win.shape[0]):
            for _, f1, f2 in BANDS:
                row.append(differential_entropy(
                    bandpass_filter(win[ch], sfreq, f1, f2)
                ))
        feats.append(row)
    return np.array(feats, dtype=float)   # (N, 10)

def moving_average_features(X: np.ndarray, k: int = 5) -> np.ndarray:
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = np.mean(X[max(0, i - k + 1):i + 1], axis=0)
    return out

def add_delta_and_asymmetry(X: np.ndarray) -> np.ndarray:
    """
    Añade a los 10 features originales:
      - delta_DE (10): X[t] - X[t-1], captura dinámica temporal
      - asimetría (5): Fp1_banda - Fp2_banda por cada banda
    Total: 25 features.
    """
    delta = np.zeros_like(X)
    delta[1:] = X[1:] - X[:-1]

    # Fp1 tiene features 0-4, Fp2 tiene features 5-9
    asymmetry = X[:, 0:5] - X[:, 5:10]

    return np.hstack([X, delta, asymmetry])   # (N, 25)


# ==============================================================
# CONSTRUCCIÓN DEL DATASET
# ==============================================================

def detect_subjects() -> list:
    pattern = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)
    ids = {pattern.match(f.name).group(1)
           for f in DATA_DIR.glob("Subject*_*.edf")
           if pattern.match(f.name)}
    subjects = sorted(ids, key=int)
    if not subjects:
        raise FileNotFoundError("No se encontraron archivos SubjectXX_Y.edf")
    return subjects


def build_dataset_extended(subjects: list) -> tuple:
    """
    Construye el dataset con 25 features (original + delta + asimetría).
    El delta se calcula dentro de cada experimento para no mezclar sujetos.
    """
    X_list, y_list, g_list = [], [], []

    for subj in subjects:
        for condition, label in [("1", 0), ("2", 1)]:
            fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"
            if not fpath.exists():
                print(f"  Aviso: falta {fpath.name}, se omite.")
                continue

            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.pick(CHANNELS)
            sfreq = raw.info["sfreq"]
            raw.crop(tmin=1.0, tmax=61.0)
            data = raw.get_data()

            X_win  = make_windows(data, sfreq)
            X_feat = extract_de_features(X_win, sfreq)
            X_feat = moving_average_features(X_feat, k=MOVING_K)

            # Mejora: features extendidas (delta dentro del experimento)
            X_feat = add_delta_and_asymmetry(X_feat)

            X_list.append(X_feat)
            y_list.append(np.full(len(X_feat), label, dtype=int))
            g_list.append(np.full(len(X_feat), int(subj), dtype=int))

    X      = np.concatenate(X_list, axis=0)
    y      = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    return X, y, groups


# ==============================================================
# EVALUACIÓN DE UNA CONFIGURACIÓN
# ==============================================================

def evaluate_config(X, y, groups, n_est, max_depth, min_split, max_feat) -> dict:
    logo = LeaveOneGroupOut()

    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_split=min_split,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_test, y_prob))

    return {
        "n_est":      n_est,
        "max_depth":  max_depth,
        "min_split":  min_split,
        "max_feat":   max_feat,
        "acc_mean":   float(np.mean(accs)),
        "acc_std":    float(np.std(accs)),
        "f1_mean":    float(np.mean(f1s)),
        "f1_std":     float(np.std(f1s)),
        "auc_mean":   float(np.mean(aucs)),
        "auc_std":    float(np.std(aucs)),
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  BÚSQUEDA DE HIPERPARÁMETROS — MÓDULO CONCENTRACIÓN")
    print("=" * 70)
    print(f"\n  Features extendidas : 10 (original) → 25 (+ delta + asimetría)")
    print(f"  Configuraciones     : {len(CONFIGS)}")
    print(f"  Folds por config    : LOSO (un fold por sujeto)")
    print(f"  Baseline TFG        : ACC={BASELINE_ACC:.3f}  F1={BASELINE_F1:.2f}\n")

    # Dataset
    print("Detectando sujetos y construyendo dataset...")
    subjects = detect_subjects()
    print(f"  Sujetos: {subjects}  ({len(subjects)} en total)")

    X, y, groups = build_dataset_extended(subjects)
    n_subjects = len(np.unique(groups))
    print(f"  Shape X : {X.shape}  (25 features por ventana)")
    print(f"  Clase 0 : {(y==0).sum()}   Clase 1: {(y==1).sum()}\n")

    # Búsqueda
    results = []

    for i, (n_est, max_depth, min_split, max_feat) in enumerate(CONFIGS, start=1):
        tag = (f"n_est={n_est:<4} depth={'None' if max_depth is None else max_depth:<5} "
               f"min_split={min_split}  max_feat={max_feat}")
        print(f"[{i:2d}/{len(CONFIGS)}] {tag}")

        res = evaluate_config(X, y, groups, n_est, max_depth, min_split, max_feat)
        results.append(res)

        delta_f1  = res["f1_mean"]  - BASELINE_F1
        delta_acc = res["acc_mean"] - BASELINE_ACC
        sign_f1   = "+" if delta_f1  >= 0 else ""
        sign_acc  = "+" if delta_acc >= 0 else ""

        print(f"         ACC  {res['acc_mean']:.4f}±{res['acc_std']:.4f}  "
              f"({sign_acc}{delta_acc:.4f} vs TFG)  |  "
              f"F1  {res['f1_mean']:.4f}±{res['f1_std']:.4f}  "
              f"({sign_f1}{delta_f1:.4f} vs TFG)  |  "
              f"AUC {res['auc_mean']:.4f}")

    # Ranking
    results_sorted = sorted(results,
                             key=lambda r: (-r["f1_mean"], -r["acc_mean"]))

    print("\n" + "=" * 70)
    print("  RANKING FINAL")
    print("=" * 70)
    print(f"  {'#':<3} {'n_est':<6} {'depth':<7} {'min_sp':<8} {'max_ft':<8} "
          f"{'ACC medio':<20} {'F1 medio':<20} {'AUC medio'}")
    print("  " + "-" * 90)

    for rank, r in enumerate(results_sorted, start=1):
        marker = " ◄" if rank == 1 else ""
        print(
            f"  {rank:<3} {r['n_est']:<6} "
            f"{'None' if r['max_depth'] is None else r['max_depth']:<7} "
            f"{r['min_split']:<8} {str(r['max_feat']):<8} "
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}   "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}   "
            f"{r['auc_mean']:.4f}{marker}"
        )

    best = results_sorted[0]
    print(f"\n  MEJOR CONFIGURACIÓN:")
    print(f"    n_estimators     = {best['n_est']}")
    print(f"    max_depth        = {best['max_depth']}")
    print(f"    min_samples_split= {best['min_split']}")
    print(f"    max_features     = {best['max_feat']}")
    print(f"    ACC medio LOSO   = {best['acc_mean']:.4f} ± {best['acc_std']:.4f}")
    print(f"    F1  medio LOSO   = {best['f1_mean']:.4f}  ± {best['f1_std']:.4f}")
    print(f"    AUC medio LOSO   = {best['auc_mean']:.4f}  ± {best['auc_std']:.4f}")
    print(f"\n  Ganancia sobre TFG original:")
    print(f"    ACC : {best['acc_mean'] - BASELINE_ACC:+.4f}")
    print(f"    F1  : {best['f1_mean']  - BASELINE_F1:+.4f}")

    # Guardar resultados
    out_path = OUT_DIR / "busqueda_rf.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Búsqueda hiperparámetros RF — módulo concentración\n")
        f.write("=" * 60 + "\n")
        f.write(f"Features: 25 (delta+asimetría)\n")
        f.write(f"Sujetos: {n_subjects}  |  Validación: LOSO\n")
        f.write(f"Baseline TFG: ACC={BASELINE_ACC}  F1={BASELINE_F1}\n\n")
        f.write(f"{'#':<3} {'n_est':<6} {'depth':<7} {'min_sp':<7} {'max_ft':<8} "
                f"{'ACC':<14} {'F1':<14} {'AUC'}\n")
        f.write("-" * 72 + "\n")
        for rank, r in enumerate(results_sorted, start=1):
            marker = " *" if rank == 1 else ""
            f.write(
                f"{rank:<3} {r['n_est']:<6} "
                f"{'None' if r['max_depth'] is None else r['max_depth']:<7} "
                f"{r['min_split']:<7} {str(r['max_feat']):<8} "
                f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
                f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
                f"{r['auc_mean']:.4f}{marker}\n"
            )
        f.write(f"\nMejor: n_est={best['n_est']}, depth={best['max_depth']}, "
                f"min_split={best['min_split']}, max_feat={best['max_feat']}\n")
        f.write(f"ACC = {best['acc_mean']:.4f} ± {best['acc_std']:.4f}\n")
        f.write(f"F1  = {best['f1_mean']:.4f} ± {best['f1_std']:.4f}\n")
        f.write(f"AUC = {best['auc_mean']:.4f} ± {best['auc_std']:.4f}\n")
        f.write(f"\nGanancia ACC: {best['acc_mean'] - BASELINE_ACC:+.4f}\n")
        f.write(f"Ganancia F1 : {best['f1_mean']  - BASELINE_F1:+.4f}\n")

    print(f"\n  Resultados guardados en: {out_path.resolve()}")
    print("\n  → Copia los valores de MEJOR CONFIGURACIÓN en")
    print("    evaluar_concentracion_mejorado.py y vuelve a ejecutarlo.\n")


if __name__ == "__main__":
    main()
