import matplotlib
matplotlib.use("Agg")

# evaluar_concentracion_mejorado.py
# Pipeline LOSO de concentración con Random Forest sobre EEGMAT (36 sujetos).
# Reconstruye el dataset desde los EDF, evalúa RF/LR/MLP con LOSO real,
# optimiza el umbral via ROC y guarda modelo, métricas e importancia de features.


from pathlib import Path
import re
import json
import joblib
import numpy as np
import mne
import matplotlib.pyplot as plt
import warnings
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

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

# Nombres de features para la importancia (25 features)
_band_names = [b for b, _, _ in BANDS]
FEATURE_NAMES = (
    [f"{ch}_{b}" for ch in ["Fp1", "Fp2"] for b in _band_names] +
    [f"delta_{ch}_{b}" for ch in ["Fp1", "Fp2"] for b in _band_names] +
    [f"asim_{b}" for b in _band_names]
)


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
    var = max(np.var(sig), 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)

def extract_de_features(X_windows: np.ndarray, sfreq: float) -> np.ndarray:
    feats = []
    for win in X_windows:                          # (2, n_samples)
        row = []
        for ch in range(win.shape[0]):
            for _, f1, f2 in BANDS:
                row.append(differential_entropy(
                    bandpass_filter(win[ch], sfreq, f1, f2)
                ))
        feats.append(row)
    return np.array(feats, dtype=float)            # (n_windows, 10)

def moving_average_features(X: np.ndarray, k: int = 5) -> np.ndarray:
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = np.mean(X[max(0, i - k + 1):i + 1], axis=0)
    return out


def add_delta_and_asymmetry(X: np.ndarray) -> np.ndarray:
    """Extiende de 10 a 25 features: + delta temporal + asimetría frontal."""
    delta     = np.zeros_like(X)
    delta[1:] = X[1:] - X[:-1]
    asymmetry = X[:, 0:5] - X[:, 5:10]
    return np.hstack([X, delta, asymmetry])


# ==============================================================
# DETECCIÓN DE SUJETOS Y CONSTRUCCIÓN DEL DATASET
# ==============================================================

def detect_subjects() -> list:
    pattern = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)
    ids = {pattern.match(f.name).group(1)
           for f in DATA_DIR.glob("Subject*_*.edf")
           if pattern.match(f.name)}
    subjects = sorted(ids, key=int)
    if not subjects:
        raise FileNotFoundError(
            "No se encontraron archivos SubjectXX_1.edf / SubjectXX_2.edf"
        )
    return subjects


def build_dataset(subjects: list) -> tuple:
    X_list, y_list, g_list = [], [], []
    sfreq_ref = None

    for subj in subjects:
        for condition, label in [("1", 0), ("2", 1)]:
            fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"
            if not fpath.exists():
                print(f"  Aviso: falta {fpath.name}, se omite.")
                continue

            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.pick(CHANNELS)
            sfreq = raw.info["sfreq"]
            if sfreq_ref is None:
                sfreq_ref = sfreq

            raw.crop(tmin=1.0, tmax=61.0)
            data = raw.get_data()                  # (2, n_samples)

            X_win  = make_windows(data, sfreq)
            X_feat = extract_de_features(X_win, sfreq)
            X_feat = moving_average_features(X_feat, k=MOVING_K)
            X_feat = add_delta_and_asymmetry(X_feat)   # 10 → 25 features

            X_list.append(X_feat)
            y_list.append(np.full(len(X_feat), label, dtype=int))
            g_list.append(np.full(len(X_feat), int(subj), dtype=int))

    X      = np.concatenate(X_list, axis=0)
    y      = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    return X, y, groups, sfreq_ref


# ==============================================================
# MODELOS
# ==============================================================

def make_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu", solver="adam",
                alpha=0.001, max_iter=500, random_state=42,
            )),
        ]),
        # Baseline: clasificador mayoritario
        "Baseline_mayoritario": None,
    }


# ==============================================================
# EVALUACIÓN LOSO COMPLETA
# ==============================================================

def evaluate_loso(X, y, groups, models: dict) -> dict:
    logo    = LeaveOneGroupOut()
    results = {name: {"accs": [], "f1s": [], "probs": [], "trues": []}
               for name in models}

    n_subjects = len(np.unique(groups))
    print(f"\n  LOSO sobre {n_subjects} sujetos...\n")

    for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups=groups), start=1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subj = int(np.unique(groups[test_idx])[0])

        line = f"  Sujeto {subj:02d}/{n_subjects}"

        for name, model in models.items():

            # Baseline mayoritario
            if model is None:
                majority = int(np.bincount(y_train).argmax())
                y_pred   = np.full_like(y_test, majority)
                prob     = np.full(len(y_test), float(majority))
            else:
                import copy
                m = copy.deepcopy(model)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(X_test)[:, 1]
                elif hasattr(m, "named_steps"):
                    prob = m.named_steps["clf"].predict_proba(X_test)[:, 1]
                else:
                    prob = y_pred.astype(float)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, zero_division=0)

            results[name]["accs"].append(acc)
            results[name]["f1s"].append(f1)
            results[name]["probs"].extend(prob.tolist())
            results[name]["trues"].extend(y_test.tolist())

            line += f"  |  {name[:6]} ACC={acc:.3f} F1={f1:.3f}"

        print(line)

    # Agregar métricas
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "acc_mean":  float(np.mean(r["accs"])),
            "acc_std":   float(np.std(r["accs"])),
            "f1_mean":   float(np.mean(r["f1s"])),
            "f1_std":    float(np.std(r["f1s"])),
            "y_true":    np.array(r["trues"]),
            "y_prob":    np.array(r["probs"]),
        }
    return summary


# ==============================================================
# UMBRAL ÓPTIMO VIA ROC
# ==============================================================

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Umbral que maximiza F1 sobre las predicciones OOF."""
    thresholds = np.linspace(0, 1, 200)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


# ==============================================================
# GRÁFICAS
# ==============================================================

def plot_roc(summary: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in summary.items():
        if name == "Baseline_mayoritario":
            continue
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC — Módulo Concentración (LOSO OOF)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / "roc_concentracion.png", dpi=200)
    plt.close(fig)
    print("  Curva ROC guardada.")


def plot_confusion(y_true, y_pred, out_dir: Path, suffix: str = ""):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reposo", "Cálculo"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusión{' — ' + suffix if suffix else ''}")
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_{suffix.replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, names: list, out_dir: Path):
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), importances[idx], color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Importancia (Gini)")
    ax.set_title("Importancia de features — Random Forest concentración")
    plt.tight_layout()
    fig.savefig(out_dir / "importancia_features_concentracion.png", dpi=200)
    plt.close(fig)
    print("  Importancia de features guardada.")


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  MÓDULO CONCENTRACIÓN — VERSIÓN MEJORADA")
    print("=" * 65)

    # 1) Detectar sujetos
    subjects = detect_subjects()
    print(f"\n  Sujetos detectados: {subjects}  ({len(subjects)} en total)")

    # 2) Construir y guardar dataset completo
    print("\nConstruyendo dataset completo...")
    X, y, groups, sfreq = build_dataset(subjects)
    n_subjects = len(np.unique(groups))

    print(f"  Shape X      : {X.shape}")
    print(f"  Clase 0      : {(y==0).sum()}  Clase 1: {(y==1).sum()}")
    print(f"  NaNs/Infs    : {np.isnan(X).sum()} / {np.isinf(X).sum()}")

    # Guardar dataset reproducible completo
    npz_path = DATA_DIR / "dataset_concentracion_completo.npz"
    np.savez(npz_path, X_feat=X, y=y, groups=groups,
             channels=np.array(CHANNELS), sfreq=sfreq)
    print(f"  Dataset guardado: {npz_path.name}")

    # 3) Evaluación LOSO de todos los modelos
    models  = make_models()
    summary = evaluate_loso(X, y, groups, models)

    # 4) Umbral óptimo para RF
    rf_true = summary["RandomForest"]["y_true"]
    rf_prob = summary["RandomForest"]["y_prob"]
    opt_thr = find_optimal_threshold(rf_true, rf_prob)
    rf_pred_opt = (rf_prob >= opt_thr).astype(int)
    acc_opt = accuracy_score(rf_true, rf_pred_opt)
    f1_opt  = f1_score(rf_true, rf_pred_opt)

    print(f"\n  Umbral óptimo RF (max F1): {opt_thr:.3f}")
    print(f"  ACC con umbral óptimo    : {acc_opt:.4f}")
    print(f"  F1  con umbral óptimo    : {f1_opt:.4f}")
    print(f"  F1  con umbral 0.5       : {summary['RandomForest']['f1_mean']:.4f} (medio LOSO)")

    # 5) Ranking final
    print("\n" + "=" * 65)
    print("  RANKING FINAL")
    print("=" * 65)
    print(f"  {'Modelo':<25} {'ACC medio':<22} {'F1 medio'}")
    print("  " + "-" * 60)

    ranked = sorted(summary.items(),
                    key=lambda x: (-x[1]["f1_mean"], -x[1]["acc_mean"]))
    for name, res in ranked:
        marker = " ◄ mejor" if name == ranked[0][0] else ""
        print(f"  {name:<25} "
              f"{res['acc_mean']:.4f} ± {res['acc_std']:.4f}   "
              f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}{marker}")

    best_name = ranked[0][0]

    # 6) Gráficas
    print("\nGenerando gráficas...")
    plot_roc(summary, OUT_DIR)

    # Matriz de confusión RF umbral 0.5
    rf_pred_05 = (rf_prob >= 0.5).astype(int)
    plot_confusion(rf_true, rf_pred_05, OUT_DIR, "RF_umbral_0.5")

    # Matriz de confusión RF umbral óptimo
    plot_confusion(rf_true, rf_pred_opt, OUT_DIR, f"RF_umbral_{opt_thr:.2f}")

    # 7) Importancia de features (reentrenar RF con todos los datos)
    print("\nEntrenando RF final para importancia de features...")
    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf_final.fit(X, y)
    plot_feature_importance(rf_final.feature_importances_, FEATURE_NAMES, OUT_DIR)

    # Guardar importancias en txt
    imp_order = np.argsort(rf_final.feature_importances_)[::-1]
    with open(OUT_DIR / "importancia_features.txt", "w", encoding="utf-8") as f:
        f.write("Importancia de features — Random Forest concentración\n")
        f.write("=" * 45 + "\n")
        for i in imp_order:
            f.write(f"  {FEATURE_NAMES[i]:<15} {rf_final.feature_importances_[i]:.4f}\n")

    # 8) Guardar métricas globales
    with open(OUT_DIR / "metricas_concentracion.txt", "w", encoding="utf-8") as f:
        f.write("Módulo concentración — versión mejorada\n")
        f.write("=" * 50 + "\n")
        f.write(f"Canales    : {CHANNELS}\n")
        f.write(f"Ventana    : {WINDOW_SEC}s  Paso: {STEP_SEC}s\n")
        f.write(f"Features   : DE + moving average k={MOVING_K}  ({X.shape[1]} features)\n")
        f.write(f"Validación : LOSO ({n_subjects} sujetos)\n\n")

        f.write("RANKING DE MODELOS\n")
        f.write("-" * 50 + "\n")
        for name, res in ranked:
            f.write(f"{name:<25} "
                    f"ACC={res['acc_mean']:.4f}±{res['acc_std']:.4f}  "
                    f"F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}\n")

        f.write(f"\nUmbral óptimo RF : {opt_thr:.3f}\n")
        f.write(f"ACC umbral óptimo: {acc_opt:.4f}\n")
        f.write(f"F1  umbral óptimo: {f1_opt:.4f}\n")

        fpr, tpr, _ = roc_curve(rf_true, rf_prob)
        roc_auc = auc(fpr, tpr)
        f.write(f"AUC RF           : {roc_auc:.4f}\n")

    print(f"\n  Métricas guardadas en {OUT_DIR / 'metricas_concentracion.txt'}")

    # 9) Guardar modelo y config
    joblib.dump(rf_final, OUT_DIR / "rf_concentracion_final.joblib")
    config = {
        "channels": CHANNELS, "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC, "moving_k": MOVING_K,
        "bands": [[n, f1, f2] for n, f1, f2 in BANDS],
        "sfreq": sfreq, "n_subjects": n_subjects,
        "model": "RandomForestClassifier", "n_estimators": 300,
        "max_depth": None, "min_samples_split": 5, "max_features": "sqrt",
        "features_extended": True, "n_features": 25,
        "optimal_threshold": float(opt_thr),
    }
    with open(OUT_DIR / "config_concentracion.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Modelo guardado: rf_concentracion_final.joblib")
    print(f"  Config guardada: config_concentracion.json")
    print("\nProceso completado.\n")


if __name__ == "__main__":
    main()
