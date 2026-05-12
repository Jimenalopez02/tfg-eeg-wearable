from pathlib import Path
import re
import json
import joblib
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path(".")
OUT_DIR = DATA_DIR / "resultados_concentracion_final"
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1) Detectar automáticamente todos los sujetos
# --------------------------------------------------
all_edf = sorted(DATA_DIR.glob("Subject*_*.edf"))
pattern = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)

subject_ids = set()
for f in all_edf:
    m = pattern.match(f.name)
    if m:
        subject_ids.add(m.group(1))

SUBJECTS = sorted(subject_ids, key=lambda x: int(x))

print("Sujetos detectados:", SUBJECTS)
print("Número total de sujetos:", len(SUBJECTS))

if len(SUBJECTS) == 0:
    raise FileNotFoundError("No se han detectado archivos SubjectXX_1.edf / SubjectXX_2.edf")

# --------------------------------------------------
# 2) Configuración fija del pipeline
# --------------------------------------------------
CHANNELS = ["EEG Fp1", "EEG Fp2"]
WINDOW_SEC = 4.0
STEP_SEC = 1.0
MOVING_K = 5

bands = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]

# --------------------------------------------------
# 3) Utilidades de señal
# --------------------------------------------------
def make_windows(data, sfreq, window_sec=4.0, step_sec=1.0):
    n_channels, n_samples = data.shape
    win = int(window_sec * sfreq)
    step = int(step_sec * sfreq)

    X = []
    for start in range(0, n_samples - win + 1, step):
        stop = start + win
        X.append(data[:, start:stop])

    return np.stack(X, axis=0)

def bandpass_filter(sig, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

def differential_entropy(sig):
    var = np.var(sig)
    var = max(var, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)

def extract_de_features(X_windows, sfreq):
    feats_all = []

    for i in range(X_windows.shape[0]):
        window = X_windows[i]   # (2, n_samples)
        feats = []

        for ch in range(window.shape[0]):
            sig = window[ch]

            for _, f1, f2 in bands:
                sig_band = bandpass_filter(sig, sfreq, f1, f2)
                de_val = differential_entropy(sig_band)
                feats.append(de_val)

        feats_all.append(feats)

    return np.array(feats_all, dtype=float)   # (n_windows, 10)

def moving_average_features(X_feat, k=5):
    X_smooth = np.zeros_like(X_feat)

    for i in range(X_feat.shape[0]):
        start = max(0, i - k + 1)
        X_smooth[i] = np.mean(X_feat[start:i+1], axis=0)

    return X_smooth

# --------------------------------------------------
# 4) Construcción del dataset completo
# --------------------------------------------------
def build_dataset():
    all_X = []
    all_y = []
    all_groups = []
    all_file_ids = []

    sfreq_ref = None

    for subj in SUBJECTS:
        for condition, label in [("1", 0), ("2", 1)]:
            fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"

            if not fpath.exists():
                print(f"Aviso: falta {fpath.name}, se omite.")
                continue

            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.pick(CHANNELS)

            sfreq = raw.info["sfreq"]
            if sfreq_ref is None:
                sfreq_ref = sfreq

            # nos quedamos con 60 s útiles
            raw.crop(tmin=1.0, tmax=61.0)

            data = raw.get_data()  # (2, n_samples)

            X_win = make_windows(
                data,
                sfreq=sfreq,
                window_sec=WINDOW_SEC,
                step_sec=STEP_SEC
            )

            X_feat = extract_de_features(X_win, sfreq)
            X_feat = moving_average_features(X_feat, k=MOVING_K)

            y_win = np.full(len(X_feat), label, dtype=int)
            g_win = np.full(len(X_feat), int(subj), dtype=int)
            file_id_win = np.array([f"Subject{subj}_{condition}"] * len(X_feat))

            all_X.append(X_feat)
            all_y.append(y_win)
            all_groups.append(g_win)
            all_file_ids.append(file_id_win)

    if len(all_X) == 0:
        raise RuntimeError("No se pudo construir el dataset.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    file_ids = np.concatenate(all_file_ids, axis=0)

    return X, y, groups, file_ids, sfreq_ref

# --------------------------------------------------
# 5) Modelo final elegido
# --------------------------------------------------
def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

# --------------------------------------------------
# 6) Evaluación Leave-One-Subject-Out
# --------------------------------------------------
print("\nConstruyendo dataset...")
X, y, groups, file_ids, sfreq = build_dataset()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sujetos únicos:", np.unique(groups))
print("Clase 0 (reposo):", np.sum(y == 0))
print("Clase 1 (cálculo mental):", np.sum(y == 1))

logo = LeaveOneGroupOut()

y_pred_all = np.zeros_like(y)
y_prob_all = np.zeros_like(y, dtype=float)

subject_results = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred_all[test_idx] = y_pred
    y_prob_all[test_idx] = y_prob

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    subj_test = int(np.unique(groups[test_idx])[0])

    subject_results.append({
        "subject": subj_test,
        "acc": float(acc),
        "f1": float(f1),
    })

    print(f"Fold {fold:02d} | sujeto test {subj_test:02d} | ACC={acc:.4f} | F1={f1:.4f}")

# --------------------------------------------------
# 7) Métricas globales
# --------------------------------------------------
acc_global = accuracy_score(y, y_pred_all)
f1_global = f1_score(y, y_pred_all)
cm = confusion_matrix(y, y_pred_all)

print("\n===== RESULTADO GLOBAL CONCENTRACIÓN =====")
print(f"ACC global = {acc_global:.4f}")
print(f"F1 global  = {f1_global:.4f}")
print("Matriz de confusión:")
print(cm)

# --------------------------------------------------
# 8) Guardar matriz de confusión
# --------------------------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Reposo", "Cálculo mental"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False)
plt.title("Matriz de confusión - Concentración")
plt.tight_layout()
plt.savefig(OUT_DIR / "matriz_confusion_concentracion.png", dpi=200)
plt.close()

# --------------------------------------------------
# 9) Guardar métricas por sujeto
# --------------------------------------------------
subject_results_sorted = sorted(subject_results, key=lambda d: d["subject"])

with open(OUT_DIR / "metricas_por_sujeto.txt", "w", encoding="utf-8") as f:
    for r in subject_results_sorted:
        f.write(f"Sujeto {r['subject']:02d}: ACC={r['acc']:.4f}, F1={r['f1']:.4f}\n")

with open(OUT_DIR / "metricas_globales.txt", "w", encoding="utf-8") as f:
    f.write("Modelo final de concentración\n")
    f.write("Canales: EEG Fp1, EEG Fp2\n")
    f.write("Ventana: 4 s\n")
    f.write("Paso: 1 s\n")
    f.write("Features: DE + moving average\n")
    f.write("Modelo: RandomForestClassifier(n_estimators=300, random_state=42)\n\n")
    f.write(f"ACC global: {acc_global:.4f}\n")
    f.write(f"F1 global: {f1_global:.4f}\n")
    f.write(f"Matriz de confusión:\n{cm}\n")

# --------------------------------------------------
# 10) Guardar predicciones out-of-fold
# --------------------------------------------------
np.savez(
    OUT_DIR / "predicciones_oof_concentracion.npz",
    y_true=y,
    y_pred=y_pred_all,
    y_prob=y_prob_all,
    groups=groups,
    file_ids=file_ids
)

# --------------------------------------------------
# 11) Entrenar modelo final con TODOS los sujetos
# --------------------------------------------------
final_model = make_model()
final_model.fit(X, y)

joblib.dump(final_model, OUT_DIR / "rf_concentracion_final.joblib")

config = {
    "channels": CHANNELS,
    "window_sec": WINDOW_SEC,
    "step_sec": STEP_SEC,
    "moving_k": MOVING_K,
    "bands": bands,
    "sfreq": sfreq,
    "features": "DE + moving average",
    "model": "RandomForestClassifier",
    "n_estimators": 300,
    "random_state": 42,
}

with open(OUT_DIR / "config_concentracion_final.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("\n===== MODELO FINAL GUARDADO =====")
print("Modelo:", OUT_DIR / "rf_concentracion_final.joblib")
print("Config:", OUT_DIR / "config_concentracion_final.json")
print("Resultados:", OUT_DIR.resolve())