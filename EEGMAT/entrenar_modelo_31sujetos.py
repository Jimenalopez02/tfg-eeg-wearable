"""
entrenar_modelo_31sujetos.py
============================
Re-entrena el Random Forest con los 31 sujetos (excluyendo los 5 holdout)
y guarda el modelo en EEGMAT/resultados_concentracion_final/rf_31sujetos_final.joblib

Ejecutar desde la carpeta Puente_HW_MLP:
    python entrenar_modelo_31sujetos.py
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from pathlib import Path
import mne
from scipy.signal import butter, sosfiltfilt

# ── Configuración ──────────────────────────────────────────────────────────────
EEGMAT_DIR  = Path("../EEGMAT").resolve()
OUTPUT_PATH = Path("../EEGMAT/resultados_concentracion_final/rf_31sujetos_final.joblib").resolve()

FS    = 500.0
BANDS = [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]

HOLDOUT    = [13, 16, 26, 30, 35]
ALL_SUBJ   = list(range(36))
TRAIN_SUBJ = [s for s in ALL_SUBJ if s not in HOLDOUT]

print(f"Sujetos entrenamiento ({len(TRAIN_SUBJ)}): {TRAIN_SUBJ}")
print(f"Sujetos holdout       ({len(HOLDOUT)}):    {HOLDOUT}\n")

# ── Funciones ──────────────────────────────────────────────────────────────────
def bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

def de(sig):
    v = max(np.var(sig), 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * v)

def moving_avg(X, k=5):
    out = np.zeros_like(X)
    for i in range(len(X)):
        s = max(0, i - k + 1)
        out[i] = X[s:i+1].mean(axis=0)
    return out

def extract_features_edf(edf_path, tmin=1.0, tmax=61.0):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(["EEG Fp1", "EEG Fp2"])
    raw.crop(tmin=tmin, tmax=tmax)
    data = raw.get_data()
    ch1, ch2 = data[0], data[1]
    win  = int(4.0 * FS)
    step = int(1.0 * FS)
    n    = min(len(ch1), len(ch2))
    feats = []
    for s in range(0, n - win + 1, step):
        w1, w2 = ch1[s:s+win], ch2[s:s+win]
        f = []
        for _, lo, hi in BANDS:
            f.append(de(bandpass(w1, FS, lo, hi)))
        for _, lo, hi in BANDS:
            f.append(de(bandpass(w2, FS, lo, hi)))
        feats.append(f)
    return np.array(feats, dtype=np.float32)

# ── Construir dataset de los 31 sujetos ───────────────────────────────────────
print("Extrayendo features de los 31 sujetos...")
all_X, all_y, all_groups = [], [], []

for idx in TRAIN_SUBJ:
    s = f"{idx:02d}"
    feats_rep  = moving_avg(extract_features_edf(EEGMAT_DIR / f"Subject{s}_1.edf"))
    feats_calc = moving_avg(extract_features_edf(EEGMAT_DIR / f"Subject{s}_2.edf"))
    n = min(len(feats_rep), len(feats_calc))
    all_X.append(feats_rep[:n])
    all_X.append(feats_calc[:n])
    all_y.extend([0]*n + [1]*n)
    all_groups.extend([idx]*(n*2))
    print(f"  Suj{idx:02d}: {n} ventanas reposo + {n} ventanas cálculo")

all_X      = np.vstack(all_X)
all_y      = np.array(all_y)
all_groups = np.array(all_groups)
print(f"\nDataset total: {len(all_y)} ventanas, {len(np.unique(all_groups))} sujetos")

# ── LOSO sobre los 31 ─────────────────────────────────────────────────────────
print("\nEjecutando LOSO sobre 31 sujetos...")
logo = LeaveOneGroupOut()
accs_loso, f1s_loso = [], []

for tr_idx, te_idx in logo.split(all_X, all_y, all_groups):
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(all_X[tr_idx], all_y[tr_idx])
    y_pred = rf.predict(all_X[te_idx])
    acc = accuracy_score(all_y[te_idx], y_pred)
    f1  = f1_score(all_y[te_idx], y_pred, zero_division=0)
    subj = np.unique(all_groups[te_idx])[0]
    accs_loso.append(acc)
    f1s_loso.append(f1)
    print(f"  Suj{subj:02d}: ACC={acc:.4f}  F1={f1:.4f}")

print(f"\nLOSO 31 sujetos → ACC={np.mean(accs_loso):.4f} ± {np.std(accs_loso):.4f}  "
      f"F1={np.mean(f1s_loso):.4f} ± {np.std(f1s_loso):.4f}")

# ── Modelo final entrenado con los 31 ─────────────────────────────────────────
print("\nEntrenando modelo final con los 31 sujetos completos...")
rf_final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_final.fit(all_X, all_y)
joblib.dump(rf_final, OUTPUT_PATH)
print(f"Modelo guardado en: {OUTPUT_PATH}")

# ── Evaluación batch sobre los 5 holdout ──────────────────────────────────────
print("\nEvaluando sobre los 5 sujetos holdout (nunca vistos)...")
accs_h, f1s_h = [], []

for idx in HOLDOUT:
    s = f"{idx:02d}"
    feats_rep  = moving_avg(extract_features_edf(EEGMAT_DIR / f"Subject{s}_1.edf"))
    feats_calc = moving_avg(extract_features_edf(EEGMAT_DIR / f"Subject{s}_2.edf"))
    n = min(len(feats_rep), len(feats_calc))
    X_h = np.vstack([feats_rep[:n], feats_calc[:n]])
    y_h = np.array([0]*n + [1]*n)
    y_pred = rf_final.predict(X_h)
    acc = accuracy_score(y_h, y_pred)
    f1  = f1_score(y_h, y_pred, zero_division=0)
    accs_h.append(acc)
    f1s_h.append(f1)
    print(f"  Suj{idx:02d}: ACC={acc:.4f}  F1={f1:.4f}")

print(f"\nHoldout 5 sujetos → ACC={np.mean(accs_h):.4f} ± {np.std(accs_h):.4f}  "
      f"F1={np.mean(f1s_h):.4f} ± {np.std(f1s_h):.4f}")

print("\nListo. Ahora ejecuta replay_holdout_5sujetos.py para la validación pseudo-online.")
