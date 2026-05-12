from pathlib import Path
import re
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(".")

# -----------------------------
# 1) Detectar automáticamente todos los sujetos
# -----------------------------
all_edf = sorted(DATA_DIR.glob("Subject*_*.edf"))

subject_ids = set()
pattern = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)

for f in all_edf:
    m = pattern.match(f.name)
    if m:
        subject_ids.add(m.group(1))

SUBJECTS = sorted(subject_ids, key=lambda x: int(x))

print("Sujetos detectados:", SUBJECTS)
print("Número total de sujetos:", len(SUBJECTS))

if len(SUBJECTS) == 0:
    raise FileNotFoundError("No se han detectado archivos SubjectXX_1.edf / SubjectXX_2.edf")

# -----------------------------
# 2) Configuración fija
# -----------------------------
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

# -----------------------------
# 3) Utilidades
# -----------------------------
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

# -----------------------------
# 4) Construcción del dataset completo
# -----------------------------
def build_dataset():
    all_X = []
    all_y = []
    all_groups = []

    for subj in SUBJECTS:
        for condition, label in [("1", 0), ("2", 1)]:
            fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"

            if not fpath.exists():
                print(f"Aviso: falta {fpath.name}, se omite ese archivo.")
                continue

            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.pick(CHANNELS)

            sfreq = raw.info["sfreq"]

            # Usamos 60 s útiles
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

            all_X.append(X_feat)
            all_y.append(y_win)
            all_groups.append(g_win)

    if len(all_X) == 0:
        raise RuntimeError("No se ha podido construir ningún bloque de datos.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    return X, y, groups

# -----------------------------
# 5) Evaluación
# -----------------------------
def evaluate_model(model, X, y, groups):
    logo = LeaveOneGroupOut()

    accs = []
    f1s = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accs.append(acc)
        f1s.append(f1)

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }

# -----------------------------
# 6) Construcción del dataset
# -----------------------------
print("\nConstruyendo dataset completo de concentración...")
X, y, groups = build_dataset()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sujetos únicos:", np.unique(groups))
print("Clase 0 (reposo):", np.sum(y == 0))
print("Clase 1 (cálculo mental):", np.sum(y == 1))

# -----------------------------
# 7) Modelos a comparar
# -----------------------------
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            random_state=42
        ))
    ]),
}

results = []

for name, model in models.items():
    print(f"\n===== Evaluando {name} =====")
    res = evaluate_model(model, X, y, groups)

    results.append({
        "model": name,
        **res
    })

    print(f"ACC medio = {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")
    print(f"F1 medio  = {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")

results_sorted = sorted(results, key=lambda r: (-r["f1_mean"], -r["acc_mean"]))

print("\n===== RANKING FINAL MODELOS CONCENTRACIÓN (TODOS LOS SUJETOS) =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. {r['model']}: "
        f"ACC={r['acc_mean']:.4f} ± {r['acc_std']:.4f}, "
        f"F1={r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
    )
    