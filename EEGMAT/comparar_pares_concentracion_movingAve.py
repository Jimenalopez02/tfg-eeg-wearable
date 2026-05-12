from pathlib import Path
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(".")

# sujetos que vamos a usar
SUBJECTS = [f"{i:02d}" for i in range(1, 11)]

# pares de canales a comparar
CHANNEL_PAIRS = {
    "Fp1_Fp2": ["EEG Fp1", "EEG Fp2"],
    "F3_F4": ["EEG F3", "EEG F4"],
    "F7_F8": ["EEG F7", "EEG F8"],
}

# parámetros
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


def make_windows(data, sfreq, window_sec=4.0, step_sec=1.0):
    n_channels, n_samples = data.shape
    win = int(window_sec * sfreq)
    step = int(step_sec * sfreq)

    X = []
    starts = []

    for start in range(0, n_samples - win + 1, step):
        stop = start + win
        X.append(data[:, start:stop])
        starts.append(start)

    return np.stack(X, axis=0), np.array(starts)


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


def build_dataset_for_pair(channels):
    all_X = []
    all_y = []
    all_groups = []

    for subj in SUBJECTS:
        for condition, label in [("1", 0), ("2", 1)]:
            fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"

            if not fpath.exists():
                raise FileNotFoundError(f"No encuentro {fpath}")

            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.pick(channels)

            sfreq = raw.info["sfreq"]

            # nos quedamos con 60 s útiles
            raw.crop(tmin=1.0, tmax=61.0)

            data = raw.get_data()  # (2, n_samples)

            X_win, _ = make_windows(
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

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    return X, y, groups


def evaluate_pair(X, y, groups):
    logo = LeaveOneGroupOut()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

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


results = []

for pair_name, channels in CHANNEL_PAIRS.items():
    print(f"\n===== Evaluando par {pair_name} =====")
    print("Canales:", channels)

    X, y, groups = build_dataset_for_pair(channels)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Sujetos:", np.unique(groups))
    print("Clase 0:", np.sum(y == 0))
    print("Clase 1:", np.sum(y == 1))

    res = evaluate_pair(X, y, groups)

    results.append({
        "pair": pair_name,
        **res
    })

    print(f"ACC medio = {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")
    print(f"F1 medio  = {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")

results_sorted = sorted(results, key=lambda r: (-r["f1_mean"], -r["acc_mean"]))

print("\n===== RANKING FINAL PARES CONCENTRACIÓN =====")
for i, r in enumerate(results_sorted, start=1):
    print(
        f"{i}. {r['pair']}: "
        f"ACC={r['acc_mean']:.4f} ± {r['acc_std']:.4f}, "
        f"F1={r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
    )