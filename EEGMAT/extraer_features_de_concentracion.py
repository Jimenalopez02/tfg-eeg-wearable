from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

DATA_DIR = Path(".")
in_path = DATA_DIR / "dataset_concentracion_base.npz"

if not in_path.exists():
    raise FileNotFoundError(f"No encuentro {in_path}")

data = np.load(in_path, allow_pickle=True)
X = data["X"]          # (n_windows, 2, 2000)
y = data["y"]
groups = data["groups"]
channels = data["channels"]
sfreq = float(data["sfreq"])

print("X crudo shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("Canales:", channels)
print("Fs:", sfreq)

bands = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]

def bandpass_filter(sig, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

def differential_entropy(sig):
    var = np.var(sig)
    var = max(var, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)

X_feat = []

for i in range(X.shape[0]):
    window = X[i]   # (2, 2000)
    feats = []

    for ch in range(window.shape[0]):
        sig = window[ch]

        for _, f1, f2 in bands:
            sig_band = bandpass_filter(sig, sfreq, f1, f2)
            de_val = differential_entropy(sig_band)
            feats.append(de_val)

    X_feat.append(feats)

X_feat = np.array(X_feat, dtype=float)

print("\n===== RESUMEN FEATURES =====")
print("X_feat shape:", X_feat.shape)
print("Número de features por ventana:", X_feat.shape[1])

out_path = DATA_DIR / "dataset_concentracion_de.npz"
np.savez(
    out_path,
    X_feat=X_feat,
    y=y,
    groups=groups,
    channels=channels,
    sfreq=sfreq
)

print("Guardado en:", out_path.resolve())