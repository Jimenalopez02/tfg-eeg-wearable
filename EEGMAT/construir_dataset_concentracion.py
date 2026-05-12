from pathlib import Path
import numpy as np
import mne

DATA_DIR = Path(".")

# sujetos pequeños para empezar hoy
SUBJECTS = ["01", "02", "03"]

# canales frontales elegidos
CHANNELS = ["EEG Fp1", "EEG Fp2"]

# parámetros de ventanas
WINDOW_SEC = 4.0
STEP_SEC = 1.0

all_X = []
all_y = []
all_groups = []

def make_windows(data, sfreq, window_sec=4.0, step_sec=1.0):
    """
    data shape: (n_channels, n_samples)
    devuelve:
        X shape: (n_windows, n_channels, n_window_samples)
        starts: índices de inicio
    """
    n_channels, n_samples = data.shape
    win = int(window_sec * sfreq)
    step = int(step_sec * sfreq)

    X = []
    starts = []

    for start in range(0, n_samples - win + 1, step):
        stop = start + win
        X.append(data[:, start:stop])
        starts.append(start)

    X = np.stack(X, axis=0)
    starts = np.array(starts)
    return X, starts


for subj in SUBJECTS:
    for condition, label in [("1", 0), ("2", 1)]:
        fpath = DATA_DIR / f"Subject{subj}_{condition}.edf"

        if not fpath.exists():
            raise FileNotFoundError(f"No encuentro {fpath}")

        print(f"\nCargando: {fpath.name}")
        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)

        # seleccionar solo Fp1 y Fp2
        raw.pick(CHANNELS)

        sfreq = raw.info["sfreq"]
        print("Fs:", sfreq)
        print("Canales:", raw.ch_names)

        # recorte opcional: quedarse con 60 s centrales/lógicos
        # como la duración reportada es ~62 s, usamos de 1 a 61 s
        raw.crop(tmin=1.0, tmax=61.0)

        data = raw.get_data()   # shape: (2, n_samples)
        print("Shape señal:", data.shape)

        X_win, starts = make_windows(
            data,
            sfreq=sfreq,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC
        )

        y_win = np.full(len(X_win), label, dtype=int)
        g_win = np.full(len(X_win), int(subj), dtype=int)

        print("Ventanas generadas:", len(X_win), "Etiqueta:", label)

        all_X.append(X_win)
        all_y.append(y_win)
        all_groups.append(g_win)

X = np.concatenate(all_X, axis=0)   # (n_windows, 2, n_samples_window)
y = np.concatenate(all_y, axis=0)   # (n_windows,)
groups = np.concatenate(all_groups, axis=0)

print("\n===== RESUMEN FINAL =====")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("Clase 0 (reposo):", np.sum(y == 0))
print("Clase 1 (cálculo mental):", np.sum(y == 1))

# guardar un primer dataset base
out_path = DATA_DIR / "dataset_concentracion_base.npz"
np.savez(out_path, X=X, y=y, groups=groups, channels=np.array(CHANNELS), sfreq=sfreq)

print("\nGuardado en:", out_path.resolve())