from pathlib import Path
import mne

DATA_DIR = Path(r"C:\Users\jimen\OneDrive\Desktop\9\UPM\TFG\EEGMAT")

files = sorted(DATA_DIR.glob("Subject01_*.edf"))

if not files:
    raise FileNotFoundError(f"No encuentro archivos EDF en {DATA_DIR}")

for f in files:
    print("\n==============================")
    print("Archivo:", f.name)

    raw = mne.io.read_raw_edf(f, preload=False, verbose=False)

    print("Canales:", raw.ch_names)
    print("Frecuencia de muestreo:", raw.info["sfreq"])
    print("Número de canales:", len(raw.ch_names))
    print("Duración (s):", raw.n_times / raw.info["sfreq"])