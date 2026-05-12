from pathlib import Path
import scipy.io as sio

# Rutas correctas desde la carpeta SEED_VIG
feature_dir = Path(r"Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands")
label_dir = Path(r"perclos_labels\perclos_labels")

# Listar archivos
feature_files = sorted(feature_dir.glob("*.mat"))
label_files = sorted(label_dir.glob("*.mat"))

print("Nº archivos feature:", len(feature_files))
print("Nº archivos label:", len(label_files))

if not feature_files:
    raise FileNotFoundError(f"No encuentro .mat en: {feature_dir.resolve()}")

if not label_files:
    raise FileNotFoundError(f"No encuentro .mat en: {label_dir.resolve()}")

# Elegimos el primero
f_path = feature_files[0]
l_path = label_files[0]

print("\nArchivo feature:", f_path.name)
print("Archivo label:", l_path.name)

# Cargar .mat de features
mat_f = sio.loadmat(f_path)
print("\nClaves en feature:")
for k, v in mat_f.items():
    if not k.startswith("__"):
        print(k, "shape=", getattr(v, "shape", None), "dtype=", getattr(v, "dtype", type(v)))

# Cargar .mat de labels
mat_l = sio.loadmat(l_path)
print("\nClaves en label:")
for k, v in mat_l.items():
    if not k.startswith("__"):
        print(k, "shape=", getattr(v, "shape", None), "dtype=", getattr(v, "dtype", type(v)))