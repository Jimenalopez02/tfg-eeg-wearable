import argparse
import csv
import socket
import struct
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HOST = "127.0.0.1"
PORT = 50040

FS_FEAT = 2.0   # las features están a 2 Hz
DT = 1.0 / FS_FEAT

FEATURE_DIR = Path(r"..\SEED_VIG\Forehead_EEG\Forehead_EEG\EEG_Feature_5Bands").resolve()
LABEL_DIR = Path(r"..\SEED_VIG\perclos_labels\perclos_labels").resolve()

# Mejor par de canales obtenido en tus experimentos
CH_A = 1
CH_B = 2

PACK_FMT = "!10f"
PACKET_SIZE = struct.calcsize(PACK_FMT)

GT_CSV = "replay_groundtruth_somnolencia.csv"


def load_feature_sequence(file_name: str):
    feat_path = FEATURE_DIR / file_name
    label_path = LABEL_DIR / file_name

    if not feat_path.exists():
        raise FileNotFoundError(f"No encuentro features: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"No encuentro labels: {label_path}")

    feat_mat = loadmat(feat_path)
    label_mat = loadmat(label_path)

    if "de_movingAve" not in feat_mat:
        raise KeyError(f"No encuentro 'de_movingAve' en {feat_path}")
    if "perclos" not in label_mat:
        raise KeyError(f"No encuentro 'perclos' en {label_path}")

    X = feat_mat["de_movingAve"]   # shape esperada: (4, T, 5)
    y = label_mat["perclos"].reshape(-1)  # shape: (T,)

    if X.ndim != 3:
        raise ValueError(f"Esperaba X con 3 dimensiones y recibí {X.shape}")

    if CH_A >= X.shape[0] or CH_B >= X.shape[0]:
        raise ValueError(f"Par de canales fuera de rango para shape {X.shape}")

    # Selección del mejor par y concatenación de 5 bandas + 5 bandas = 10 features
    Xa = X[CH_A, :, :]   # (T, 5)
    Xb = X[CH_B, :, :]   # (T, 5)

    feat_seq = np.concatenate([Xa, Xb], axis=1).astype(np.float32)  # (T, 10)

    T = min(len(feat_seq), len(y))
    feat_seq = feat_seq[:T]
    y = y[:T].astype(np.float32)

    return feat_seq, y


def pack_features(feat_vec: np.ndarray) -> bytes:
    if feat_vec.shape[0] != 10:
        raise ValueError(f"Esperaba 10 features y recibí {feat_vec.shape}")
    return struct.pack(PACK_FMT, *feat_vec.tolist())


def write_groundtruth_csv(y: np.ndarray):
    with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "time_sec", "perclos_true"])
        for i, val in enumerate(y):
            writer.writerow([i, i / FS_FEAT, float(val)])


def main():
    parser = argparse.ArgumentParser(description="Replay temporal de features reales SEED-VIG")
    parser.add_argument("--file", required=True, help="Ejemplo: 1_20151124_noon_2.mat")
    args = parser.parse_args()

    feat_seq, y = load_feature_sequence(args.file)
    write_groundtruth_csv(y)

    print(f"[REPLAY-SOMN] Archivo: {args.file}")
    print(f"[REPLAY-SOMN] Features shape: {feat_seq.shape}")
    print(f"[REPLAY-SOMN] Labels shape  : {y.shape}")
    print(f"[REPLAY-SOMN] Esperando conexión en {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            print(f"[REPLAY-SOMN] Cliente conectado desde {addr}")

            t0 = time.perf_counter()

            for i, feat_vec in enumerate(feat_seq):
                target_time = t0 + i * DT
                now = time.perf_counter()
                sleep_time = target_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                packet = pack_features(feat_vec)
                conn.sendall(packet)

                if i % int(FS_FEAT) == 0:
                    print(
                        f"[REPLAY-SOMN] t={i / FS_FEAT:7.2f}s "
                        f"perclos={y[i]:.4f} "
                        f"feat0={feat_vec[0]:.4f} feat9={feat_vec[-1]:.4f}"
                    )

    print("\n[REPLAY-SOMN] Replay terminado.")
    print(f"[REPLAY-SOMN] Ground truth guardado en: {Path(GT_CSV).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[REPLAY-SOMN] Detenido por usuario.")