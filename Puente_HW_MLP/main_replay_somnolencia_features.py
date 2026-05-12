import csv
import socket
import struct
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

HOST = "127.0.0.1"
PORT = 50040

PACK_FMT = "!10f"
PACKET_SIZE = struct.calcsize(PACK_FMT)

PIPELINE_PATH = Path(r"..\SEED_VIG\resultados_finales\pipeline_mlp_2canales.pkl").resolve()

PRED_CSV = "predictions_log_somnolencia_features.csv"


def init_csv(path, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def recv_exact(sock, nbytes):
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Conexión cerrada por el replay.")
        buf += chunk
    return buf


def parse_packet(packet: bytes):
    return np.array(struct.unpack(PACK_FMT, packet), dtype=np.float32)


def main():
    print("DEBUG: entrando en main()")

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"No encuentro el pipeline en: {PIPELINE_PATH}")

    print(f"[MAIN-SOMN-FEAT] Cargando pipeline: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)

    init_csv(
        PRED_CSV,
        [
            "idx",
            "timestamp",
            "prediction_raw",
            "prediction_clipped",
            "prediction_smooth",
        ],
    )

    pred_queue = deque(maxlen=5)

    idx = 0

    print(f"[MAIN-SOMN-FEAT] Conectando a {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        print("[MAIN-SOMN-FEAT] Conectado. Iniciando bucle...\n")

        try:
            while True:
                packet = recv_exact(sock, PACKET_SIZE)
                feat_vec = parse_packet(packet)

                X_live = feat_vec.reshape(1, -1)

                y_pred_raw = float(pipeline.predict(X_live)[0])
                y_pred_clipped = float(np.clip(y_pred_raw, 0.0, 1.0))

                pred_queue.append(y_pred_clipped)
                y_pred_smooth = float(np.mean(pred_queue))

                timestamp_now = time.time()

                append_csv(
                    PRED_CSV,
                    [
                        idx,
                        timestamp_now,
                        y_pred_raw,
                        y_pred_clipped,
                        y_pred_smooth,
                    ],
                )

                if idx % 2 == 0:
                    print("\n[MAIN-SOMN-FEAT] Vector de 10 features:")
                    print(np.round(feat_vec, 4))
                    print(f"[MAIN-SOMN-FEAT] Predicción bruta    : {y_pred_raw:.4f}")
                    print(f"[MAIN-SOMN-FEAT] Predicción acotada  : {y_pred_clipped:.4f}")
                    print(f"[MAIN-SOMN-FEAT] Predicción suavizada: {y_pred_smooth:.4f}")

                idx += 1

        except ConnectionError as e:
            print(f"\n[MAIN-SOMN-FEAT] Fin de conexión: {e}")

        except KeyboardInterrupt:
            print("\n[MAIN-SOMN-FEAT] Detenido por usuario.")


if __name__ == "__main__":
    print("DEBUG: archivo cargado")
    try:
        main()
    except Exception as e:
        print(f"[MAIN-SOMN-FEAT] ERROR: {type(e).__name__}: {e}")
        raise
    