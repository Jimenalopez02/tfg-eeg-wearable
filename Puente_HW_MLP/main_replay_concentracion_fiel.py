import csv
import json
import socket
import struct
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt

HOST = "127.0.0.1"
PORT = 50030
PACKET_SIZE = 8  # 2 float32
FS = 500.0

MODEL_DIR = Path(r"..\EEGMAT\resultados_concentracion_final").resolve()
MODEL_PATH = MODEL_DIR / "rf_concentracion_final.joblib"
CONFIG_PATH = MODEL_DIR / "config_concentracion_final.json"

RAW_CSV = "raw_signal_log_concentracion_fiel.csv"
FEATURES_CSV = "features_log_concentracion_fiel.csv"
PRED_CSV = "predictions_log_concentracion_fiel.csv"

CLASS_NAMES = {
    0: "Reposo",
    1: "Cálculo mental",
}


def init_csv(path, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_packet_float(packet: bytes):
    return struct.unpack("!2f", packet)


def bandpass_filter(sig, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def differential_entropy(sig):
    var = np.var(sig)
    var = max(var, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def extract_feature_vector_exact(ch1, ch2, fs, bands):
    feats = []

    for _, f1, f2 in bands:
        sig_band = bandpass_filter(ch1, fs, f1, f2)
        feats.append(differential_entropy(sig_band))

    for _, f1, f2 in bands:
        sig_band = bandpass_filter(ch2, fs, f1, f2)
        feats.append(differential_entropy(sig_band))

    return np.array(feats, dtype=np.float32)


def recv_exact(sock, nbytes):
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Conexión cerrada por el replay.")
        buf += chunk
    return buf


def main():
    print("DEBUG: entrando en main()")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No encuentro el modelo en: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No encuentro la config en: {CONFIG_PATH}")

    print(f"[MAIN-FIEL] Modelo: {MODEL_PATH}")
    print(f"[MAIN-FIEL] Config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = joblib.load(MODEL_PATH)

    bands = cfg["bands"]
    window_sec = float(cfg["window_sec"])   # 4 s
    step_sec = float(cfg["step_sec"])       # 1 s
    moving_k = int(cfg["moving_k"])         # 5

    win_samples = int(window_sec * FS)
    step_samples = int(step_sec * FS)

    print(f"[MAIN-FIEL] FS={FS}")
    print(f"[MAIN-FIEL] win_samples={win_samples}")
    print(f"[MAIN-FIEL] step_samples={step_samples}")
    print(f"[MAIN-FIEL] moving_k={moving_k}")

    init_csv(RAW_CSV, ["timestamp", "sample_count", "ch1", "ch2"])
    init_csv(FEATURES_CSV, ["timestamp", "sample_count"] + [f"f{i}" for i in range(10)])
    init_csv(
        PRED_CSV,
        [
            "timestamp",
            "sample_count",
            "prob_reposo",
            "prob_calculo_mental",
            "pred_class",
            "pred_class_smooth",
        ],
    )

    buf1 = deque(maxlen=win_samples)
    buf2 = deque(maxlen=win_samples)
    feat_queue = deque(maxlen=moving_k)
    pred_queue = deque(maxlen=5)

    count = 0
    next_pred_at = win_samples

    print(f"[MAIN-FIEL] Conectando a {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        print("[MAIN-FIEL] Conectado. Iniciando bucle...\n")

        try:
            while True:
                packet = recv_exact(sock, PACKET_SIZE)
                ch1, ch2 = parse_packet_float(packet)

                timestamp_now = time.time()
                append_csv(RAW_CSV, [timestamp_now, count + 1, ch1, ch2])

                buf1.append(ch1)
                buf2.append(ch2)
                count += 1

                if count % int(FS) == 0:
                    print(
                        f"[MAIN-FIEL] muestras_totales={count} "
                        f"buf_len={len(buf1)} "
                        f"std1={np.std(np.array(buf1)):.4f} "
                        f"std2={np.std(np.array(buf2)):.4f}"
                    )

                if len(buf1) == win_samples and count >= next_pred_at:
                    ch1_win = np.array(buf1, dtype=float)
                    ch2_win = np.array(buf2, dtype=float)

                    feat_vec = extract_feature_vector_exact(ch1_win, ch2_win, FS, bands)
                    feat_queue.append(feat_vec)
                    feat_smooth = np.mean(np.stack(feat_queue, axis=0), axis=0)

                    append_csv(FEATURES_CSV, [timestamp_now, count] + feat_smooth.tolist())

                    X_live = feat_smooth.reshape(1, -1)
                    probas = model.predict_proba(X_live)[0]
                    pred_class = int(np.argmax(probas))

                    pred_queue.append(pred_class)
                    values, counts = np.unique(np.array(pred_queue), return_counts=True)
                    pred_class_smooth = int(values[np.argmax(counts)])

                    append_csv(
                        PRED_CSV,
                        [
                            timestamp_now,
                            count,
                            float(probas[0]),
                            float(probas[1]),
                            pred_class,
                            pred_class_smooth,
                        ],
                    )

                    print("\n[MAIN-FIEL] Vector de 10 features:")
                    print(np.round(feat_smooth, 4))
                    print(
                        f"[MAIN-FIEL] Predicción actual   : {CLASS_NAMES[pred_class]} "
                        f"(p_reposo={probas[0]:.3f}, p_calculo={probas[1]:.3f})"
                    )
                    print(
                        f"[MAIN-FIEL] Predicción suavizada: {CLASS_NAMES[pred_class_smooth]}"
                    )

                    next_pred_at += step_samples

        except ConnectionError as e:
            print(f"\n[MAIN-FIEL] Fin de conexión: {e}")

        except KeyboardInterrupt:
            print("\n[MAIN-FIEL] Detenido por usuario.")


if __name__ == "__main__":
    print("DEBUG: archivo cargado")
    try:
        main()
    except Exception as e:
        print(f"[MAIN-FIEL] ERROR: {type(e).__name__}: {e}")
        raise