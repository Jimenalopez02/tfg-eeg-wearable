import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

from data_source import SocketSource
from parser_utils import parse_packet, adc_to_centered, PACKET_SIZE
from buffer_utils import RingBuffer2Ch
from logger_utils import (
    init_raw_csv,
    append_raw_csv,
    init_features_csv,
    append_features_csv,
)

HOST = "127.0.0.1"
PORT = 50000

# Fs del emulador / hardware conceptual
FS_IN = 250

# Rutas del modelo final de concentración
MODEL_DIR = Path(r"..\EEGMAT\resultados_concentracion_final").resolve()
MODEL_PATH = MODEL_DIR / "rf_concentracion_final.joblib"
CONFIG_PATH = MODEL_DIR / "config_concentracion_final.json"

RAW_CSV = "raw_signal_log_concentracion.csv"
FEATURES_CSV = "features_log_concentracion.csv"
PREDICTIONS_CSV = "predictions_log_concentracion.csv"

CLASS_NAMES = {
    0: "Reposo",
    1: "Cálculo mental",
}


def init_predictions_csv(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return

    header = [
        "timestamp",
        "sample_count",
        "seq",
        "prob_reposo",
        "prob_calculo_mental",
        "pred_class",
        "pred_class_smooth",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_predictions_csv(
    csv_path: str,
    timestamp: float,
    sample_count: int,
    seq: int,
    prob_reposo: float,
    prob_calculo_mental: float,
    pred_class: int,
    pred_class_smooth: int,
) -> None:
    row = [
        timestamp,
        sample_count,
        seq,
        prob_reposo,
        prob_calculo_mental,
        pred_class,
        pred_class_smooth,
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def apply_notch(signal: np.ndarray, fs: int, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, signal)


def apply_bandpass(signal: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def preprocess_channel(x: np.ndarray, fs: int) -> np.ndarray:
    x = x - np.mean(x)
    x = apply_notch(x, fs, f0=50.0, q=30.0)
    x = apply_bandpass(x, fs, low=0.5, high=45.0, order=4)
    return x


def differential_entropy_from_band(x: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    low, high = band
    xb = apply_bandpass(x, fs, low=low, high=high, order=4)

    var = np.var(xb)
    var = max(var, 1e-8)

    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return float(de)


def extract_feature_vector(ch1: np.ndarray, ch2: np.ndarray, fs: int, bands_dict: dict[str, tuple[float, float]]) -> np.ndarray:
    ch1_p = preprocess_channel(ch1, fs)
    ch2_p = preprocess_channel(ch2, fs)

    feats = []
    for _, band in bands_dict.items():
        feats.append(differential_entropy_from_band(ch1_p, fs, band))
    for _, band in bands_dict.items():
        feats.append(differential_entropy_from_band(ch2_p, fs, band))

    return np.array(feats, dtype=np.float32)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No encuentro el modelo en: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No encuentro la config en: {CONFIG_PATH}")

    print(f"[CONC] Cargando modelo desde: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fs_target = int(cfg["sfreq"])          # 500 Hz del entrenamiento EEGMAT
    window_sec = float(cfg["window_sec"])  # 4 s
    step_sec = float(cfg["step_sec"])      # 1 s
    moving_k = int(cfg["moving_k"])        # 5

    bands_list = cfg["bands"]
    bands_dict = {name: (low, high) for name, low, high in bands_list}

    buffer_seconds = max(8, int(window_sec) + 2)
    buffer_size = FS_IN * buffer_seconds
    step_samples = int(FS_IN * step_sec)

    print("[CONC] Modelo cargado correctamente.")
    print("[CONC] Inicializando logs...")
    init_raw_csv(RAW_CSV)
    init_features_csv(FEATURES_CSV)
    init_predictions_csv(PREDICTIONS_CSV)

    buffer = RingBuffer2Ch(buffer_size)
    feature_queue = deque(maxlen=moving_k)
    pred_queue = deque(maxlen=5)

    count = 0
    last_seq = None
    next_feature_at = int(window_sec * FS_IN)

    source = SocketSource(HOST, PORT)

    print(f"[CONC] Conectando a {HOST}:{PORT}...")
    source.connect()
    print("[CONC] Conectado al emulador.")

    try:
        while True:
            packet = source.recv_exact(PACKET_SIZE)
            seq, ch1_adc, ch2_adc = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[CONC] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq

            ch1 = adc_to_centered(ch1_adc)
            ch2 = adc_to_centered(ch2_adc)

            timestamp_now = time.time()

            append_raw_csv(
                RAW_CSV,
                timestamp_now,
                count + 1,
                seq,
                ch1_adc,
                ch2_adc,
                ch1,
                ch2,
            )

            buffer.append(ch1, ch2)
            count += 1

            if count % FS_IN == 0:
                x1_dbg, x2_dbg = buffer.get_ordered()
                print(
                    f"[CONC] muestras_totales={count} "
                    f"buffer_len={len(buffer)} full={buffer.is_full} "
                    f"ch1_std={np.std(x1_dbg):.2f} ch2_std={np.std(x2_dbg):.2f}"
                )

            # cuando ya hay al menos una ventana útil y toca inferencia
            if len(buffer) >= int(window_sec * FS_IN) and count >= next_feature_at:
                x1_all, x2_all = buffer.get_ordered()

                win_len = int(window_sec * FS_IN)
                ch1_win = x1_all[-win_len:]
                ch2_win = x2_all[-win_len:]

                # remuestreo a 500 Hz para acercarnos al dominio del entrenamiento EEGMAT
                target_len = int(round(len(ch1_win) * fs_target / FS_IN))
                ch1_rs = resample(ch1_win, target_len)
                ch2_rs = resample(ch2_win, target_len)

                feat_vec = extract_feature_vector(ch1_rs, ch2_rs, fs_target, bands_dict)

                # moving average de features
                feature_queue.append(feat_vec)
                feat_smooth = np.mean(np.stack(feature_queue, axis=0), axis=0)

                append_features_csv(
                    FEATURES_CSV,
                    timestamp_now,
                    count,
                    seq,
                    feat_smooth,
                )

                X_live = feat_smooth.reshape(1, -1)

                probas = model.predict_proba(X_live)[0]
                pred_class = int(np.argmax(probas))

                pred_queue.append(pred_class)

                # voto mayoritario simple sobre las últimas predicciones
                values, counts = np.unique(np.array(pred_queue), return_counts=True)
                pred_class_smooth = int(values[np.argmax(counts)])

                append_predictions_csv(
                    PREDICTIONS_CSV,
                    timestamp_now,
                    count,
                    seq,
                    float(probas[0]),
                    float(probas[1]),
                    pred_class,
                    pred_class_smooth,
                )

                print("\n[CONC] Vector de 10 features (DE suavizada):")
                print(np.round(feat_smooth, 4))
                print(
                    f"[CONC] Predicción actual     : {CLASS_NAMES[pred_class]} "
                    f"(p_reposo={probas[0]:.3f}, p_calculo={probas[1]:.3f})"
                )
                print(
                    f"[CONC] Predicción suavizada  : {CLASS_NAMES[pred_class_smooth]}"
                )
                print(f"[CONC] Guardado en CSV | muestra={count} | seq={seq}")

                next_feature_at += step_samples

    finally:
        source.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CONC] Detenido por usuario.")