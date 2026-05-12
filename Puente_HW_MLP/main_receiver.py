import time
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

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

FS = 250
BUFFER_SECONDS = 8
BUFFER_SIZE = FS * BUFFER_SECONDS

STEP_SECONDS = 1
STEP_SAMPLES = FS * STEP_SECONDS

SMOOTH_WINDOWS = 3

RAW_CSV = "raw_signal_log.csv"
FEATURES_CSV = "features_log.csv"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


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


def extract_feature_vector(ch1: np.ndarray, ch2: np.ndarray, fs: int) -> np.ndarray:
    ch1_p = preprocess_channel(ch1, fs)
    ch2_p = preprocess_channel(ch2, fs)

    feats = []
    for _, band in BANDS.items():
        feats.append(differential_entropy_from_band(ch1_p, fs, band))
    for _, band in BANDS.items():
        feats.append(differential_entropy_from_band(ch2_p, fs, band))

    return np.array(feats, dtype=np.float32)


def main() -> None:
    print(f"[MAIN] Inicializando log de señal: {RAW_CSV}")
    print(f"[MAIN] Inicializando log de features: {FEATURES_CSV}")
    init_raw_csv(RAW_CSV)
    init_features_csv(FEATURES_CSV)

    buffer = RingBuffer2Ch(BUFFER_SIZE)
    smooth_queue = deque(maxlen=SMOOTH_WINDOWS)

    count = 0
    last_seq = None
    next_feature_at = BUFFER_SIZE

    source = SocketSource(HOST, PORT)

    print(f"[MAIN] Conectando a {HOST}:{PORT}...")
    source.connect()
    print("[MAIN] Conectado.")

    try:
        while True:
            packet = source.recv_exact(PACKET_SIZE)
            seq, ch1_adc, ch2_adc = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[MAIN] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq

            ch1 = adc_to_centered(ch1_adc)
            ch2 = adc_to_centered(ch2_adc)

            timestamp_now = time.time()

            append_raw_csv(
                RAW_CSV,
                timestamp_now,
                count,
                seq,
                ch1_adc,
                ch2_adc,
                ch1,
                ch2,
            )

            buffer.append(ch1, ch2)
            count += 1

            if count % FS == 0:
                x1, x2 = buffer.get_ordered()
                print(
                    f"[MAIN] muestras_totales={count} "
                    f"buffer_len={len(buffer)} full={buffer.is_full} "
                    f"ch1_std={np.std(x1):.2f} ch2_std={np.std(x2):.2f}"
                )

            if len(buffer) >= BUFFER_SIZE and count >= next_feature_at:
                x1, x2 = buffer.get_ordered()

                feat_vec = extract_feature_vector(x1, x2, FS)
                smooth_queue.append(feat_vec)
                feat_smooth = np.mean(np.stack(smooth_queue, axis=0), axis=0)

                append_features_csv(
                    FEATURES_CSV,
                    timestamp_now,
                    count,
                    seq,
                    feat_smooth,
                )

                print("\n[MAIN] Vector de 10 features (DE suavizada):")
                print(np.round(feat_smooth, 4))
                print(f"[MAIN] Guardado en CSV | muestra={count} | seq={seq}")

                next_feature_at += STEP_SAMPLES

    finally:
        source.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Detenido por usuario.")