import socket
import struct
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

HOST = "127.0.0.1"
PORT = 50000
PACKET_SIZE = 6
HEADER = 0xAA

FS = 250
BUFFER_SECONDS = 8
BUFFER_SIZE = FS * BUFFER_SECONDS

STEP_SECONDS = 1
STEP_SAMPLES = FS * STEP_SECONDS

SMOOTH_WINDOWS = 3  # moving average sobre 3 ventanas

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Conexión cerrada.")
        data += chunk
    return data


def parse_packet(packet: bytes) -> tuple[int, int, int]:
    h, seq, ch1_msb, ch1_lsb, ch2_msb, ch2_lsb = struct.unpack("!6B", packet)

    if h != HEADER:
        raise ValueError(f"Cabecera inválida: {h:#x}")

    ch1 = (ch1_msb << 8) | ch1_lsb
    ch2 = (ch2_msb << 8) | ch2_lsb
    return seq, ch1, ch2


class RingBuffer2Ch:
    def __init__(self, size: int):
        self.size = size
        self.ch1 = np.zeros(size, dtype=np.float32)
        self.ch2 = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.is_full = False

    def append(self, x1: float, x2: float) -> None:
        self.ch1[self.index] = x1
        self.ch2[self.index] = x2
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.is_full = True

    def get_ordered(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_full:
            return self.ch1[:self.index].copy(), self.ch2[:self.index].copy()

        ch1_ordered = np.concatenate((self.ch1[self.index:], self.ch1[:self.index]))
        ch2_ordered = np.concatenate((self.ch2[self.index:], self.ch2[:self.index]))
        return ch1_ordered, ch2_ordered

    def __len__(self) -> int:
        return self.size if self.is_full else self.index


def adc_to_centered(adc_value: int) -> float:
    return float(adc_value - 512)


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
    var = max(var, 1e-8)  # evitar log(0)

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
    print(f"[FEAT] Conectando a {HOST}:{PORT}...")

    buffer = RingBuffer2Ch(BUFFER_SIZE)
    smooth_queue = deque(maxlen=SMOOTH_WINDOWS)

    count = 0
    last_seq = None
    next_feature_at = BUFFER_SIZE

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        print("[FEAT] Conectado.")

        while True:
            packet = recv_exact(sock, PACKET_SIZE)
            seq, ch1_adc, ch2_adc = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[FEAT] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq

            ch1 = adc_to_centered(ch1_adc)
            ch2 = adc_to_centered(ch2_adc)

            buffer.append(ch1, ch2)
            count += 1

            if count % FS == 0:
                x1, x2 = buffer.get_ordered()
                print(
                    f"[FEAT] muestras_totales={count} buffer_len={len(buffer)} full={buffer.is_full} "
                    f"ch1_std={np.std(x1):.2f} ch2_std={np.std(x2):.2f}"
                )

            if len(buffer) >= BUFFER_SIZE and count >= next_feature_at:
                x1, x2 = buffer.get_ordered()

                feat_vec = extract_feature_vector(x1, x2, FS)
                smooth_queue.append(feat_vec)

                feat_smooth = np.mean(np.stack(smooth_queue, axis=0), axis=0)

                print("\n[FEAT] Vector de 10 features (DE suavizada):")
                print(np.round(feat_smooth, 4))

                next_feature_at += STEP_SAMPLES


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[FEAT] Detenido por usuario.")