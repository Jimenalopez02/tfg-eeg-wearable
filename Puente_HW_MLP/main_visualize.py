import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

from data_source import SocketSource
from parser_utils import parse_packet, adc_to_centered, PACKET_SIZE
from buffer_utils import RingBuffer2Ch

HOST = "127.0.0.1"
PORT = 50000

FS = 250
BUFFER_SECONDS = 8
BUFFER_SIZE = FS * BUFFER_SECONDS

STEP_SECONDS = 1
STEP_SAMPLES = FS * STEP_SECONDS

SMOOTH_WINDOWS = 3
PRED_SMOOTH_WINDOWS = 5
PLOT_EVERY_SAMPLES = 100
PRED_HISTORY = 30

PIPELINE_PATH = Path(r"..\SEED_VIG\resultados_finales\pipeline_mlp_2canales.pkl").resolve()

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FEATURE_NAMES = [
    "ch1_d", "ch1_t", "ch1_a", "ch1_b", "ch1_g",
    "ch2_d", "ch2_t", "ch2_a", "ch2_b", "ch2_g",
]

FEATURE_COLORS = ["tab:blue"] * 5 + ["tab:orange"] * 5


def classify_prediction(p: float) -> str:
    if p < 0.33:
        return "Baja"
    elif p < 0.66:
        return "Media"
    return "Alta"


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
    return float(0.5 * np.log(2 * np.pi * np.e * var))


def extract_feature_vector(ch1: np.ndarray, ch2: np.ndarray, fs: int) -> np.ndarray:
    ch1_p = preprocess_channel(ch1, fs)
    ch2_p = preprocess_channel(ch2, fs)

    feats = []
    for band in BANDS.values():
        feats.append(differential_entropy_from_band(ch1_p, fs, band))
    for band in BANDS.values():
        feats.append(differential_entropy_from_band(ch2_p, fs, band))

    return np.array(feats, dtype=np.float32)


def setup_figure():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    ax_sig, ax_feat, ax_pred = axes

    # Señal
    line_ch1, = ax_sig.plot([], [], label="Canal 1")
    line_ch2, = ax_sig.plot([], [], label="Canal 2")
    ax_sig.set_title("Señal en buffer (últimos 8 s)")
    ax_sig.set_xlabel("Tiempo (s)")
    ax_sig.set_ylabel("ADC centrado")
    ax_sig.legend()
    ax_sig.grid(True, alpha=0.3)

    # Features
    x_feat = np.arange(10)
    bars = ax_feat.bar(x_feat, np.zeros(10), color=FEATURE_COLORS)
    ax_feat.set_xticks(x_feat)
    ax_feat.set_xticklabels(FEATURE_NAMES, rotation=45)
    ax_feat.set_title("Último vector de 10 features")
    ax_feat.set_ylabel("Valor")
    ax_feat.grid(True, axis="y", alpha=0.3)

    # Predicción
    ax_pred.axhspan(0.00, 0.33, alpha=0.12, color="green")
    ax_pred.axhspan(0.33, 0.66, alpha=0.12, color="gold")
    ax_pred.axhspan(0.66, 1.00, alpha=0.12, color="red")

    line_clip, = ax_pred.plot([], [], label="Acotada", linewidth=1.5)
    line_smooth, = ax_pred.plot([], [], label="Suavizada", linewidth=2.5)

    pred_text = ax_pred.text(
        0.02,
        0.92,
        "Estado: -- | Valor: --",
        transform=ax_pred.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax_pred.set_title("Predicción online")
    ax_pred.set_xlabel("Ventanas")
    ax_pred.set_ylabel("Predicción")
    ax_pred.set_ylim(0.0, 1.0)
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax_sig, ax_feat, ax_pred, line_ch1, line_ch2, bars, line_clip, line_smooth, pred_text


def main() -> None:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"No encuentro el pipeline en: {PIPELINE_PATH}")

    print(f"[VIS] Cargando pipeline desde: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    print("[VIS] Pipeline cargado correctamente.")

    buffer = RingBuffer2Ch(BUFFER_SIZE)
    feat_queue = deque(maxlen=SMOOTH_WINDOWS)
    pred_queue = deque(maxlen=PRED_SMOOTH_WINDOWS)

    pred_hist_clip = deque(maxlen=PRED_HISTORY)
    pred_hist_smooth = deque(maxlen=PRED_HISTORY)

    count = 0
    last_seq = None
    next_feature_at = BUFFER_SIZE

    (
        fig,
        ax_sig,
        ax_feat,
        ax_pred,
        line_ch1,
        line_ch2,
        bars,
        line_clip,
        line_smooth,
        pred_text,
    ) = setup_figure()

    source = SocketSource(HOST, PORT)
    print(f"[VIS] Conectando a {HOST}:{PORT}...")
    source.connect()
    print("[VIS] Conectado.")

    try:
        while plt.fignum_exists(fig.number):
            packet = source.recv_exact(PACKET_SIZE)
            seq, ch1_adc, ch2_adc = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[VIS] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq

            ch1 = adc_to_centered(ch1_adc)
            ch2 = adc_to_centered(ch2_adc)

            buffer.append(ch1, ch2)
            count += 1

            if count % FS == 0:
                print(f"[VIS] muestras_totales={count} buffer_len={len(buffer)} full={buffer.is_full}")

            # Actualización de la señal
            if count % PLOT_EVERY_SAMPLES == 0:
                x1, x2 = buffer.get_ordered()
                n = len(x1)

                if n > 1:
                    t_axis = np.linspace(max(0, BUFFER_SECONDS - n / FS), BUFFER_SECONDS, n)
                    line_ch1.set_data(t_axis, x1)
                    line_ch2.set_data(t_axis, x2)

                    ax_sig.set_xlim(t_axis[0], t_axis[-1])

                    y_min = min(np.min(x1), np.min(x2))
                    y_max = max(np.max(x1), np.max(x2))
                    ax_sig.set_ylim(y_min - 10, y_max + 10)

            # Features + predicción
            if len(buffer) >= BUFFER_SIZE and count >= next_feature_at:
                print(f"[VIS] Entrando en cálculo de features en muestra {count}")

                x1, x2 = buffer.get_ordered()

                feat_vec = extract_feature_vector(x1, x2, FS)
                feat_queue.append(feat_vec)
                feat_smooth = np.mean(np.stack(feat_queue, axis=0), axis=0)

                X_live = feat_smooth.reshape(1, -1)
                y_pred_raw = float(pipeline.predict(X_live)[0])
                y_pred_clipped = float(np.clip(y_pred_raw, 0.0, 1.0))

                pred_queue.append(y_pred_clipped)
                y_pred_smooth = float(np.mean(pred_queue))

                print("\n[VIS] Vector de 10 features (DE suavizada):")
                print(np.round(feat_smooth, 4))
                print(f"[VIS] Predicción bruta    : {y_pred_raw:.4f}")
                print(f"[VIS] Predicción acotada  : {y_pred_clipped:.4f}")
                print(f"[VIS] Predicción suavizada: {y_pred_smooth:.4f}")

                # Actualizar barras
                for bar, val in zip(bars, feat_smooth):
                    bar.set_height(float(val))

                feat_min = float(np.min(feat_smooth))
                feat_max = float(np.max(feat_smooth))
                if abs(feat_max - feat_min) < 1e-6:
                    feat_max = feat_min + 1.0
                ax_feat.set_ylim(feat_min - 0.2, feat_max + 0.2)

                # Actualizar histórico de predicción
                pred_hist_clip.append(y_pred_clipped)
                pred_hist_smooth.append(y_pred_smooth)

                x_pred = np.arange(len(pred_hist_clip))
                line_clip.set_data(x_pred, list(pred_hist_clip))
                line_smooth.set_data(x_pred, list(pred_hist_smooth))

                ax_pred.set_xlim(0, max(10, len(pred_hist_clip)))
                ax_pred.set_ylim(0.0, 1.0)

                estado = classify_prediction(y_pred_smooth)
                pred_text.set_text(f"Estado: {estado} | Valor suavizado: {y_pred_smooth:.3f}")

                next_feature_at += STEP_SAMPLES

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

    finally:
        source.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[VIS] Detenido por usuario.")