import argparse
import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, resample

from data_source import SocketSource
from parser_utils import parse_packet, adc_to_centered, PACKET_SIZE
from buffer_utils import RingBuffer2Ch
from logger_utils import (
    init_raw_csv,
    append_raw_csv,
    init_features_csv,
    append_features_csv,
)

# =========================================================
# CONFIGURACIÓN BASE
# =========================================================

HOST = "127.0.0.1"
PORT = 50031
FS_IN = 250  # frecuencia del emulador / hardware conceptual

BASE_DIR = Path(__file__).resolve().parent

BANDS_COMMON = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# =========================================================
# UTILIDADES COMUNES
# =========================================================

def init_predictions_csv(csv_path: str, mode: str) -> None:
    if os.path.exists(csv_path):
        return

    if mode == "somnolencia":
        header = [
            "timestamp",
            "sample_count",
            "seq",
            "prediction_raw",
            "prediction_clipped",
            "prediction_smooth",
        ]
    else:
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


def append_predictions_csv(csv_path: str, row: list) -> None:
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def apply_notch(signal: np.ndarray, fs: int, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
   
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal)


def apply_bandpass(signal: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def preprocess_channel(x: np.ndarray, fs: int) -> np.ndarray:
    x = x - np.mean(x)
    x = apply_notch(x, fs, f0=50.0, q=30.0)
    x = apply_bandpass(x, fs, low=0.5, high=45.0, order=4)
    return x


def differential_entropy_from_band(x: np.ndarray, fs: int, band: tuple) -> float:
    low, high = band
    xb = apply_bandpass(x, fs, low=low, high=high, order=4)
    var = max(np.var(xb), 1e-8)
    return float(0.5 * np.log(2 * np.pi * np.e * var))


def extract_feature_vector(ch1: np.ndarray, ch2: np.ndarray, fs: int, bands_dict: dict) -> np.ndarray:
    ch1_p = preprocess_channel(ch1, fs)
    ch2_p = preprocess_channel(ch2, fs)

    feats = []
    for _, band in bands_dict.items():
        feats.append(differential_entropy_from_band(ch1_p, fs, band))
    for _, band in bands_dict.items():
        feats.append(differential_entropy_from_band(ch2_p, fs, band))

    return np.array(feats, dtype=np.float32)


# =========================================================
# CONFIGURACIÓN DE MODOS
# =========================================================

def load_mode_config(mode: str) -> dict:
    if mode == "somnolencia":
        pipeline_path = Path(r"..\SEED_VIG\resultados_finales\pipeline_mlp_2canales.pkl").resolve()

        if not pipeline_path.exists():
            raise FileNotFoundError(f"No encuentro el pipeline de somnolencia en: {pipeline_path}")

        return {
            "mode": "somnolencia",
            "model_path": pipeline_path,
            "config_path": None,
            "bands": BANDS_COMMON,
            "fs_target": FS_IN,
            "window_sec": 8.0,
            "step_sec": 1.0,
            "feature_smooth_windows": 3,
            "prediction_smooth_windows": 5,
            "raw_csv": "raw_signal_log_somnolencia.csv",
            "features_csv": "features_log_somnolencia.csv",
            "predictions_csv": "predictions_log_somnolencia.csv",
        }

    elif mode == "concentracion":
        model_dir = Path(r"..\EEGMAT\resultados_concentracion_final").resolve()
        model_path = model_dir / "rf_31sujetos_final.joblib"
        config_path = model_dir / "config_concentracion_final.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No encuentro el modelo de concentración en: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"No encuentro la config de concentración en: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        bands = {name: (low, high) for name, low, high in cfg_json["bands"]}

        return {
            "mode": "concentracion",
            "model_path": model_path,
            "config_path": config_path,
            "bands": bands,
            "fs_target": int(cfg_json["sfreq"]),
            "window_sec": float(cfg_json["window_sec"]),
            "step_sec": float(cfg_json["step_sec"]),
            "feature_smooth_windows": int(cfg_json["moving_k"]),
            "prediction_smooth_windows": 5,
            "raw_csv": "raw_signal_log_concentracion.csv",
            "features_csv": "features_log_concentracion.csv",
            "predictions_csv": "predictions_log_concentracion.csv",
        }

    else:
        raise ValueError(f"Modo no soportado: {mode}")


# =========================================================
# FUENTE DE DATOS
# =========================================================

def build_source(source: str):
    if source == "emulador":
        s = SocketSource(HOST, PORT)
        s.connect()
        return s
    elif source == "bluno":
        raise NotImplementedError("La fuente 'bluno' la dejamos preparada para el siguiente paso.")
    else:
        raise ValueError(f"Fuente no soportada: {source}")


# =========================================================
# POSTPROCESADO POR MODO
# =========================================================

def postprocess_somnolencia(model, X_live, pred_queue):
    y_pred_raw = float(model.predict(X_live)[0])
    y_pred_clipped = float(np.clip(y_pred_raw, 0.0, 1.0))

    pred_queue.append(y_pred_clipped)
    y_pred_smooth = float(np.mean(pred_queue))

    return {
        "raw": y_pred_raw,
        "clipped": y_pred_clipped,
        "smooth": y_pred_smooth,
        "csv_row_tail": [y_pred_raw, y_pred_clipped, y_pred_smooth],
        "print_lines": [
            f"[SOMN] Predicción bruta    : {y_pred_raw:.4f}",
            f"[SOMN] Predicción acotada  : {y_pred_clipped:.4f}",
            f"[SOMN] Predicción suavizada: {y_pred_smooth:.4f}",
        ],
    }


def postprocess_concentracion(model, X_live, pred_queue):
    probas = model.predict_proba(X_live)[0]
    pred_class = int(np.argmax(probas))

    pred_queue.append(pred_class)
    values, counts = np.unique(np.array(pred_queue), return_counts=True)
    pred_class_smooth = int(values[np.argmax(counts)])

    class_names = {0: "Reposo", 1: "Cálculo mental"}

    return {
        "probas": probas,
        "pred_class": pred_class,
        "pred_class_smooth": pred_class_smooth,
        "csv_row_tail": [float(probas[0]), float(probas[1]), pred_class, pred_class_smooth],
        "print_lines": [
            f"[CONC] Predicción actual    : {class_names[pred_class]} "
            f"(p_reposo={probas[0]:.3f}, p_calculo={probas[1]:.3f})",
            f"[CONC] Predicción suavizada : {class_names[pred_class_smooth]}",
        ],
    }


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Main unificado para somnolencia y concentración")
    parser.add_argument("--modo", choices=["somnolencia", "concentracion"], required=True)
    parser.add_argument("--source", choices=["emulador", "bluno"], default="emulador")
    args = parser.parse_args()

    cfg = load_mode_config(args.modo)

    print("======================================")
    print("MAIN UNIFICADO")
    print("Modo           :", args.modo)
    print("Fuente         :", args.source)
    print("Modelo         :", cfg["model_path"])
    print("Fs entrada     :", FS_IN)
    print("Fs objetivo    :", cfg["fs_target"])
    print("Ventana (s)    :", cfg["window_sec"])
    print("Paso (s)       :", cfg["step_sec"])
    print("Smooth feats   :", cfg["feature_smooth_windows"])
    print("Smooth pred    :", cfg["prediction_smooth_windows"])
    print("======================================")

    model = joblib.load(cfg["model_path"])

    init_raw_csv(cfg["raw_csv"])
    init_features_csv(cfg["features_csv"])
    init_predictions_csv(cfg["predictions_csv"], cfg["mode"])

    window_samples_in = int(cfg["window_sec"] * FS_IN)
    buffer_seconds = max(8, int(cfg["window_sec"]) + 2)
    buffer_size = FS_IN * buffer_seconds
    step_samples = int(FS_IN * cfg["step_sec"])

    buffer = RingBuffer2Ch(buffer_size)
    feature_queue = deque(maxlen=cfg["feature_smooth_windows"])
    pred_queue = deque(maxlen=cfg["prediction_smooth_windows"])

    source = build_source(args.source)

    count = 0
    last_seq = None
    next_feature_at = window_samples_in

    print("[MAIN] Fuente conectada. Iniciando bucle...\n")

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
                cfg["raw_csv"],
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
                tag = "[SOMN]" if args.modo == "somnolencia" else "[CONC]"
                print(
                    f"{tag} muestras_totales={count} "
                    f"buffer_len={len(buffer)} full={buffer.is_full} "
                    f"ch1_std={np.std(x1_dbg):.2f} ch2_std={np.std(x2_dbg):.2f}"
                )

            if len(buffer) >= window_samples_in and count >= next_feature_at:
                x1_all, x2_all = buffer.get_ordered()

                ch1_win = x1_all[-window_samples_in:]
                ch2_win = x2_all[-window_samples_in:]

                if cfg["fs_target"] != FS_IN:
                    target_len = int(round(len(ch1_win) * cfg["fs_target"] / FS_IN))
                    ch1_proc = resample(ch1_win, target_len)
                    ch2_proc = resample(ch2_win, target_len)
                    fs_feat = cfg["fs_target"]
                else:
                    ch1_proc = ch1_win
                    ch2_proc = ch2_win
                    fs_feat = FS_IN

                feat_vec = extract_feature_vector(
                    ch1_proc,
                    ch2_proc,
                    fs_feat,
                    cfg["bands"]
                )

                feature_queue.append(feat_vec)
                feat_smooth = np.mean(np.stack(feature_queue, axis=0), axis=0)

                append_features_csv(
                    cfg["features_csv"],
                    timestamp_now,
                    count,
                    seq,
                    feat_smooth,
                )

                X_live = feat_smooth.reshape(1, -1)

                if args.modo == "somnolencia":
                    out = postprocess_somnolencia(model, X_live, pred_queue)
                else:
                    out = postprocess_concentracion(model, X_live, pred_queue)

                append_predictions_csv(
                    cfg["predictions_csv"],
                    [timestamp_now, count, seq] + out["csv_row_tail"]
                )

                tag = "[SOMN]" if args.modo == "somnolencia" else "[CONC]"
                print(f"\n{tag} Vector de 10 features (DE suavizada):")
                print(np.round(feat_smooth, 4))
                for line in out["print_lines"]:
                    print(line)
                print(f"{tag} Guardado en CSV | muestra={count} | seq={seq}")

                next_feature_at += step_samples

    finally:
        source.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Detenido por usuario.")
