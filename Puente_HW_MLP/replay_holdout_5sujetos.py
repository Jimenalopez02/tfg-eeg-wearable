"""
replay_holdout_5sujetos.py
==========================
Replay pseudo-online de los 5 sujetos holdout del EEGMAT que el modelo
rf_concentracion_final nunca vio durante el entrenamiento (seed=42).

Sujetos holdout: 13, 16, 26, 30, 35

Ejecutar desde la carpeta Puente_HW_MLP:
    python replay_holdout_5sujetos.py

Genera por sujeto:
  - holdout_suj{XX}_predictions.csv
  - holdout_suj{XX}_prob_vs_estado.png
  - holdout_suj{XX}_clase_vs_estado.png

Y al final un resumen global en holdout_resumen.txt
"""

import csv
import json
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Configuración ──────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 50031  # puerto distinto al original para no interferir

HOLDOUT_SUBJECTS = [13, 16, 26, 30, 35]

# Rutas relativas desde Puente_HW_MLP (igual que en tus scripts originales)
EEGMAT_DIR  = Path("../EEGMAT").resolve()
MODEL_PATH = Path("../EEGMAT/resultados_concentracion_final/rf_31sujetos_final.joblib").resolve()
CONFIG_PATH = Path("../EEGMAT/resultados_concentracion_final/config_concentracion_final.json").resolve()
OUTPUT_DIR  = Path(".").resolve()  # guarda los resultados en Puente_HW_MLP

FS   = 500.0
DT   = 1.0 / FS
PACKET_SIZE = 8  # 2 × float32


# ── Utilidades de señal ────────────────────────────────────────────────────────
def bandpass_filter(sig, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def differential_entropy(sig):
    var = np.var(sig)
    var = max(var, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def extract_feature_vector(ch1, ch2, fs, bands):
    feats = []
    for _, f1, f2 in bands:
        feats.append(differential_entropy(bandpass_filter(ch1, fs, f1, f2)))
    for _, f1, f2 in bands:
        feats.append(differential_entropy(bandpass_filter(ch2, fs, f1, f2)))
    return np.array(feats, dtype=np.float32)


# ── Carga de EDF ──────────────────────────────────────────────────────────────
def load_segment(edf_path, tmin=1.0, tmax=61.0):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(["EEG Fp1", "EEG Fp2"])
    raw.crop(tmin=tmin, tmax=tmax)
    data = raw.get_data()
    return data[0], data[1]


def build_sequence(subject_idx):
    s = f"{subject_idx:02d}"
    return [
        {"file": EEGMAT_DIR / f"Subject{s}_1.edf", "label": 0, "state": "Reposo"},
        {"file": EEGMAT_DIR / f"Subject{s}_2.edf", "label": 1, "state": "Cálculo mental"},
        {"file": EEGMAT_DIR / f"Subject{s}_1.edf", "label": 0, "state": "Reposo"},
    ]


# ── Hilo SERVIDOR (replay → socket) ───────────────────────────────────────────
def server_thread(subject_idx, ready_event, timeline_out):
    sequence = build_sequence(subject_idx)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        ready_event.set()  # avisa al cliente que ya está escuchando

        conn, _ = srv.accept()
        with conn:
            global_n = 0
            for seg_idx, seg in enumerate(sequence):
                ch1, ch2 = load_segment(seg["file"])
                n = min(len(ch1), len(ch2))
                start_sec = global_n / FS
                end_sec   = start_sec + n / FS

                timeline_out.append({
                    "label":      seg["label"],
                    "start_sec":  start_sec,
                    "end_sec":    end_sec,
                    "state":      seg["state"],
                })

                print(f"  [SERVER] Suj{subject_idx:02d} seg{seg_idx} "
                      f"({seg['state']}) {n/FS:.1f}s")

                t0 = time.perf_counter()
                for i in range(n):
                    target = t0 + i * DT
                    now = time.perf_counter()
                    if target - now > 0:
                        time.sleep(target - now)
                    conn.sendall(struct.pack("!2f", float(ch1[i]), float(ch2[i])))
                    global_n += 1


# ── Hilo CLIENTE (socket → features → modelo) ─────────────────────────────────
def client_thread(model, cfg, results_out):
    bands      = cfg["bands"]
    win_sec    = float(cfg["window_sec"])
    step_sec   = float(cfg["step_sec"])
    moving_k   = int(cfg["moving_k"])

    win_samples  = int(win_sec * FS)
    step_samples = int(step_sec * FS)

    buf1 = deque(maxlen=win_samples)
    buf2 = deque(maxlen=win_samples)
    feat_queue = deque(maxlen=moving_k)
    pred_queue = deque(maxlen=5)

    count        = 0
    next_pred_at = win_samples

    def recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Servidor cerró la conexión.")
            buf += chunk
        return buf

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))

        try:
            while True:
                packet = recv_exact(sock, PACKET_SIZE)
                ch1_v, ch2_v = struct.unpack("!2f", packet)

                buf1.append(ch1_v)
                buf2.append(ch2_v)
                count += 1

                if len(buf1) == win_samples and count >= next_pred_at:
                    ch1_win = np.array(buf1, dtype=float)
                    ch2_win = np.array(buf2, dtype=float)

                    feat = extract_feature_vector(ch1_win, ch2_win, FS, bands)
                    feat_queue.append(feat)
                    feat_smooth = np.mean(np.stack(feat_queue), axis=0)

                    probas     = model.predict_proba(feat_smooth.reshape(1, -1))[0]
                    pred_class = int(np.argmax(probas))

                    pred_queue.append(pred_class)
                    vals, cnts = np.unique(np.array(pred_queue), return_counts=True)
                    pred_smooth = int(vals[np.argmax(cnts)])

                    results_out.append({
                        "time_sec":        count / FS,
                        "prob_reposo":     float(probas[0]),
                        "prob_calculo":    float(probas[1]),
                        "pred_class":      pred_class,
                        "pred_smooth":     pred_smooth,
                    })

                    next_pred_at += step_samples

        except ConnectionError:
            pass  # replay terminado, salida limpia


# ── Análisis y gráficas por sujeto ────────────────────────────────────────────
def analyze_subject(subject_idx, results, timeline, model, cfg):
    if not results:
        print(f"  [ANÁLISIS] Suj{subject_idx:02d}: sin predicciones.")
        return None

    df_times = np.array([r["time_sec"]    for r in results])
    df_prob  = np.array([r["prob_calculo"] for r in results])
    df_pred  = np.array([r["pred_smooth"]  for r in results])

    # Construir etiqueta esperada según timeline
    expected = np.zeros(len(df_times), dtype=int)
    for seg in timeline:
        mask = (df_times >= seg["start_sec"]) & (df_times < seg["end_sec"])
        expected[mask] = seg["label"]

    acc = accuracy_score(expected, df_pred)
    f1  = f1_score(expected, df_pred, zero_division=0)

    print(f"  [ANÁLISIS] Suj{subject_idx:02d} → ACC={acc:.4f}  F1={f1:.4f}")

    # ── CSV ──
    csv_path = OUTPUT_DIR / f"holdout_suj{subject_idx:02d}_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time_sec","prob_reposo","prob_calculo",
            "pred_class","pred_smooth","expected"])
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({**r, "expected": int(expected[i])})

    # ── Gráfica probabilidad ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_times, df_prob, label="P(Cálculo mental)", color="steelblue")
    ax.step(df_times, expected, where="post", linestyle="--",
            color="darkorange", label="Estado esperado")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Probabilidad")
    ax.set_title(f"Sujeto {subject_idx:02d} — Probabilidad predicha vs estado esperado")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"holdout_suj{subject_idx:02d}_prob_vs_estado.png", dpi=150)
    plt.close()

    # ── Gráfica clase ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(df_times, df_pred,   where="post", label="Clase predicha (suavizada)")
    ax.step(df_times, expected,  where="post", linestyle="--",
            color="darkorange", label="Estado esperado")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Clase")
    ax.set_title(f"Sujeto {subject_idx:02d} — Clase predicha vs estado esperado")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Reposo", "Cálculo mental"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"holdout_suj{subject_idx:02d}_clase_vs_estado.png", dpi=150)
    plt.close()

    return {"subject": subject_idx, "acc": acc, "f1": f1, "n_preds": len(results)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Comprobaciones iniciales
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config no encontrada: {CONFIG_PATH}")

    print(f"[HOLDOUT] Modelo: {MODEL_PATH}")
    print(f"[HOLDOUT] EEGMAT: {EEGMAT_DIR}")
    print(f"[HOLDOUT] Sujetos holdout: {HOLDOUT_SUBJECTS}\n")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = joblib.load(MODEL_PATH)

    all_results = []

    for subj in HOLDOUT_SUBJECTS:
        print(f"\n{'='*55}")
        print(f"  Sujeto {subj:02d}")
        print(f"{'='*55}")

        timeline = []
        results  = []

        ready_event = threading.Event()

        srv = threading.Thread(
            target=server_thread,
            args=(subj, ready_event, timeline),
            daemon=True,
        )
        cli = threading.Thread(
            target=client_thread,
            args=(model, cfg, results),
            daemon=True,
        )

        srv.start()
        ready_event.wait()   # espera a que el servidor esté escuchando
        time.sleep(0.05)     # margen mínimo antes de conectar
        cli.start()

        srv.join()
        cli.join(timeout=10)  # timeout de seguridad

        metrics = analyze_subject(subj, results, timeline, model, cfg)
        if metrics:
            all_results.append(metrics)

    # ── Resumen global ──
    print(f"\n{'='*55}")
    print("  RESUMEN HOLDOUT — 5 sujetos nunca vistos")
    print(f"{'='*55}")

    resumen_path = OUTPUT_DIR / "holdout_resumen.txt"
    lines = ["RESUMEN HOLDOUT — 5 sujetos nunca vistos\n",
             f"Modelo: {MODEL_PATH}\n\n"]

    accs, f1s = [], []
    for r in all_results:
        line = (f"  Sujeto {r['subject']:02d}:  "
                f"ACC={r['acc']:.4f}   F1={r['f1']:.4f}   "
                f"({r['n_preds']} predicciones)")
        print(line)
        lines.append(line + "\n")
        accs.append(r["acc"])
        f1s.append(r["f1"])

    if accs:
        mean_line = (f"\n  Media:     ACC={np.mean(accs):.4f}   F1={np.mean(f1s):.4f}")
        std_line  = (f"  Std:       ACC={np.std(accs):.4f}   F1={np.std(f1s):.4f}")
        print(mean_line)
        print(std_line)
        lines += [mean_line + "\n", std_line + "\n"]

    with open(resumen_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[HOLDOUT] Resultados guardados en: {OUTPUT_DIR}")
    print("[HOLDOUT] Archivos generados:")
    for r in all_results:
        s = r["subject"]
        print(f"  holdout_suj{s:02d}_predictions.csv")
        print(f"  holdout_suj{s:02d}_prob_vs_estado.png")
        print(f"  holdout_suj{s:02d}_clase_vs_estado.png")
    print("  holdout_resumen.txt")


if __name__ == "__main__":
    main()
