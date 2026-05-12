import argparse
import csv
import math
import socket
import struct
import time
from pathlib import Path

import mne
import numpy as np

HOST = "127.0.0.1"
PORT = 50000
HEADER = 0xAA
FS_OUT = 250  # frecuencia a la que emulamos la transmisión
DT = 1.0 / FS_OUT

EEGMAT_DIR = Path(r"..\EEGMAT").resolve()
TIMELINE_CSV = "replay_timeline_concentracion.csv"


def clamp_adc(x: float) -> int:
    return max(0, min(1023, int(round(x))))


def pack_sample(seq: int, ch1: int, ch2: int) -> bytes:
    ch1_msb = (ch1 >> 8) & 0xFF
    ch1_lsb = ch1 & 0xFF
    ch2_msb = (ch2 >> 8) & 0xFF
    ch2_lsb = ch2 & 0xFF
    return struct.pack("!6B", HEADER, seq, ch1_msb, ch1_lsb, ch2_msb, ch2_lsb)


def robust_adc_map(sig: np.ndarray, center_adc: float = 512.0, target_amp: float = 120.0) -> np.ndarray:
    """
    Convierte una señal EEG real (uV) a escala ADC pseudo-Bluno.
    Se centra por mediana y se escala con percentil 99 para evitar clipping excesivo.
    """
    x = sig.astype(float).copy()
    x = x - np.median(x)

    p99 = np.percentile(np.abs(x), 99)
    if p99 < 1e-9:
        p99 = 1.0

    gain = target_amp / p99
    adc = center_adc + gain * x
    adc = np.clip(np.round(adc), 0, 1023).astype(int)
    return adc


def load_segment_as_adc(edf_path: Path, channels=("EEG Fp1", "EEG Fp2"), tmin=1.0, tmax=61.0, fs_out=250):
    if not edf_path.exists():
        raise FileNotFoundError(f"No encuentro {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(list(channels))
    raw.crop(tmin=tmin, tmax=tmax)
    raw.resample(fs_out, npad="auto")

    data = raw.get_data()  # shape (2, n_samples)

    ch1_adc = robust_adc_map(data[0])
    ch2_adc = robust_adc_map(data[1])

    return ch1_adc, ch2_adc, fs_out


def build_sequence(subject: str):
    """
    Secuencia reposo -> cálculo mental -> reposo
    para ver claramente dos transiciones.
    """
    s = f"{int(subject):02d}"

    seq = [
        {
            "file": EEGMAT_DIR / f"Subject{s}_1.edf",
            "label": 0,
            "state_name": "Reposo",
        },
        {
            "file": EEGMAT_DIR / f"Subject{s}_2.edf",
            "label": 1,
            "state_name": "Cálculo mental",
        },
        {
            "file": EEGMAT_DIR / f"Subject{s}_1.edf",
            "label": 0,
            "state_name": "Reposo",
        },
    ]
    return seq


def write_timeline(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_idx",
            "file_name",
            "label",
            "state_name",
            "start_sec",
            "end_sec",
            "duration_sec",
        ])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Replay pseudo-real de EEGMAT para concentración")
    parser.add_argument("--subject", default="01", help="Sujeto a usar, por ejemplo 01")
    args = parser.parse_args()

    sequence = build_sequence(args.subject)

    print(f"[REPLAY-CONC] Esperando conexión en {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            print(f"[REPLAY-CONC] Cliente conectado desde {addr}")

            seq = 0
            global_n = 0
            timeline_rows = []

            for seg_idx, seg in enumerate(sequence):
                ch1_adc, ch2_adc, fs_loaded = load_segment_as_adc(seg["file"], fs_out=FS_OUT)

                n_samples = min(len(ch1_adc), len(ch2_adc))
                duration_sec = n_samples / FS_OUT
                start_sec = global_n / FS_OUT
                end_sec = start_sec + duration_sec

                timeline_rows.append([
                    seg_idx,
                    seg["file"].name,
                    seg["label"],
                    seg["state_name"],
                    round(start_sec, 3),
                    round(end_sec, 3),
                    round(duration_sec, 3),
                ])

                print(
                    f"[REPLAY-CONC] Segmento {seg_idx}: {seg['file'].name} | "
                    f"estado={seg['state_name']} | duración={duration_sec:.2f}s"
                )

                t0 = time.perf_counter()
                start_n = global_n

                for i in range(n_samples):
                    target_time = t0 + (i * DT)
                    now = time.perf_counter()
                    sleep_time = target_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    packet = pack_sample(seq, int(ch1_adc[i]), int(ch2_adc[i]))
                    conn.sendall(packet)

                    if i % FS_OUT == 0:
                        t_rel = (global_n - start_n) / FS_OUT
                        print(
                            f"[REPLAY-CONC] t_seg={t_rel:6.2f}s seq={seq:3d} "
                            f"estado={seg['state_name']:<15} ch1={int(ch1_adc[i]):4d} ch2={int(ch2_adc[i]):4d}"
                        )

                    seq = (seq + 1) % 256
                    global_n += 1

            write_timeline(timeline_rows, TIMELINE_CSV)
            print(f"\n[REPLAY-CONC] Replay terminado.")
            print(f"[REPLAY-CONC] Timeline guardado en: {Path(TIMELINE_CSV).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[REPLAY-CONC] Detenido por usuario.")