import argparse
import csv
import socket
import struct
import time
from pathlib import Path

import mne
import numpy as np

HOST = "127.0.0.1"
PORT = 50030
FS = 500.0
DT = 1.0 / FS

EEGMAT_DIR = Path(r"..\EEGMAT").resolve()
TIMELINE_CSV = "replay_timeline_concentracion_fiel.csv"


def pack_sample_float(ch1: float, ch2: float) -> bytes:
    return struct.pack("!2f", float(ch1), float(ch2))


def load_segment(edf_path: Path, channels=("EEG Fp1", "EEG Fp2"), tmin=1.0, tmax=61.0):
    if not edf_path.exists():
        raise FileNotFoundError(f"No encuentro {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(list(channels))
    raw.crop(tmin=tmin, tmax=tmax)

    sfreq = raw.info["sfreq"]
    if abs(sfreq - FS) > 1e-6:
        raise ValueError(f"Esperaba Fs=500 Hz y encontré {sfreq}")

    data = raw.get_data()  # (2, n_samples)
    return data[0], data[1]


def build_sequence(subject: str):
    s = f"{int(subject):02d}"
    return [
        {"file": EEGMAT_DIR / f"Subject{s}_1.edf", "label": 0, "state_name": "Reposo"},
        {"file": EEGMAT_DIR / f"Subject{s}_2.edf", "label": 1, "state_name": "Cálculo mental"},
        {"file": EEGMAT_DIR / f"Subject{s}_1.edf", "label": 0, "state_name": "Reposo"},
    ]


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
    parser = argparse.ArgumentParser(description="Replay fiel de EEGMAT para concentración")
    parser.add_argument("--subject", default="01", help="Sujeto a usar, por ejemplo 01")
    args = parser.parse_args()

    sequence = build_sequence(args.subject)

    print(f"[REPLAY-FIEL] Esperando conexión en {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            print(f"[REPLAY-FIEL] Cliente conectado desde {addr}")

            global_n = 0
            timeline_rows = []

            for seg_idx, seg in enumerate(sequence):
                ch1, ch2 = load_segment(seg["file"])
                n_samples = min(len(ch1), len(ch2))
                duration_sec = n_samples / FS
                start_sec = global_n / FS
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
                    f"[REPLAY-FIEL] Segmento {seg_idx}: {seg['file'].name} | "
                    f"estado={seg['state_name']} | duración={duration_sec:.2f}s"
                )

                t0 = time.perf_counter()
                start_n = global_n

                for i in range(n_samples):
                    target_time = t0 + i * DT
                    now = time.perf_counter()
                    sleep_time = target_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    packet = pack_sample_float(ch1[i], ch2[i])
                    conn.sendall(packet)

                    if i % int(FS) == 0:
                        t_rel = (global_n - start_n) / FS
                        print(
                            f"[REPLAY-FIEL] t_seg={t_rel:6.2f}s "
                            f"estado={seg['state_name']:<15} "
                            f"ch1={ch1[i]:8.3f} ch2={ch2[i]:8.3f}"
                        )

                    global_n += 1

            write_timeline(timeline_rows, TIMELINE_CSV)
            print("\n[REPLAY-FIEL] Replay terminado.")
            print(f"[REPLAY-FIEL] Timeline guardado en: {Path(TIMELINE_CSV).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[REPLAY-FIEL] Detenido por usuario.")