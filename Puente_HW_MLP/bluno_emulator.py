import math
import random
import socket
import struct
import time

HOST = "127.0.0.1"
PORT = 50000
FS = 250
DT = 1.0 / FS
HEADER = 0xAA


def clamp_adc(x: float) -> int:
    return max(0, min(1023, int(round(x))))


def generate_sample(t: float) -> tuple[int, int]:
    base = 512

    ch1 = (
        base
        + 60 * math.sin(2 * math.pi * 10 * t)
        + 25 * math.sin(2 * math.pi * 4 * t)
        + random.gauss(0, 8)
    )

    ch2 = (
        base
        + 55 * math.sin(2 * math.pi * 10 * t + 0.4)
        + 20 * math.sin(2 * math.pi * 6 * t)
        + random.gauss(0, 8)
    )

    if int(t) % 15 == 0 and (t - int(t)) < 0.2:
        ch1 += 120
        ch2 -= 100

    return clamp_adc(ch1), clamp_adc(ch2)


def pack_sample(seq: int, ch1: int, ch2: int) -> bytes:
    ch1_msb = (ch1 >> 8) & 0xFF
    ch1_lsb = ch1 & 0xFF
    ch2_msb = (ch2 >> 8) & 0xFF
    ch2_lsb = ch2 & 0xFF
    return struct.pack("!6B", HEADER, seq, ch1_msb, ch1_lsb, ch2_msb, ch2_lsb)


def main() -> None:
    print(f"[EMU] Esperando conexión en {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            print(f"[EMU] Cliente conectado desde {addr}")

            seq = 0
            t0 = time.perf_counter()
            n = 0

            while True:
                target_time = t0 + n * DT
                now = time.perf_counter()
                sleep_time = target_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                t = n * DT
                ch1, ch2 = generate_sample(t)
                packet = pack_sample(seq, ch1, ch2)
                conn.sendall(packet)

                if n % FS == 0:
                    print(f"[EMU] t={t:7.2f}s seq={seq:3d} ch1={ch1:4d} ch2={ch2:4d}")

                seq = (seq + 1) % 256
                n += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EMU] Detenido por usuario.")