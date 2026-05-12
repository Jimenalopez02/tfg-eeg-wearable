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

STATE_DURATION_SEC = 30  # alterna cada 30 s


def clamp_adc(x: float) -> int:
    return max(0, min(1023, int(round(x))))


def get_state(t: float) -> int:
    # 0 = reposo, 1 = cálculo mental
    return int(t // STATE_DURATION_SEC) % 2


def generate_sample(t: float) -> tuple[int, int, int]:
    base = 512
    state = get_state(t)

    if state == 0:
        # REPOSO:
        # alfa muy dominante, poco beta, poco ruido
        ch1 = (
            base
            + 95 * math.sin(2 * math.pi * 10 * t)
            + 12 * math.sin(2 * math.pi * 6 * t)
            + 5 * math.sin(2 * math.pi * 20 * t)
            + random.gauss(0, 4)
        )
        ch2 = (
            base
            + 90 * math.sin(2 * math.pi * 10 * t + 0.25)
            + 10 * math.sin(2 * math.pi * 6 * t)
            + 5 * math.sin(2 * math.pi * 18 * t)
            + random.gauss(0, 4)
        )

    else:
        # CÁLCULO MENTAL:
        # menos alfa, mucho más beta/theta, más ruido
        ch1 = (
            base
            + 20 * math.sin(2 * math.pi * 10 * t)
            + 45 * math.sin(2 * math.pi * 20 * t)
            + 38 * math.sin(2 * math.pi * 5 * t)
            + 18 * math.sin(2 * math.pi * 30 * t)
            + random.gauss(0, 12)
        )
        ch2 = (
            base
            + 18 * math.sin(2 * math.pi * 10 * t + 0.15)
            + 42 * math.sin(2 * math.pi * 18 * t)
            + 35 * math.sin(2 * math.pi * 5 * t + 0.35)
            + 16 * math.sin(2 * math.pi * 28 * t)
            + random.gauss(0, 12)
        )

    # artefacto breve ocasional
    if int(t) % 20 == 0 and (t - int(t)) < 0.08:
        ch1 += 60
        ch2 -= 50

    return clamp_adc(ch1), clamp_adc(ch2), state


def pack_sample(seq: int, ch1: int, ch2: int) -> bytes:
    ch1_msb = (ch1 >> 8) & 0xFF
    ch1_lsb = ch1 & 0xFF
    ch2_msb = (ch2 >> 8) & 0xFF
    ch2_lsb = ch2 & 0xFF
    return struct.pack("!6B", HEADER, seq, ch1_msb, ch1_lsb, ch2_msb, ch2_lsb)


def main() -> None:
    print(f"[EMU-CONC] Esperando conexión en {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        conn, addr = server.accept()
        with conn:
            print(f"[EMU-CONC] Cliente conectado desde {addr}")

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
                ch1, ch2, state = generate_sample(t)
                packet = pack_sample(seq, ch1, ch2)
                conn.sendall(packet)

                if n % FS == 0:
                    state_name = "Reposo" if state == 0 else "Cálculo mental"
                    print(
                        f"[EMU-CONC] t={t:7.2f}s seq={seq:3d} "
                        f"estado={state_name:<15} ch1={ch1:4d} ch2={ch2:4d}"
                    )

                seq = (seq + 1) % 256
                n += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EMU-CONC] Detenido por usuario.")