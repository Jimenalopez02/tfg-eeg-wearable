import socket
import struct
import numpy as np

HOST = "127.0.0.1"
PORT = 50000
PACKET_SIZE = 6
HEADER = 0xAA

FS = 250
BUFFER_SECONDS = 8
BUFFER_SIZE = FS * BUFFER_SECONDS


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
    """
    De momento centramos la señal respecto al punto medio ADC.
    Más adelante, si calibras ganancia y referencia, podrás pasar a voltios.
    """
    return float(adc_value - 512)


def main() -> None:
    print(f"[BUF] Conectando a {HOST}:{PORT}...")

    buffer = RingBuffer2Ch(BUFFER_SIZE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        print("[BUF] Conectado.")

        count = 0
        last_seq = None

        while True:
            packet = recv_exact(sock, PACKET_SIZE)
            seq, ch1_adc, ch2_adc = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[BUF] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq

            ch1 = adc_to_centered(ch1_adc)
            ch2 = adc_to_centered(ch2_adc)

            buffer.append(ch1, ch2)
            count += 1

            if count % FS == 0:
                x1, x2 = buffer.get_ordered()
                print(
                    f"[BUF] muestras_totales={count} "
                    f"buffer_len={len(buffer)} "
                    f"full={buffer.is_full} "
                    f"ch1_mean={np.mean(x1):.2f} ch1_std={np.std(x1):.2f} "
                    f"ch2_mean={np.mean(x2):.2f} ch2_std={np.std(x2):.2f}"
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[BUF] Detenido por usuario.")