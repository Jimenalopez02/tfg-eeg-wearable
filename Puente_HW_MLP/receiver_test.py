import socket
import struct

HOST = "127.0.0.1"
PORT = 50000
PACKET_SIZE = 6
HEADER = 0xAA


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


def main() -> None:
    print(f"[RCV] Conectando a {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        print("[RCV] Conectado.")

        count = 0
        last_seq = None

        while True:
            packet = recv_exact(sock, PACKET_SIZE)
            seq, ch1, ch2 = parse_packet(packet)

            if last_seq is not None:
                expected = (last_seq + 1) % 256
                if seq != expected:
                    print(f"[RCV] Pérdida o desorden: esperado={expected}, recibido={seq}")

            last_seq = seq
            count += 1

            if count % 250 == 0:
                print(f"[RCV] muestras={count} seq={seq} ch1={ch1} ch2={ch2}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RCV] Detenido por usuario.")