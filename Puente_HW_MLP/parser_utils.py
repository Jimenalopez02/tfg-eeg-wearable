import struct

PACKET_SIZE = 6
HEADER = 0xAA


def parse_packet(packet: bytes) -> tuple[int, int, int]:
    h, seq, ch1_msb, ch1_lsb, ch2_msb, ch2_lsb = struct.unpack("!6B", packet)

    if h != HEADER:
        raise ValueError(f"Cabecera inválida: {h:#x}")

    ch1 = (ch1_msb << 8) | ch1_lsb
    ch2 = (ch2_msb << 8) | ch2_lsb
    return seq, ch1, ch2


def adc_to_centered(adc_value: int) -> float:
    return float(adc_value - 512)