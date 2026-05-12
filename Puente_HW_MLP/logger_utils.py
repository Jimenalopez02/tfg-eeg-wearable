import csv
import os


def init_raw_csv(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return

    header = [
        "timestamp",
        "sample_count",
        "seq",
        "ch1_adc",
        "ch2_adc",
        "ch1_centered",
        "ch2_centered",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_raw_csv(
    csv_path: str,
    timestamp: float,
    sample_count: int,
    seq: int,
    ch1_adc: int,
    ch2_adc: int,
    ch1_centered: float,
    ch2_centered: float,
) -> None:
    row = [
        timestamp,
        sample_count,
        seq,
        ch1_adc,
        ch2_adc,
        ch1_centered,
        ch2_centered,
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def init_features_csv(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return

    header = [
        "timestamp",
        "sample_count",
        "seq",
        "f_ch1_delta",
        "f_ch1_theta",
        "f_ch1_alpha",
        "f_ch1_beta",
        "f_ch1_gamma",
        "f_ch2_delta",
        "f_ch2_theta",
        "f_ch2_alpha",
        "f_ch2_beta",
        "f_ch2_gamma",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_features_csv(
    csv_path: str,
    timestamp: float,
    sample_count: int,
    seq: int,
    features,
) -> None:
    row = [timestamp, sample_count, seq] + list(features)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)