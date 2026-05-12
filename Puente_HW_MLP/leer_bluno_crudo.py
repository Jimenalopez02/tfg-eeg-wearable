import serial
import time

SERIAL_PORT = "COM5"   # cambia esto
BAUDRATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
time.sleep(2)

print("Escuchando datos del Bluno...\n")

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(line)
except KeyboardInterrupt:
    print("\nLectura detenida.")
finally:
    ser.close()
    print("Puerto cerrado.")