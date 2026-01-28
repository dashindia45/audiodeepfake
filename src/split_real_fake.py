import os
import shutil

FLAC_DIR = "../ASVspoof2019_LA_dev/flac"
PROTOCOL = "../ASVspoof2019_LA_dev/ASVspoof2019.LA.dev.protocol.txt"

REAL_OUT = "../data/raw/real"
FAKE_OUT = "../data/raw/fake"

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

with open(PROTOCOL, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_id = parts[1]
        label = parts[-1]

        src_file = os.path.join(FLAC_DIR, file_id + ".flac")

        if not os.path.exists(src_file):
            continue

        if label == "bonafide":
            shutil.copy(src_file, REAL_OUT)
        elif label == "spoof":
            shutil.copy(src_file, FAKE_OUT)

print("✅ Real and Fake files separated successfully")
