import os
import shutil


# Dataset Root Folder


DATASET_ROOT = "../LA"
PROTOCOL_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols")

OUTPUT_ROOT = "../data/raw"


# Function to Split Real / Fake using Protocol


def split_dataset(split_name, protocol_file):

    print(f"\nProcessing {split_name} set...")

    FLAC_DIR = os.path.join(DATASET_ROOT, f"ASVspoof2019_LA_{split_name}", "flac")

    REAL_OUT = os.path.join(OUTPUT_ROOT, split_name, "real")
    FAKE_OUT = os.path.join(OUTPUT_ROOT, split_name, "fake")

    os.makedirs(REAL_OUT, exist_ok=True)
    os.makedirs(FAKE_OUT, exist_ok=True)

    with open(protocol_file, "r") as f:
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

    print(f"✅ {split_name} split completed")



# Splitting


split_dataset(
    "train",
    os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
)

split_dataset(
    "dev",
    os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
)

split_dataset(
    "eval",
    os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")
)

print("\n🎉 Real/Fake separation completed for train/dev/eval")