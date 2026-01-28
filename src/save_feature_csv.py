import os
import csv
from feature_extraction import extract_features

REAL_DIR = "../data/processed/real"
FAKE_DIR = "../data/processed/fake"

OUTPUT_CSV = "../features.csv"

header = [
    "label",
    "breath_count",
    "mean_breath_duration",
    "var_breath_duration",
    "var_breath_interval",
    "breath_interval_entropy",
    "pause_count",
    "mean_pause_duration",
    "long_pause_ratio",
    "mean_speech_after_breath",
    "breath_speech_alignment"
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # REAL samples
    for file in os.listdir(REAL_DIR):
        if file.endswith(".wav"):
            feats = extract_features(os.path.join(REAL_DIR, file))
            writer.writerow(["real"] + feats)

    # FAKE samples
    for file in os.listdir(FAKE_DIR):
        if file.endswith(".wav"):
            feats = extract_features(os.path.join(FAKE_DIR, file))
            writer.writerow(["fake"] + feats)

print("✅ Features saved to features.csv")
