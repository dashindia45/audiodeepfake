import os
import csv
from feature_extraction import extract_features

DATA_ROOT = "../data/processed"
FEATURE_ROOT = "../features"

os.makedirs(FEATURE_ROOT, exist_ok=True)


# =====================================================
# Header Definition
# =====================================================

header = [
    "label",

    # Physiology
    "breath_count",
    "mean_breath_duration",
    "var_breath_duration",
    "var_breath_interval",
    "breath_interval_entropy",
    "pause_count",
    "mean_pause_duration",
    "long_pause_ratio",
    "mean_speech_after_breath",
    "breath_speech_alignment",
]

# MFCC Mean (13)
for i in range(13):
    header.append(f"spec_mfcc{i+1}_mean")

# MFCC Std (13)
for i in range(13):
    header.append(f"spec_mfcc{i+1}_std")

# Spectral Stats
header += [
    "spec_centroid_mean",
    "spec_centroid_std",
    "spec_rolloff_mean",
    "spec_rolloff_std",
    "spec_flatness_mean",
    "spec_flatness_std"
]


# =====================================================
# Function to process one dataset split
# =====================================================

def process_split(split):

    REAL_DIR = os.path.join(DATA_ROOT, split, "real")
    FAKE_DIR = os.path.join(DATA_ROOT, split, "fake")

    OUTPUT_CSV = os.path.join(FEATURE_ROOT, f"features_{split}.csv")

    processed = 0
    skipped = 0

    with open(OUTPUT_CSV, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        # REAL samples
        for file in os.listdir(REAL_DIR):

            if file.endswith(".wav"):

                path = os.path.join(REAL_DIR, file)

                feats = extract_features(path)

                if feats is None:
                    skipped += 1
                    continue

                writer.writerow(["real"] + feats)
                processed += 1

        # FAKE samples
        for file in os.listdir(FAKE_DIR):

            if file.endswith(".wav"):

                path = os.path.join(FAKE_DIR, file)

                feats = extract_features(path)

                if feats is None:
                    skipped += 1
                    continue

                writer.writerow(["fake"] + feats)
                processed += 1

    print(f"\nSaved features for {split}")
    print("Output:", OUTPUT_CSV)
    print("Processed:", processed)
    print("Skipped short files:", skipped)


# =====================================================
# Run for train/dev/eval
# =====================================================

if __name__ == "__main__":

    process_split("train")
    process_split("dev")
    process_split("eval")

    print("\n✅ Hybrid Features saved successfully")