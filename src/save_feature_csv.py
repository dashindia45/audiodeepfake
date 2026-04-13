
import os
import csv
from feature_extraction import extract_features
from tqdm import tqdm  

# =====================================================
# PATH CONFIG
# =====================================================

DATA_ROOT = "../data/processed"
FEATURE_ROOT = "../features"

os.makedirs(FEATURE_ROOT, exist_ok=True)

# =====================================================
# HEADER
# =====================================================

header = ["label"]

header += ["breath_count", "pause_count"]

for i in range(20):
    header.append(f"mfcc_{i+1}_mean")
for i in range(20):
    header.append(f"mfcc_{i+1}_std")

for i in range(20):
    header.append(f"delta_{i+1}_mean")
for i in range(20):
    header.append(f"delta_{i+1}_std")

for i in range(20):
    header.append(f"delta2_{i+1}_mean")
for i in range(20):
    header.append(f"delta2_{i+1}_std")

header += [
    "centroid_mean", "centroid_std",
    "bandwidth_mean", "bandwidth_std"
]

for i in range(7):
    header.append(f"contrast_{i+1}")

for i in range(12):
    header.append(f"chroma_{i+1}")

header += ["zcr_mean", "zcr_std", "energy_entropy"]

EXPECTED_FEATURE_LEN = len(header) - 1
print(f"✅ Expected feature length: {EXPECTED_FEATURE_LEN}")

# =====================================================
# PROCESS FUNCTION
# =====================================================

def process_split(split):

    real_dir = os.path.join(DATA_ROOT, split, "real")
    fake_dir = os.path.join(DATA_ROOT, split, "fake")

    output_csv = os.path.join(FEATURE_ROOT, f"featurenew_{split}.csv")

    processed = 0
    skipped = 0

    # ✅ Combine all files with labels
    all_files = []

    for f in os.listdir(real_dir):
        if f.endswith(".wav"):
            all_files.append((os.path.join(real_dir, f), "real"))

    for f in os.listdir(fake_dir):
        if f.endswith(".wav"):
            all_files.append((os.path.join(fake_dir, f), "fake"))

    print(f"\n📦 {split.upper()} → Total files: {len(all_files)}")

    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        # ✅ SINGLE CLEAN PROGRESS BAR
        for path, label in tqdm(
            all_files,
            desc=f"{split.upper()} Processing",
            unit="file",
            ncols=100
        ):

            feats = extract_features(path)

            # 🔥 DEBUG
            if feats is None:
                print(f"\n⚠️ Skipped (None): {path}")
                skipped += 1
                continue

            if len(feats) != EXPECTED_FEATURE_LEN:
                print(f"\n⚠️ Skipped (Length Mismatch): {path}")
                print(f"   Got: {len(feats)}, Expected: {EXPECTED_FEATURE_LEN}")
                skipped += 1
                continue

            writer.writerow([label] + feats.tolist())
            processed += 1

    print(f"\n📊 Split: {split}")
    print(f"Saved → {output_csv}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("\n🚀 Starting Feature Extraction Pipeline...\n")

    process_split("train")
    process_split("dev")
    process_split("eval")

    print("\n✅ All feature files generated successfully!")