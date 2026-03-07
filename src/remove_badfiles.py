import os
import librosa

# Minimum acceptable audio duration (seconds)
MIN_DURATION = 1.0

DATA_ROOT = "../data/processed"


# =====================================================
# Function: Validate folder without deleting files
# =====================================================
def validate_folder(folder):

    skipped = 0
    corrupted = 0
    total = 0

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            total += 1
            path = os.path.join(folder, file)

            try:
                y, sr = librosa.load(path, sr=16000)
                duration = len(y) / sr

                if duration < MIN_DURATION:
                    print(f"Skipping short file ({duration:.2f}s): {file}")
                    skipped += 1

            except Exception as e:
                print("Corrupted file detected:", file)
                corrupted += 1

    return total, skipped, corrupted


# =====================================================
# Function: Validate dataset split
# =====================================================
def validate_split(split):

    print("\n" + "=" * 50)
    print(f"VALIDATING SPLIT: {split.upper()}")
    print("=" * 50)

    real_folder = os.path.join(DATA_ROOT, split, "real")
    fake_folder = os.path.join(DATA_ROOT, split, "fake")

    r_total, r_skip, r_corrupt = validate_folder(real_folder)
    f_total, f_skip, f_corrupt = validate_folder(fake_folder)

    print("\nREAL AUDIO")
    print("Total files:", r_total)
    print("Short (<1s):", r_skip)
    print("Corrupted:", r_corrupt)

    print("\nFAKE AUDIO")
    print("Total files:", f_total)
    print("Short (<1s):", f_skip)
    print("Corrupted:", f_corrupt)


# =====================================================
# Main execution
# =====================================================
if __name__ == "__main__":

    validate_split("train")
    validate_split("dev")
    validate_split("eval")

    print("\nDataset validation completed ✅")