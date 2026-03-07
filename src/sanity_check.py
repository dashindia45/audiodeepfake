import os
import librosa
import numpy as np


# =====================================================
# Function: Analyze audio folder
# =====================================================
# This checks:
# - number of files
# - mean duration
# - duration variance
# - corrupted / short files
# =====================================================

def analyze_folder(folder):

    durations = []
    bad_files = 0

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            path = os.path.join(folder, file)

            try:
                y, sr = librosa.load(path, sr=16000)
                duration = len(y) / sr

                durations.append(duration)

                # mark extremely short audio as problematic
                if duration < 1.0:
                    bad_files += 1

            except:
                bad_files += 1

    if len(durations) == 0:
        return 0, 0, bad_files, 0

    return np.mean(durations), np.std(durations), bad_files, len(durations)


# =====================================================
# Function: Run sanity check for a dataset split
# =====================================================

def sanity_check_split(split):

    REAL_DIR = f"../data/processed/{split}/real"
    FAKE_DIR = f"../data/processed/{split}/fake"

    real_mean, real_std, real_bad, real_total = analyze_folder(REAL_DIR)
    fake_mean, fake_std, fake_bad, fake_total = analyze_folder(FAKE_DIR)

    print("\n" + "=" * 50)
    print(f"DATASET SPLIT: {split.upper()}")
    print("=" * 50)

    print("\nREAL AUDIO:")
    print("Total files:", real_total)
    print("Mean duration:", round(real_mean, 2), "sec")
    print("Std duration:", round(real_std, 2), "sec")
    print("Bad/short files:", real_bad)

    print("\nFAKE AUDIO:")
    print("Total files:", fake_total)
    print("Mean duration:", round(fake_mean, 2), "sec")
    print("Std duration:", round(fake_std, 2), "sec")
    print("Bad/short files:", fake_bad)


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":

    sanity_check_split("train")
    sanity_check_split("dev")
    sanity_check_split("eval")

    print("\n✅ Dataset sanity check completed")