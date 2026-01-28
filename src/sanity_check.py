import os
import librosa
import numpy as np

REAL_DIR = "../data/processed/real"
FAKE_DIR = "../data/processed/fake"

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

                if duration < 1.0:
                    bad_files += 1
            except:
                bad_files += 1

    return np.mean(durations), np.std(durations), bad_files, len(durations)


real_mean, real_std, real_bad, real_total = analyze_folder(REAL_DIR)
fake_mean, fake_std, fake_bad, fake_total = analyze_folder(FAKE_DIR)

print("REAL AUDIO:")
print("Total files:", real_total)
print("Mean duration:", round(real_mean, 2), "sec")
print("Std duration:", round(real_std, 2), "sec")
print("Bad/short files:", real_bad)

print("\nFAKE AUDIO:")
print("Total files:", fake_total)
print("Mean duration:", round(fake_mean, 2), "sec")
print("Std duration:", round(fake_std, 2), "sec")
print("Bad/short files:", fake_bad)
