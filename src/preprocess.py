import librosa
import soundfile as sf
import os


# =====================================================
# Function: Preprocess a folder of audio files
# =====================================================
# What it does:
# - Loads audio
# - Resamples to 16kHz
# - Converts to mono
# - Normalizes amplitude
# - Saves as WAV
#
# Why:
# - Ensures consistent audio format for feature extraction
# - Removes variation caused by sampling rate differences
# =====================================================

def preprocess_folder(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):

        if file.endswith(".flac") or file.endswith(".wav"):

            in_path = os.path.join(input_dir, file)

            try:
                # Load audio
                y, sr = librosa.load(in_path, sr=16000, mono=True)

                # Normalize amplitude
                y = librosa.util.normalize(y)

                # Save processed audio
                out_path = os.path.join(
                    output_dir,
                    file.replace(".flac", ".wav")
                )

                sf.write(out_path, y, 16000)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Processed files saved to {output_dir}")


# =====================================================
# Function: Preprocess one dataset split
# =====================================================
# This keeps train/dev/eval separated
# =====================================================

def preprocess_split(split):

    preprocess_folder(
        f"../data/raw/{split}/real",
        f"../data/processed/{split}/real"
    )

    preprocess_folder(
        f"../data/raw/{split}/fake",
        f"../data/processed/{split}/fake"
    )


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":

    preprocess_split("train")
    preprocess_split("dev")
    preprocess_split("eval")

    print("\n✅ Audio preprocessing completed for train/dev/eval")