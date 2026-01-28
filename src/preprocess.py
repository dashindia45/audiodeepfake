import librosa
import soundfile as sf
import os

def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".flac") or file.endswith(".wav"):
            in_path = os.path.join(input_dir, file)

            # Load audio
            y, sr = librosa.load(in_path, sr=16000, mono=True)

            # Normalize amplitude
            y = librosa.util.normalize(y)

            # Save processed audio
            out_path = os.path.join(output_dir, file.replace(".flac", ".wav"))
            sf.write(out_path, y, 16000)

    print(f"Processed files saved to {output_dir}")

if __name__ == "__main__":
    preprocess_folder("../data/raw/real", "../data/processed/real")
    preprocess_folder("../data/raw/fake", "../data/processed/fake")
