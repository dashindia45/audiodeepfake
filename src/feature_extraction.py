import librosa
import numpy as np
import os
from scipy.stats import entropy

# Global setting sampling rate and min durstion for short clips

SR = 16000                 
MIN_DURATION = 1.0         

#  feature extrcation main function

def extract_features(audio_path):
   
   # Load audio 
    
    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception:
        print(f"⚠️ Error loading file: {audio_path}")
        return None

    # Skip very short audio clips (not useful)
    if len(y) / sr < MIN_DURATION:
        return None

    # 1️ BASIC FRAME-LEVEL FEATURES
    

    # Energy of signal
    rms = librosa.feature.rms(y=y)[0]

    # Zero Crossing Rate (noise/roughness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Spectral flatness (noise vs tonal)
    S = np.abs(librosa.stft(y))
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Normalize energy for stability
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    flat_norm = (flatness - flatness.min()) / (flatness.max() - flatness.min() + 1e-6)

    # 2️ PHYSIOLOGICAL FEATURES (Breathing / Pauses)
    

    # Breath detection: low energy + high flatness
    breath_frames = np.logical_and(rms_norm < 0.3, flat_norm > 0.5)

    # Pause detection: very low energy
    pause_frames = rms_norm < 0.1

    breath_count = np.sum(breath_frames)
    pause_count = np.sum(pause_frames)

    
    # 3️ MFCC FEATURES 
 

    # MFCC captures speech characteristics
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # velocity
    mfcc_delta = librosa.feature.delta(mfcc)

    # acceleration
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Statistical summaries
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)

    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    
    # 4️ SPECTRAL FEATURES (VERY IMPORTANT)
   

    # Frequency center of mass
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Spread of frequencies
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

   #contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Harmonic structure
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Summaries
    centroid_stats = [np.mean(centroid), np.std(centroid)]
    bandwidth_stats = [np.mean(bandwidth), np.std(bandwidth)]

    contrast_mean = np.mean(contrast, axis=1)
    chroma_mean = np.mean(chroma, axis=1)

   
    # 5️ NOISE / IRREGULARITY FEATURES
    

    # Normalize energy distribution
    prob_energy = rms / (np.sum(rms) + 1e-6)

    # Entropy → randomness / irregularity
    energy_entropy = entropy(prob_energy)

    zcr_stats = [np.mean(zcr), np.std(zcr)]


    # 6️ FINAL FEATURE VECTOR 
    

    features = np.hstack([

        # --- Physiology (light weight) ---
        breath_count,
        pause_count,

        # --- MFCC ---
        mfcc_mean,
        mfcc_std,

        # --- Temporal Dynamics ---
        delta_mean,
        delta_std,
        delta2_mean,
        delta2_std,

        # --- Spectral ---
        centroid_stats,
        bandwidth_stats,
        contrast_mean,
        chroma_mean,

        # --- Noise ---
        zcr_stats,
        energy_entropy
    ])

    return features.astype(float)


#simple testing

if __name__ == "__main__":

    test_dir = "../data/processed/train/real"

    files = os.listdir(test_dir)

    test_file = os.path.join(test_dir, files[0])

    features = extract_features(test_file)

    print("\n✅ Feature vector extracted successfully!")
    print("Feature length:", len(features))
    print(features)