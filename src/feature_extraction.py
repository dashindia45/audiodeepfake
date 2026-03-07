import librosa
import numpy as np
import os
from scipy.stats import entropy

SR = 16000
FRAME_LEN = int(0.03 * SR)
HOP_LEN = int(0.01 * SR)

MIN_DURATION = 1.0


def extract_features(audio_path):

    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception as e:
        print("Error loading:", audio_path)
        return None

    duration = len(y) / sr

    # Skip short audio (<1 sec)
    if duration < MIN_DURATION:
        return None

    # =====================================================
    # Frame-level Physiology Features
    # =====================================================

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )[0]

    S = np.abs(librosa.stft(
        y,
        n_fft=FRAME_LEN,
        hop_length=HOP_LEN
    ))

    flat = librosa.feature.spectral_flatness(S=S)[0]

    # Normalize
    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    flat_n = (flat - flat.min()) / (flat.max() - flat.min() + 1e-6)

    # Detect breath frames
    breath_frames = np.logical_and(rms_n < 0.3, flat_n > 0.5)

    # Detect pause frames
    pause_frames = rms_n < 0.1

    # Convert frames to segments
    def frames_to_segments(frames):

        segments = []
        start = None

        for i, val in enumerate(frames):

            if val and start is None:
                start = i

            elif not val and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(frames)))

        return segments

    breath_segments = frames_to_segments(breath_frames)
    pause_segments = frames_to_segments(pause_frames)

    # ---------------- Breathing Features ----------------

    breath_count = len(breath_segments)

    breath_durations = [
        (e - s) * HOP_LEN / SR
        for s, e in breath_segments
    ]

    mean_breath_dur = np.mean(breath_durations) if breath_durations else 0
    var_breath_dur = np.var(breath_durations) if breath_durations else 0

    breath_intervals = []

    for i in range(len(breath_segments) - 1):

        gap = (
            breath_segments[i+1][0]
            - breath_segments[i][1]
        ) * HOP_LEN / SR

        if gap > 0:
            breath_intervals.append(gap)

    var_breath_interval = np.var(breath_intervals) if breath_intervals else 0

    interval_entropy = (
        entropy(np.histogram(breath_intervals, bins=5)[0] + 1e-6)
        if breath_intervals else 0
    )

    # ---------------- Pause Features ----------------

    pause_durations = [
        (e - s) * HOP_LEN / SR
        for s, e in pause_segments
    ]

    pause_count = len(pause_durations)

    mean_pause_dur = (
        np.mean(pause_durations)
        if pause_durations else 0
    )

    long_pause_ratio = (
        sum(d > 0.5 for d in pause_durations) / pause_count
        if pause_count else 0
    )

    # ---------------- Speech–Breath Coupling ----------------

    speech_after_breath = []
    alignment_score = 0

    for s, e in breath_segments:

        next_speech_start = e
        dur = 0

        while (
            next_speech_start < len(rms_n)
            and rms_n[next_speech_start] > 0.3
        ):
            dur += HOP_LEN / SR
            next_speech_start += 1

        speech_after_breath.append(dur)

        if dur > 0.5:
            alignment_score += 1

    mean_speech_after_breath = (
        np.mean(speech_after_breath)
        if speech_after_breath else 0
    )

    alignment_score = (
        alignment_score / breath_count
        if breath_count else 0
    )

    # =====================================================
    # Spectral Features
    # =====================================================

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    flat_mean = np.mean(flat)
    flat_std = np.std(flat)

    return [float(x) for x in [
    breath_count,
    mean_breath_dur,
    var_breath_dur,
    var_breath_interval,
    interval_entropy,
    pause_count,
    mean_pause_dur,
    long_pause_ratio,
    mean_speech_after_breath,
    alignment_score,
    *mfcc_means,
    *mfcc_stds,
    centroid_mean,
    centroid_std,
    rolloff_mean,
    rolloff_std,
    flat_mean,
    flat_std
]]


# =====================================================
# Simple test
# =====================================================

if __name__ == "__main__":

    test_dir = "../data/processed/train/real"

    file_list = os.listdir(test_dir)

    test_file = os.path.join(test_dir, file_list[0])

    features = extract_features(test_file)

    print("Extracted features:\n", features)