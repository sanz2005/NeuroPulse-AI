"""
EMG Preprocessing Pipeline for NeuroPulse AI
Dataset: NinaPro DB2
Sampling Rate: 2000 Hz → resampled to 256 Hz
Task: Muscle Anomaly Detection (normal vs fatigue/anomaly)
Supports subfolder structure: data/raw/emg/DB2_S1/S1_E1_A1.mat
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.io import loadmat
import pickle
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
ORIG_SAMPLE_RATE   = 2000
TARGET_SAMPLE_RATE = 256
WINDOW_SIZE        = 256
STEP_SIZE          = 128
N_CHANNELS         = 12

NORMAL_GESTURES  = list(range(0, 20))
ANOMALY_GESTURES = list(range(20, 50))


def bandpass_filter(signal, lowcut=20.0, highcut=500.0,
                    fs=2000.0, order=4):
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signal[:, ch])
    return filtered


def resample_signal(signal, orig_fs=2000.0, target_fs=256.0):
    num_samples = int(signal.shape[0] * target_fs / orig_fs)
    resampled   = np.zeros((num_samples, signal.shape[1]))
    for ch in range(signal.shape[1]):
        resampled[:, ch] = resample(signal[:, ch], num_samples)
    return resampled


def normalize_signal(signal):
    normalized = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        mean = np.mean(signal[:, ch])
        std  = np.std(signal[:, ch])
        normalized[:, ch] = (signal[:, ch] - mean) / (std + 1e-8)
    return normalized


def load_ninapro_file(mat_path):
    """
    Load a NinaPro DB2 .mat file.
    Handles DB2 structure: emg [samples x 12], restimulus [samples x 1]
    """
    try:
        mat = loadmat(mat_path)

        # Find EMG data
        emg = None
        for key in ['emg', 'EMG', 'data']:
            if key in mat:
                emg = mat[key].astype(np.float32)
                break

        # Find labels
        labels = None
        for key in ['restimulus', 'stimulus', 'label']:
            if key in mat:
                labels = mat[key].flatten().astype(np.int32)
                break

        if emg is None or labels is None:
            return None, None

        # Fix channels
        if emg.shape[1] > N_CHANNELS:
            emg = emg[:, :N_CHANNELS]
        elif emg.shape[1] < N_CHANNELS:
            pad = np.zeros((emg.shape[0], N_CHANNELS - emg.shape[1]))
            emg = np.hstack([emg, pad])

        return emg, labels

    except Exception as e:
        print(f"  Error loading {mat_path}: {e}")
        return None, None


def create_windows(signal, labels, window_size=256, step_size=128):
    windows    = []
    bin_labels = []
    n_samples  = signal.shape[0]

    for start in range(0, n_samples - window_size, step_size):
        end    = start + window_size
        window = signal[start:end, :].T  # [channels, window_size]
        window = (window - window.mean()) / (window.std() + 1e-8)

        window_labels  = labels[start:end]
        majority_label = int(np.bincount(
            np.clip(window_labels, 0, 49)
        ).argmax())

        binary_label = 1 if majority_label in ANOMALY_GESTURES else 0

        windows.append(window.astype(np.float32))
        bin_labels.append(binary_label)

    return (np.array(windows,    dtype=np.float32),
            np.array(bin_labels, dtype=np.int64))


def collect_all_mat_files(raw_dir):
    """
    Recursively collect all .mat files from raw_dir.
    Handles both flat and subfolder structures:
      - data/raw/emg/file.mat
      - data/raw/emg/DB2_S1/S1_E1_A1.mat
    """
    mat_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.mat'):
                mat_files.append(os.path.join(root, f))
    return mat_files


def process_all_files(raw_dir, output_dir,
                      test_split=0.2, val_split=0.1):
    """
    Process all NinaPro .mat files recursively and save splits.
    """
    all_windows = []
    all_labels  = []

    print(f"Scanning for .mat files in: {raw_dir}")
    mat_files = collect_all_mat_files(raw_dir)

    if not mat_files:
        print("No .mat files found.")
        print("Place NinaPro DB2 .mat files in data/raw/emg/")
        return {}

    print(f"Found {len(mat_files)} .mat files\n")

    for mat_path in tqdm(mat_files, desc="Processing EMG files"):
        try:
            emg, labels = load_ninapro_file(mat_path)
            if emg is None:
                continue

            # Filter → resample → normalize
            emg = bandpass_filter(emg, fs=ORIG_SAMPLE_RATE)
            emg = resample_signal(emg,
                                   orig_fs=ORIG_SAMPLE_RATE,
                                   target_fs=TARGET_SAMPLE_RATE)

            # Resample labels to match
            new_len       = emg.shape[0]
            label_indices = np.linspace(0, len(labels) - 1,
                                         new_len).astype(int)
            labels_rs     = labels[label_indices]

            emg = normalize_signal(emg)

            windows, bin_labels = create_windows(
                emg, labels_rs,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE
            )

            all_windows.append(windows)
            all_labels.append(bin_labels)

        except Exception as e:
            print(f"  Error: {mat_path}: {e}")
            continue

    if not all_windows:
        print("No windows extracted.")
        return {}

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels,  axis=0)

    print(f"\nTotal windows:  {len(X)}")
    print(f"Normal  (0):    {np.sum(y == 0)}")
    print(f"Anomaly (1):    {np.sum(y == 1)}")
    print(f"Anomaly ratio:  {np.sum(y==1)/len(y)*100:.2f}%")

    # ── Split ──────────────────────────────────────────────────────────────────
    np.random.seed(42)
    indices    = np.random.permutation(len(X))
    test_size  = int(len(X) * test_split)
    val_size   = int(len(X) * val_split)
    train_size = len(X) - test_size - val_size

    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    splits = {
        'train': (X[train_idx], y[train_idx]),
        'val':   (X[val_idx],   y[val_idx]),
        'test':  (X[test_idx],  y[test_idx])
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, (X_split, y_split) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, 'X.npy'), X_split)
        np.save(os.path.join(split_dir, 'y.npy'), y_split)
        print(f"Saved {split_name}: {X_split.shape} → {split_dir}")

    stats = {
        'total':            len(X),
        'normal':           int(np.sum(y == 0)),
        'anomaly':          int(np.sum(y == 1)),
        'train_size':       train_size,
        'val_size':         val_size,
        'test_size':        test_size,
        'window_size':      WINDOW_SIZE,
        'n_channels':       N_CHANNELS,
        'sample_rate':      TARGET_SAMPLE_RATE,
        'files_processed':  len(mat_files)
    }

    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nEMG preprocessing complete.")
    return stats


if __name__ == "__main__":
    stats = process_all_files(
        raw_dir='data/raw/emg',
        output_dir='data/processed/emg',
        test_split=0.2,
        val_split=0.1
    )
    if stats:
        print("\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")