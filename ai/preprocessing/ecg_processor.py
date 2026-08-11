"""
ECG Preprocessing Pipeline for NeuroPulse AI
Dataset: MIT-BIH Arrhythmia Database
Sampling Rate: 360 Hz → resampled to 256 Hz
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

# ── Label mapping ──────────────────────────────────────────────────────────────
# MIT-BIH annotation symbols → binary classification
# 0 = Normal, 1 = Anomaly (Arrhythmia)
LABEL_MAP = {
    'N': 0,  # Normal beat
    'L': 1,  # Left bundle branch block
    'R': 1,  # Right bundle branch block
    'A': 1,  # Atrial premature beat
    'V': 1,  # Premature ventricular contraction
    'F': 1,  # Fusion of ventricular and normal beat
    '/': 1,  # Paced beat
    'f': 1,  # Fusion of paced and normal beat
    'j': 1,  # Nodal (junctional) escape beat
    'E': 1,  # Ventricular escape beat
    'e': 1,  # Atrial escape beat
    'J': 1,  # Nodal premature beat
    'a': 1,  # Aberrated atrial premature beat
    'S': 1,  # Supraventricular premature beat
    'Q': 1,  # Unclassifiable beat
}

# MIT-BIH record numbers (48 recordings)
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5,
                    highcut: float = 45.0, fs: float = 360.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to ECG signal.
    Removes baseline wander (< 0.5 Hz) and high-freq noise (> 45 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per segment.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std


def resample_signal(signal: np.ndarray,
                    orig_fs: float = 360.0,
                    target_fs: float = 256.0) -> np.ndarray:
    """
    Resample ECG from 360 Hz to 256 Hz for unified platform sampling rate.
    """
    num_samples = int(len(signal) * target_fs / orig_fs)
    return resample(signal, num_samples)


def segment_signal(signal: np.ndarray, annotations,
                   window_size: int = 256,
                   fs: float = 256.0) -> tuple:
    """
    Segment ECG into fixed windows around R-peaks.
    Each segment is window_size samples centered on R-peak.

    Returns:
        segments: np.ndarray of shape [N, window_size]
        labels:   np.ndarray of shape [N]
    """
    segments = []
    labels = []
    half = window_size // 2

    # Convert annotation samples to resampled rate
    scale = fs / 360.0
    ann_samples = (annotations.sample * scale).astype(int)

    for i, (sample, symbol) in enumerate(zip(ann_samples, annotations.symbol)):
        # Skip if symbol not in label map
        if symbol not in LABEL_MAP:
            continue

        # Boundary check
        if sample - half < 0 or sample + half > len(signal):
            continue

        segment = signal[sample - half: sample + half]
        segment = normalize_signal(segment)

        segments.append(segment)
        labels.append(LABEL_MAP[symbol])

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_mitbih_record(record_path: str) -> tuple:
    """
    Load a single MIT-BIH record.

    Args:
        record_path: Full path without extension e.g. 'data/raw/ecg/100'

    Returns:
        signal: Raw ECG signal (Lead II, channel 0)
        annotations: wfdb Annotation object
    """
    record = wfdb.rdrecord(record_path, channels=[0])
    annotations = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]  # Lead II
    return signal, annotations


def process_all_records(raw_dir: str, output_dir: str,
                        window_size: int = 256,
                        test_split: float = 0.2,
                        val_split: float = 0.1) -> dict:
    """
    Process all MIT-BIH records and save train/val/test splits.

    Args:
        raw_dir:     Path to raw ECG files e.g. 'data/raw/ecg'
        output_dir:  Path to save processed files e.g. 'data/processed/ecg'
        window_size: Samples per segment (default 256 = 1 second at 256Hz)
        test_split:  Fraction for test set
        val_split:   Fraction for validation set

    Returns:
        stats: Dictionary with dataset statistics
    """
    all_segments = []
    all_labels = []

    print(f"Processing MIT-BIH records from: {raw_dir}")
    print(f"Window size: {window_size} samples ({window_size/256:.2f}s at 256Hz)\n")

    available_records = []
    for rec in MITBIH_RECORDS:
        rec_path = os.path.join(raw_dir, rec)
        if os.path.exists(rec_path + '.dat'):
            available_records.append(rec)

    if not available_records:
        print("No MIT-BIH records found in raw_dir.")
        print("Download instructions:")
        print("  pip install wfdb")
        print("  python -c \"import wfdb; wfdb.dl_database('mitdb', 'data/raw/ecg')\"")
        return {}

    for rec in tqdm(available_records, desc="Processing ECG records"):
        try:
            rec_path = os.path.join(raw_dir, rec)
            signal, annotations = load_mitbih_record(rec_path)

            # Step 1: Bandpass filter (0.5 - 45 Hz)
            signal = bandpass_filter(signal, fs=360.0)

            # Step 2: Resample to 256 Hz
            signal = resample_signal(signal, orig_fs=360.0, target_fs=256.0)

            # Step 3: Segment around R-peaks
            segments, labels = segment_signal(signal, annotations,
                                               window_size=window_size)

            all_segments.append(segments)
            all_labels.append(labels)

        except Exception as e:
            print(f"  Error processing record {rec}: {e}")
            continue

    if not all_segments:
        print("No segments extracted.")
        return {}

    # Concatenate all records
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\nTotal segments: {len(X)}")
    print(f"Normal (0): {np.sum(y == 0)}")
    print(f"Anomaly (1): {np.sum(y == 1)}")
    print(f"Class balance: {np.sum(y==1)/len(y)*100:.1f}% anomaly")

    # ── Train / Val / Test Split ───────────────────────────────────────────────
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    test_size = int(len(X) * test_split)
    val_size = int(len(X) * val_split)
    train_size = len(X) - test_size - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    splits = {
        'train': (X[train_idx], y[train_idx]),
        'val':   (X[val_idx],   y[val_idx]),
        'test':  (X[test_idx],  y[test_idx])
    }

    # ── Save to disk ───────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    for split_name, (X_split, y_split) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, 'X.npy'), X_split)
        np.save(os.path.join(split_dir, 'y.npy'), y_split)
        print(f"Saved {split_name}: {X_split.shape} → {split_dir}")

    stats = {
        'total': len(X),
        'normal': int(np.sum(y == 0)),
        'anomaly': int(np.sum(y == 1)),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'window_size': window_size,
        'sample_rate': 256,
        'records_processed': len(available_records)
    }

    # Save stats
    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nECG preprocessing complete.")
    return stats


def download_mitbih(raw_dir: str = 'data/raw/ecg'):
    """
    Download MIT-BIH Arrhythmia Database from PhysioNet.
    Requires wfdb package.
    """
    print("Downloading MIT-BIH Arrhythmia Database...")
    print("This may take 5-10 minutes depending on connection speed.")
    os.makedirs(raw_dir, exist_ok=True)
    wfdb.dl_database('mitdb', raw_dir)
    print(f"Download complete. Files saved to: {raw_dir}")


if __name__ == "__main__":
    # ── Run preprocessing ──────────────────────────────────────────────────────
    # Step 1: Download dataset (run once)
    # download_mitbih('data/raw/ecg')

    # Step 2: Process all records
    stats = process_all_records(
        raw_dir='data/raw/ecg',
        output_dir='data/processed/ecg',
        window_size=256,
        test_split=0.2,
        val_split=0.1
    )

    if stats:
        print("\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")