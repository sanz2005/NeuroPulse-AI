"""
EEG Preprocessing Pipeline for NeuroPulse AI
Dataset: CHB-MIT Scalp EEG Database
Sampling Rate: 256 Hz (native, no resampling needed)
Task: Seizure Detection (ictal vs interictal)
"""

import os
import shutil
import numpy as np
import mne
from scipy.signal import butter, filtfilt
import pickle
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 256       # Hz
WINDOW_SIZE = 256       # 1 second window
STEP_SIZE   = 128       # 50% overlap
N_CHANNELS  = 18        # Standard CHB-MIT channels used

# CHB-MIT patient records to process — full dataset (all 24 patients downloaded)
PATIENTS = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07',
            'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14',
            'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21',
            'chb22', 'chb23', 'chb24']


def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 70.0,
                    fs: float = 256.0,
                    order: int = 4) -> np.ndarray:
    """
    Bandpass filter for EEG: 0.5 - 70 Hz.
    Removes DC drift and high-freq muscle artifacts.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Apply filter to each channel
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        filtered[ch] = filtfilt(b, a, signal[ch])
    return filtered


def notch_filter(signal: np.ndarray,
                 notch_freq: float = 50.0,
                 fs: float = 256.0) -> np.ndarray:
    """
    Notch filter to remove 50Hz power line noise.
    """
    nyq = 0.5 * fs
    low = (notch_freq - 1) / nyq
    high = (notch_freq + 1) / nyq
    b, a = butter(4, [low, high], btype='bandstop')
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        filtered[ch] = filtfilt(b, a, signal[ch])
    return filtered


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per channel per window.
    signal shape: [channels, samples]
    """
    normalized = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        mean = np.mean(signal[ch])
        std  = np.std(signal[ch])
        if std == 0:
            normalized[ch] = signal[ch] - mean
        else:
            normalized[ch] = (signal[ch] - mean) / std
    return normalized


def load_edf_file(edf_path: str,
                  target_channels: int = 18) -> tuple:
    """
    Load a CHB-MIT .edf file using MNE.

    Returns:
        data:    np.ndarray [channels, samples]
        sfreq:   float sampling frequency
        is_seizure_file: bool (True if filename contains seizure)
    """
    # Suppress MNE verbose output
    mne.set_log_level('WARNING')

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # Get data as numpy array [channels, samples]
    data, _ = raw[:]

    # Use first target_channels channels
    if data.shape[0] > target_channels:
        data = data[:target_channels, :]
    elif data.shape[0] < target_channels:
        # Pad with zeros if fewer channels
        pad = np.zeros((target_channels - data.shape[0], data.shape[1]))
        data = np.vstack([data, pad])

    return data, sfreq


def parse_seizure_annotations(summary_path: str) -> dict:
    """
    Parse CHB-MIT summary file to get seizure start/end times.

    Args:
        summary_path: Path to chbXX-summary.txt

    Returns:
        seizure_times: dict {filename: [(start_sec, end_sec), ...]}
    """
    seizure_times = {}

    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return seizure_times

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    seizure_starts = []
    seizure_ends   = []

    for line in lines:
        line = line.strip()

        if line.startswith('File Name:'):
            # Save previous file's seizures
            if current_file and seizure_starts:
                seizure_times[current_file] = list(zip(seizure_starts, seizure_ends))
            current_file  = line.split(': ')[1].strip()
            seizure_starts = []
            seizure_ends   = []

        elif 'Seizure Start Time' in line:
            try:
                start = int(line.split(': ')[1].replace('seconds', '').strip())
                seizure_starts.append(start)
            except:
                pass

        elif 'Seizure End Time' in line:
            try:
                end = int(line.split(': ')[1].replace('seconds', '').strip())
                seizure_ends.append(end)
            except:
                pass

    # Save last file
    if current_file and seizure_starts:
        seizure_times[current_file] = list(zip(seizure_starts, seizure_ends))

    return seizure_times


def create_windows(data: np.ndarray,
                   seizure_intervals: list,
                   sfreq: float,
                   window_size: int = 256,
                   step_size: int = 128) -> tuple:
    """
    Slide a window across EEG signal and label each window.

    Args:
        data:              [channels, samples]
        seizure_intervals: list of (start_sec, end_sec) tuples
        sfreq:             sampling frequency
        window_size:       samples per window
        step_size:         step between windows

    Returns:
        windows: [N, channels, window_size]
        labels:  [N] — 0=interictal, 1=ictal(seizure)
    """
    windows = []
    labels  = []
    n_samples = data.shape[1]

    # Convert seizure times to sample indices
    seizure_samples = [
        (int(s * sfreq), int(e * sfreq))
        for s, e in seizure_intervals
    ]

    for start in range(0, n_samples - window_size, step_size):
        end    = start + window_size
        window = data[:, start:end]

        # Normalize window
        window = normalize_signal(window)

        # Label: 1 if window overlaps with any seizure period
        label = 0
        for (sz_start, sz_end) in seizure_samples:
            if start < sz_end and end > sz_start:
                label = 1
                break

        windows.append(window)
        labels.append(label)

    return (np.array(windows, dtype=np.float32),
            np.array(labels,  dtype=np.int64))


def process_patient(patient_dir: str,
                    output_dir: str,
                    patient_id: str) -> tuple:
    """
    Process all EDF files for one CHB-MIT patient.

    Args:
        patient_dir: e.g. 'data/raw/eeg/chb01'
        output_dir:  e.g. 'data/processed/eeg'
        patient_id:  e.g. 'chb01'

    Returns:
        all_windows, all_labels
    """
    summary_path = os.path.join(patient_dir, f'{patient_id}-summary.txt')
    seizure_times = parse_seizure_annotations(summary_path)

    edf_files = [f for f in os.listdir(patient_dir)
                 if f.endswith('.edf')]

    if not edf_files:
        print(f"No EDF files found in {patient_dir}")
        return np.array([]), np.array([])

    all_windows = []
    all_labels  = []

    for edf_file in tqdm(edf_files, desc=f"Processing {patient_id}"):
        edf_path = os.path.join(patient_dir, edf_file)

        try:
            # Load EDF
            data, sfreq = load_edf_file(edf_path)

            # Apply filters
            data = bandpass_filter(data, fs=sfreq)
            data = notch_filter(data, fs=sfreq)

            # Get seizure intervals for this file
            intervals = seizure_times.get(edf_file, [])

            # Create windows
            windows, labels = create_windows(
                data, intervals, sfreq,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE
            )

            all_windows.append(windows)
            all_labels.append(labels)

        except Exception as e:
            print(f"  Error processing {edf_file}: {e}")
            continue

    if not all_windows:
        return np.array([]), np.array([])

    return (np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels,  axis=0))


def process_all_patients(raw_dir: str,
                         output_dir: str,
                         test_split: float = 0.2,
                         val_split:  float = 0.1) -> dict:
    """
    Process all CHB-MIT patients and save train/val/test splits.

    IMPORTANT: This version splits by PATIENT, not by window. Windows are
    extracted with 50% overlap, so adjacent windows are highly correlated;
    splitting by window (as in the original implementation) risks leaking
    near-duplicate windows across train/val/test, which inflates reported
    metrics. Splitting by patient guarantees no two windows from the same
    recording appear in different splits, which is also the clinically
    correct evaluation setup (testing generalization to unseen patients).

    This version also never holds more than one patient's data in RAM at a
    time, writing directly to disk-backed memmap arrays, to avoid the
    out-of-memory crash that occurs when concatenating all patients'
    windows into a single in-memory array.

    Args:
        raw_dir:    'data/raw/eeg'
        output_dir: 'data/processed/eeg'
    """
    print(f"Processing CHB-MIT EEG records from: {raw_dir}\n")

    available_patients = []
    for p in PATIENTS:
        p_dir = os.path.join(raw_dir, p)
        if os.path.exists(p_dir):
            available_patients.append(p)

    if not available_patients:
        print("No CHB-MIT patient directories found.")
        print("Download instructions:")
        print("  python -c \"import wfdb; wfdb.dl_database('chbmit', 'data/raw/eeg', records=['chb01'])\"")
        return {}

    # ── PASS 1: process each patient individually, cache to disk ───────────
    # One patient at a time keeps peak memory low regardless of how many
    # patients are processed in total.
    temp_dir = os.path.join(output_dir, '_patient_cache')
    os.makedirs(temp_dir, exist_ok=True)

    patient_counts = {}
    patient_seizure_counts = {}

    for patient_id in available_patients:
        patient_dir = os.path.join(raw_dir, patient_id)
        windows, labels = process_patient(patient_dir, output_dir, patient_id)

        if len(windows) == 0:
            print(f"  {patient_id}: no windows extracted, skipping.")
            continue

        np.save(os.path.join(temp_dir, f'{patient_id}_X.npy'), windows)
        np.save(os.path.join(temp_dir, f'{patient_id}_y.npy'), labels)
        patient_counts[patient_id] = windows.shape[0]
        patient_seizure_counts[patient_id] = int(np.sum(labels == 1))

        del windows, labels  # free this patient's memory before the next one

    processed_patients = list(patient_counts.keys())
    if not processed_patients:
        print("No windows extracted.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {}

    total_windows_all = sum(patient_counts.values())
    total_seizure_all = sum(patient_seizure_counts.values())
    print(f"\nTotal windows:      {total_windows_all}")
    print(f"Interictal (0):     {total_windows_all - total_seizure_all}")
    print(f"Ictal/Seizure (1):  {total_seizure_all}")
    print(f"Seizure ratio:      {total_seizure_all/total_windows_all*100:.2f}%")

    # ── PATIENT-LEVEL SPLIT ─────────────────────────────────────────────────
    # Stratify by whether a patient contributes any seizure windows at all,
    # so seizure-heavy patients (e.g. chb12, chb15) don't all land in one
    # split, while still guaranteeing no window-level leakage.
    rng = np.random.RandomState(42)
    seizure_patients    = [p for p in processed_patients if patient_seizure_counts[p] > 0]
    nonseizure_patients = [p for p in processed_patients if patient_seizure_counts[p] == 0]
    rng.shuffle(seizure_patients)
    rng.shuffle(nonseizure_patients)

    def _split_list(lst, test_frac, val_frac):
        n = len(lst)
        if n == 0:
            return [], [], []
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac)) if n > 2 else 0
        n_test = min(n_test, n - 1) if n > 1 else 0
        n_val  = min(n_val, n - n_test - 1) if (n - n_test) > 1 else 0
        test  = lst[:n_test]
        val   = lst[n_test:n_test + n_val]
        train = lst[n_test + n_val:]
        return train, val, test

    sz_train,  sz_val,  sz_test  = _split_list(seizure_patients,    test_split, val_split)
    ns_train,  ns_val,  ns_test  = _split_list(nonseizure_patients, test_split, val_split)

    split_patients = {
        'train': sz_train + ns_train,
        'val':   sz_val + ns_val,
        'test':  sz_test + ns_test,
    }

    print("\nPatient-level split (prevents window-overlap leakage):")
    for split_name, plist in split_patients.items():
        n_windows  = sum(patient_counts[p] for p in plist)
        n_seizures = sum(patient_seizure_counts[p] for p in plist)
        print(f"  {split_name:5s}: {len(plist):2d} patients, {n_windows:6d} windows, "
              f"{n_seizures:4d} seizure windows -> {sorted(plist)}")

    # ── PASS 2: write each split directly to a disk-backed memmap array ────
    # Only one patient's array is ever loaded into RAM at a time; the output
    # array itself lives on disk via np.lib.format.open_memmap, so peak
    # memory stays low even for the full 24-patient dataset.
    os.makedirs(output_dir, exist_ok=True)

    overall_stats = {'total': 0, 'interictal': 0, 'ictal': 0}

    for split_name, plist in split_patients.items():
        if not plist:
            print(f"  WARNING: no patients assigned to '{split_name}' split.")
            continue

        total_n = sum(patient_counts[p] for p in plist)
        sample  = np.load(os.path.join(temp_dir, f'{plist[0]}_X.npy'), mmap_mode='r')
        window_shape = sample.shape[1:]
        dtype = sample.dtype
        del sample

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        X_path = os.path.join(split_dir, 'X.npy')
        y_path = os.path.join(split_dir, 'y.npy')

        X_mm = np.lib.format.open_memmap(
            X_path, mode='w+', dtype=dtype, shape=(total_n, *window_shape))
        y_mm = np.lib.format.open_memmap(
            y_path, mode='w+', dtype=np.int64, shape=(total_n,))

        offset = 0
        for p in plist:
            Xp = np.load(os.path.join(temp_dir, f'{p}_X.npy'))
            yp = np.load(os.path.join(temp_dir, f'{p}_y.npy'))
            n = Xp.shape[0]
            X_mm[offset:offset + n] = Xp
            y_mm[offset:offset + n] = yp
            offset += n
            del Xp, yp

        X_mm.flush()
        y_mm.flush()
        print(f"Saved {split_name}: {(total_n, *window_shape)} -> {split_dir}")

        overall_stats['total']      += total_n
        overall_stats['interictal'] += int(np.sum(y_mm == 0))
        overall_stats['ictal']      += int(np.sum(y_mm == 1))
        overall_stats[f'{split_name}_size']     = total_n
        overall_stats[f'{split_name}_patients'] = sorted(plist)

        del X_mm, y_mm

    shutil.rmtree(temp_dir, ignore_errors=True)

    overall_stats['window_size'] = WINDOW_SIZE
    overall_stats['n_channels']  = N_CHANNELS
    overall_stats['sample_rate'] = SAMPLE_RATE
    overall_stats['patients']    = processed_patients

    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(overall_stats, f)

    print(f"\nEEG preprocessing complete.")
    return overall_stats


if __name__ == "__main__":
    stats = process_all_patients(
        raw_dir='data/raw/eeg',
        output_dir='data/processed/eeg',
        test_split=0.2,
        val_split=0.1
    )
    if stats:
        print("\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")