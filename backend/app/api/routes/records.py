"""
Raw Record Loader for NeuroPulse AI
Loads actual raw dataset records for interactive browsing.
ECG: MIT-BIH .dat/.hea files via wfdb
EEG: CHB-MIT .edf files via mne
EMG: NinaPro .mat files via scipy
"""

import sys
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.signal import butter, filtfilt, resample

sys.path.append('.')

router = APIRouter()

# ── Data cache ─────────────────────────────────────────────────────────────────
_record_cache = {}


# ══════════════════════════════════════════════════════════════════════════════
# ECG — MIT-BIH Raw Records
# ══════════════════════════════════════════════════════════════════════════════

MIT_BIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Clinical descriptions per record
MIT_BIH_INFO = {
    '100': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '101': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '102': {'description': 'Paced rhythm',                     'has_anomaly': True},
    '103': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '104': {'description': 'Paced rhythm with PVCs',           'has_anomaly': True},
    '105': {'description': 'Sinus tachycardia with PVCs',      'has_anomaly': True},
    '106': {'description': 'Premature ventricular contractions','has_anomaly': True},
    '107': {'description': 'Ventricular bigeminy',             'has_anomaly': True},
    '108': {'description': 'ST changes with PVCs',             'has_anomaly': True},
    '109': {'description': 'Ventricular bigeminy',             'has_anomaly': True},
    '111': {'description': 'Normal with bundle branch block',  'has_anomaly': True},
    '112': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '113': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '114': {'description': 'PVCs with compensatory pauses',    'has_anomaly': True},
    '115': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '116': {'description': 'PVCs, fusion beats',               'has_anomaly': True},
    '117': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '118': {'description': 'LBBB with PACs',                   'has_anomaly': True},
    '119': {'description': 'Ventricular bigeminy',             'has_anomaly': True},
    '200': {'description': 'PVCs, PACs',                       'has_anomaly': True},
    '201': {'description': 'PACs, PVCs',                       'has_anomaly': True},
    '202': {'description': 'PVCs',                             'has_anomaly': True},
    '203': {'description': 'Frequent PVCs, runs of VT',        'has_anomaly': True},
    '205': {'description': 'Ventricular bigeminy, PVCs',       'has_anomaly': True},
    '207': {'description': 'VT, ventricular flutter/fibrillation','has_anomaly': True},
    '208': {'description': 'PVCs, PACs, aberrant conduction',  'has_anomaly': True},
    '209': {'description': 'PACs',                             'has_anomaly': True},
    '210': {'description': 'PVCs, aberrant conduction',        'has_anomaly': True},
    '212': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '213': {'description': 'Frequent PVCs, bigeminy',          'has_anomaly': True},
    '214': {'description': 'LBBB',                             'has_anomaly': True},
    '215': {'description': 'Normal with PVCs',                 'has_anomaly': True},
    '217': {'description': 'Paced rhythm',                     'has_anomaly': True},
    '219': {'description': 'Normal with PVCs',                 'has_anomaly': True},
    '220': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '221': {'description': 'Ventricular tachycardia runs',     'has_anomaly': True},
    '222': {'description': 'PACs',                             'has_anomaly': True},
    '223': {'description': 'PVCs, bigeminy',                   'has_anomaly': True},
    '228': {'description': 'PVCs, bigeminy',                   'has_anomaly': True},
    '230': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '231': {'description': 'RBBB',                             'has_anomaly': True},
    '232': {'description': 'Normal sinus rhythm',              'has_anomaly': False},
    '233': {'description': 'Frequent PVCs, bigeminy',          'has_anomaly': True},
    '234': {'description': 'Normal with LBBB',                 'has_anomaly': True},
}


def bandpass_filter(signal, lowcut=0.5, highcut=45.0,
                     fs=360.0, order=4):
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


@router.get("/ecg/list")
async def list_ecg_records():
    """List all available MIT-BIH ECG records."""
    raw_dir = 'data/raw/ecg'
    available = []

    for rec in MIT_BIH_RECORDS:
        dat_path = os.path.join(raw_dir, f'{rec}.dat')
        if os.path.exists(dat_path):
            info = MIT_BIH_INFO.get(rec, {
                'description': 'ECG Record',
                'has_anomaly': False
            })
            available.append({
                'record_id':   rec,
                'name':        f'MIT-BIH Record {rec}',
                'description': info['description'],
                'has_anomaly': info['has_anomaly'],
                'dataset':     'MIT-BIH Arrhythmia Database',
                'source':      'PhysioNet'
            })

    return {
        'modality': 'ecg',
        'total':    len(available),
        'records':  available
    }


@router.get("/ecg/record/{record_id}")
async def get_ecg_record(
    record_id: str,
    duration_sec: float = 10.0
):
    """
    Load a raw MIT-BIH ECG record.
    Returns signal data for the specified duration.
    """
    import wfdb

    cache_key = f'ecg_{record_id}_{duration_sec}'
    if cache_key in _record_cache:
        return _record_cache[cache_key]

    raw_dir  = 'data/raw/ecg'
    rec_path = os.path.join(raw_dir, record_id)

    if not os.path.exists(rec_path + '.dat'):
        raise HTTPException(
            status_code=404,
            detail=f"Record {record_id} not found in data/raw/ecg/"
        )

    try:
        # Load record
        record = wfdb.rdrecord(rec_path, channels=[0])
        signal = record.p_signal[:, 0]
        fs     = record.fs  # 360 Hz for MIT-BIH

        # Load annotations
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            annotations = [
                {
                    'sample':  int(s),
                    'symbol':  sym,
                    'time_sec': round(s / fs, 3)
                }
                for s, sym in zip(ann.sample, ann.symbol)
                if sym not in ['+', '~', '|', '[', ']', '!', '"', 'x']
            ]
        except:
            annotations = []

        # Filter signal
        signal_filtered = bandpass_filter(signal, fs=fs)

        # Take requested duration
        n_samples = min(int(duration_sec * fs), len(signal_filtered))
        signal_chunk = signal_filtered[:n_samples]

        # Normalize
        mean = np.mean(signal_chunk)
        std  = np.std(signal_chunk) + 1e-8
        signal_norm = ((signal_chunk - mean) / std).tolist()

        # Get annotations in this range
        anns_in_range = [
            a for a in annotations
            if a['sample'] < n_samples
        ]

        # Anomaly regions (samples with non-normal beats)
        normal_symbols  = {'N', 'L', 'R', 'e', 'j'}
        anomaly_regions = [
            {
                'sample':   a['sample'],
                'time_sec': a['time_sec'],
                'symbol':   a['symbol'],
                'type':     'arrhythmia'
            }
            for a in anns_in_range
            if a['symbol'] not in normal_symbols
        ]

        info = MIT_BIH_INFO.get(record_id, {
            'description': 'ECG Record',
            'has_anomaly': False
        })

        result = {
            'record_id':        record_id,
            'name':             f'MIT-BIH Record {record_id}',
            'description':      info['description'],
            'signal':           signal_norm,
            'sample_rate':      fs,
            'duration_sec':     round(n_samples / fs, 2),
            'n_samples':        n_samples,
            'annotations':      anns_in_range[:100],
            'anomaly_regions':  anomaly_regions[:50],
            'has_anomaly':      len(anomaly_regions) > 0,
            'total_beats':      len(anns_in_range),
            'anomaly_beats':    len(anomaly_regions),
        }

        _record_cache[cache_key] = result
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading record {record_id}: {str(e)}"
        )


@router.get("/ecg/region/{record_id}")
async def get_ecg_region(
    record_id:    str,
    start_sample: int = 0,
    end_sample:   int = 256
):
    """
    Extract a specific region from ECG record for SNN analysis.
    Returns the exact 256-sample window.
    """
    import wfdb

    raw_dir  = 'data/raw/ecg'
    rec_path = os.path.join(raw_dir, record_id)

    if not os.path.exists(rec_path + '.dat'):
        raise HTTPException(status_code=404,
                            detail=f"Record {record_id} not found")

    try:
        record = wfdb.rdrecord(rec_path, channels=[0])
        signal = record.p_signal[:, 0]
        fs     = record.fs

        # Get spike train from encoding
        from ai.encoding.delta_modulation import delta_modulation_encode

        # Ensure window is 256 samples
        window_size = 256
        end_sample  = start_sample + window_size
        end_sample  = min(end_sample, len(signal))
        start_sample = max(0, end_sample - window_size)

        window  = signal[start_sample:end_sample]

        # Filter + normalize
        if len(window) == window_size:
            window = bandpass_filter(
                window, fs=float(fs)
            )
        mean   = np.mean(window)
        std    = np.std(window) + 1e-8
        window = (window - mean) / std

        spike_train = delta_modulation_encode(
            window, threshold=0.05
        )

        # Get annotations in this region
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            region_anns = [
                {'sample': int(s) - start_sample, 'symbol': sym}
                for s, sym in zip(ann.sample, ann.symbol)
                if start_sample <= s < end_sample
            ]
        except:
            region_anns = []

        normal_symbols = {'N', 'L', 'R', 'e', 'j'}
        has_anomaly    = any(
            a['symbol'] not in normal_symbols
            for a in region_anns
        )

        return {
            'record_id':    record_id,
            'start_sample': start_sample,
            'end_sample':   end_sample,
            'signal':       window.tolist(),
            'spike_train':  spike_train.tolist(),
            'spike_count':  int(spike_train.sum()),
            'annotations':  region_anns,
            'has_anomaly':  has_anomaly,
            'window_size':  window_size,
            'sample_rate':  int(fs),
            'time_start_sec': round(start_sample / fs, 3),
            'time_end_sec':   round(end_sample / fs, 3),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting region: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# EEG — CHB-MIT Raw Records
# ══════════════════════════════════════════════════════════════════════════════

CHB_PATIENTS = {
    'chb01': {
        'name':        'CHB-MIT Patient 01',
        'age':         11,
        'gender':      'F',
        'description': 'Focal seizures, right hemisphere',
    },
    'chb02': {
        'name':        'CHB-MIT Patient 02',
        'age':         11,
        'gender':      'M',
        'description': 'Generalized seizures',
    },
    'chb03': {
        'name':        'CHB-MIT Patient 03',
        'age':         14,
        'gender':      'F',
        'description': 'Focal seizures',
    },
}


@router.get("/eeg/list")
async def list_eeg_records():
    """List all available CHB-MIT EEG records."""
    raw_dir   = 'data/raw/eeg'
    available = []

    for patient_id, info in CHB_PATIENTS.items():
        patient_dir = os.path.join(raw_dir, patient_id)
        if not os.path.exists(patient_dir):
            continue

        edf_files = sorted([
            f for f in os.listdir(patient_dir)
            if f.endswith('.edf') and
            not f.endswith('.seizures')
        ])

        for edf_file in edf_files:
            seizure_file = os.path.join(
                patient_dir, edf_file + '.seizures'
            )
            has_seizure = os.path.exists(seizure_file)

            available.append({
                'record_id':   f"{patient_id}/{edf_file}",
                'patient_id':  patient_id,
                'filename':    edf_file,
                'name':        f"{info['name']} — {edf_file}",
                'description': info['description'],
                'has_seizure': has_seizure,
                'dataset':     'CHB-MIT Scalp EEG Database',
            })

    return {
        'modality': 'eeg',
        'total':    len(available),
        'records':  available
    }


@router.get("/eeg/record/{patient_id}/{filename}")
async def get_eeg_record(
    patient_id:   str,
    filename:     str,
    duration_sec: float = 30.0
):
    """Load a raw CHB-MIT EEG file."""
    import mne
    mne.set_log_level('WARNING')

    edf_path = os.path.join(
        'data/raw/eeg', patient_id, filename
    )

    if not os.path.exists(edf_path):
        raise HTTPException(
            status_code=404,
            detail=f"EEG file not found: {edf_path}"
        )

    cache_key = f'eeg_{patient_id}_{filename}_{duration_sec}'
    if cache_key in _record_cache:
        return _record_cache[cache_key]

    try:
        raw  = mne.io.read_raw_edf(
            edf_path, preload=True, verbose=False
        )
        sfreq = raw.info['sfreq']
        data, times = raw[:18, :]

        n_samples = min(
            int(duration_sec * sfreq), data.shape[1]
        )
        data_chunk = data[:, :n_samples]

        # Normalize each channel
        normalized = []
        for ch in range(min(18, data_chunk.shape[0])):
            ch_data = data_chunk[ch]
            mean    = np.mean(ch_data)
            std     = np.std(ch_data) + 1e-12
            normalized.append(
                ((ch_data - mean) / std).tolist()
            )

        # Check for seizure annotations
        seizure_file = edf_path + '.seizures'
        seizure_regions = []
        if os.path.exists(seizure_file):
            with open(seizure_file, 'r') as f:
                content = f.read()
            # Parse seizure times
            import re
            starts = re.findall(
                r'Seizure Start Time:\s+(\d+)', content
            )
            ends   = re.findall(
                r'Seizure End Time:\s+(\d+)', content
            )
            for s, e in zip(starts, ends):
                start_s = int(s)
                end_s   = int(e)
                if start_s < duration_sec:
                    seizure_regions.append({
                        'start_sec':    start_s,
                        'end_sec':      end_s,
                        'start_sample': int(start_s * sfreq),
                        'end_sample':   int(end_s * sfreq),
                    })

        result = {
            'record_id':       f'{patient_id}/{filename}',
            'patient_id':      patient_id,
            'filename':        filename,
            'channels':        normalized,
            'n_channels':      len(normalized),
            'sample_rate':     int(sfreq),
            'n_samples':       n_samples,
            'duration_sec':    round(n_samples / sfreq, 2),
            'seizure_regions': seizure_regions,
            'has_seizure':     len(seizure_regions) > 0,
            'channel_names':   raw.ch_names[:18],
        }

        _record_cache[cache_key] = result
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading EEG: {str(e)}"
        )


@router.get("/eeg/region/{patient_id}/{filename}")
async def get_eeg_region(
    patient_id:   str,
    filename:     str,
    start_sample: int = 0
):
    """Extract 256-sample EEG window for SNN analysis."""
    import mne
    mne.set_log_level('WARNING')

    edf_path = os.path.join(
        'data/raw/eeg', patient_id, filename
    )

    if not os.path.exists(edf_path):
        raise HTTPException(status_code=404,
                            detail="EEG file not found")

    try:
        from ai.encoding.delta_modulation import (
            delta_modulation_encode
        )

        raw   = mne.io.read_raw_edf(
            edf_path, preload=True, verbose=False
        )
        sfreq = raw.info['sfreq']
        data, _ = raw[:18, :]

        window_size  = 256
        end_sample   = start_sample + window_size
        end_sample   = min(end_sample, data.shape[1])
        start_sample = max(0, end_sample - window_size)

        window = data[:18, start_sample:end_sample]

        # Normalize
        normalized = np.zeros_like(window)
        for ch in range(window.shape[0]):
            mean = np.mean(window[ch])
            std  = np.std(window[ch]) + 1e-12
            normalized[ch] = (window[ch] - mean) / std

        # Spike train from channel 0
        spike_train = delta_modulation_encode(
            normalized[0], threshold=0.05
        )

        # Check seizure overlap
        seizure_file = edf_path + '.seizures'
        in_seizure   = False
        if os.path.exists(seizure_file):
            with open(seizure_file, 'r') as f:
                content = f.read()
            import re
            starts = re.findall(
                r'Seizure Start Time:\s+(\d+)', content
            )
            ends   = re.findall(
                r'Seizure End Time:\s+(\d+)', content
            )
            for s, e in zip(starts, ends):
                sz_start = int(s) * int(sfreq)
                sz_end   = int(e) * int(sfreq)
                if (start_sample < sz_end and
                        end_sample > sz_start):
                    in_seizure = True
                    break

        return {
            'record_id':    f'{patient_id}/{filename}',
            'start_sample': start_sample,
            'end_sample':   end_sample,
            'signal':       normalized[0].tolist(),
            'all_channels': normalized.tolist(),
            'spike_train':  spike_train.tolist(),
            'spike_count':  int(spike_train.sum()),
            'has_seizure':  in_seizure,
            'window_size':  window_size,
            'sample_rate':  int(sfreq),
            'time_start_sec': round(start_sample / sfreq, 3),
            'time_end_sec':   round(end_sample / sfreq, 3),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting EEG region: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# EMG — NinaPro Raw Records
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/emg/list")
async def list_emg_records():
    """List all available NinaPro EMG files."""
    raw_dir   = 'data/raw/emg'
    available = []

    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.mat'):
                subject = os.path.basename(root)
                available.append({
                    'record_id':   f"{subject}/{f}",
                    'subject':     subject,
                    'filename':    f,
                    'name':        f"NinaPro {subject} — {f}",
                    'description': 'EMG muscle activation recording',
                    'dataset':     'NinaPro DB2',
                })

    return {
        'modality': 'emg',
        'total':    len(available),
        'records':  available
    }


@router.get("/emg/record/{subject}/{filename}")
async def get_emg_record(
    subject:      str,
    filename:     str,
    duration_sec: float = 10.0
):
    """Load a raw NinaPro EMG file."""
    from scipy.io import loadmat
    from scipy.signal import butter, filtfilt, resample

    mat_path = os.path.join('data/raw/emg', subject, filename)

    if not os.path.exists(mat_path):
        raise HTTPException(
            status_code=404,
            detail=f"EMG file not found: {mat_path}"
        )

    cache_key = f'emg_{subject}_{filename}_{duration_sec}'
    if cache_key in _record_cache:
        return _record_cache[cache_key]

    try:
        mat = loadmat(mat_path)

        emg = None
        for key in ['emg', 'EMG', 'data']:
            if key in mat:
                emg = mat[key].astype(np.float32)
                break

        labels = None
        for key in ['restimulus', 'stimulus', 'label']:
            if key in mat:
                labels = mat[key].flatten().astype(np.int32)
                break

        if emg is None:
            raise ValueError("No EMG data found in .mat file")

        orig_fs    = 2000
        target_fs  = 256
        n_channels = min(12, emg.shape[1])
        emg        = emg[:, :n_channels]

        # Bandpass filter
        nyq  = 0.5 * orig_fs
        low  = 20 / nyq
        high = min(500 / nyq, 0.99)
        b, a = butter(4, [low, high], btype='band')
        emg_filtered = np.zeros_like(emg)
        for ch in range(n_channels):
            emg_filtered[:, ch] = filtfilt(b, a, emg[:, ch])

        # Resample to 256 Hz
        n_target = int(emg_filtered.shape[0] * target_fs / orig_fs)
        emg_resampled = np.zeros((n_target, n_channels))
        for ch in range(n_channels):
            emg_resampled[:, ch] = resample(
                emg_filtered[:, ch], n_target
            )

        # Take duration
        n_samples = min(
            int(duration_sec * target_fs),
            emg_resampled.shape[0]
        )
        emg_chunk = emg_resampled[:n_samples, :]

        # Normalize
        normalized = []
        for ch in range(n_channels):
            mean = np.mean(emg_chunk[:, ch])
            std  = np.std(emg_chunk[:, ch]) + 1e-8
            normalized.append(
                ((emg_chunk[:, ch] - mean) / std).tolist()
            )

        # Label info
        anomaly_regions = []
        if labels is not None:
            label_resampled = labels[
                np.linspace(0, len(labels)-1, n_samples
            ).astype(int)]
            for i in range(0, n_samples - 10, 10):
                if label_resampled[i] >= 20:
                    anomaly_regions.append({
                        'start_sample': i,
                        'end_sample':   min(i + 256, n_samples),
                        'gesture':      int(label_resampled[i])
                    })

        result = {
            'record_id':       f'{subject}/{filename}',
            'subject':         subject,
            'filename':        filename,
            'channels':        normalized,
            'n_channels':      n_channels,
            'sample_rate':     target_fs,
            'n_samples':       n_samples,
            'duration_sec':    round(n_samples / target_fs, 2),
            'anomaly_regions': anomaly_regions[:20],
            'has_anomaly':     len(anomaly_regions) > 0,
        }

        _record_cache[cache_key] = result
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading EMG: {str(e)}"
        )


@router.get("/emg/region/{subject}/{filename}")
async def get_emg_region(
    subject:      str,
    filename:     str,
    start_sample: int = 0
):
    """Extract 256-sample EMG window for SNN analysis."""
    from scipy.io import loadmat
    from scipy.signal import butter, filtfilt, resample
    from ai.encoding.rate_encoding import rate_encode_batch

    mat_path = os.path.join('data/raw/emg', subject, filename)

    if not os.path.exists(mat_path):
        raise HTTPException(status_code=404,
                            detail="EMG file not found")

    try:
        mat    = loadmat(mat_path)
        emg    = None
        labels = None

        for key in ['emg', 'EMG', 'data']:
            if key in mat:
                emg = mat[key].astype(np.float32)
                break

        for key in ['restimulus', 'stimulus', 'label']:
            if key in mat:
                labels = mat[key].flatten().astype(np.int32)
                break

        if emg is None:
            raise ValueError("No EMG data")

        n_channels = min(12, emg.shape[1])
        emg        = emg[:, :n_channels]

        orig_fs   = 2000
        target_fs = 256

        # Filter
        nyq  = 0.5 * orig_fs
        low  = 20 / nyq
        high = min(500 / nyq, 0.99)
        b, a = butter(4, [low, high], btype='band')
        for ch in range(n_channels):
            emg[:, ch] = filtfilt(b, a, emg[:, ch])

        # Resample
        n_target      = int(emg.shape[0] * target_fs / orig_fs)
        emg_resampled = np.zeros((n_target, n_channels))
        for ch in range(n_channels):
            emg_resampled[:, ch] = resample(emg[:, ch], n_target)

        window_size  = 256
        end_sample   = start_sample + window_size
        end_sample   = min(end_sample, emg_resampled.shape[0])
        start_sample = max(0, end_sample - window_size)

        window = emg_resampled[start_sample:end_sample, :]

        # Normalize + transpose to [channels, samples]
        window_norm = np.zeros_like(window)
        for ch in range(n_channels):
            mean = np.mean(window[:, ch])
            std  = np.std(window[:, ch]) + 1e-8
            window_norm[:, ch] = (window[:, ch] - mean) / std

        window_t = window_norm.T  # [channels, samples]

        # Rate encode channel 0 for display
        spike_train = rate_encode_batch(
            window_norm[:, 0:1].T, timesteps=256
        )[0]

        # Label
        is_anomaly = False
        if labels is not None:
            scale    = target_fs / orig_fs
            l_start  = int(start_sample / scale)
            l_end    = int(end_sample / scale)
            l_end    = min(l_end, len(labels))
            if l_start < len(labels):
                chunk_labels = labels[l_start:l_end]
                is_anomaly   = bool(np.any(chunk_labels >= 20))

        return {
            'record_id':    f'{subject}/{filename}',
            'start_sample': start_sample,
            'end_sample':   end_sample,
            'signal':       window_norm[:, 0].tolist(),
            'all_channels': window_t.tolist(),
            'spike_train':  spike_train.tolist(),
            'spike_count':  int(spike_train.sum()),
            'has_anomaly':  is_anomaly,
            'window_size':  window_size,
            'sample_rate':  target_fs,
            'time_start_sec': round(start_sample / target_fs, 3),
            'time_end_sec':   round(end_sample / target_fs, 3),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting EMG region: {str(e)}"
        )