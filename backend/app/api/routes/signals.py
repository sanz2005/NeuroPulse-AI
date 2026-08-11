"""
Signal API Routes for NeuroPulse AI
Serves preprocessed biosignal windows for streaming.
"""

import sys
import numpy as np
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.append('.')

router = APIRouter()

# Cache loaded data
_signal_cache = {}


def load_signal_data(modality: str) -> tuple:
    """Load test data for streaming simulation."""
    global _signal_cache
    if modality in _signal_cache:
        return _signal_cache[modality]

    try:
        X = np.load(f'data/processed/{modality}/test/X.npy')
        y = np.load(f'data/processed/{modality}/test/y.npy')
        _signal_cache[modality] = (X, y)
        return X, y
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Signal data not found for {modality}: {e}"
        )


class SignalWindow(BaseModel):
    modality:    str
    window_idx:  int
    signal:      List
    label:       int
    is_anomaly:  bool
    sample_rate: int = 256


@router.get("/ecg/{window_idx}", response_model=SignalWindow)
async def get_ecg_window(window_idx: int):
    """Get a specific ECG signal window."""
    X, y = load_signal_data('ecg')
    idx  = window_idx % len(X)
    return SignalWindow(
        modality='ecg',
        window_idx=idx,
        signal=X[idx].tolist(),
        label=int(y[idx]),
        is_anomaly=bool(y[idx] == 1),
        sample_rate=256
    )


@router.get("/eeg/{window_idx}", response_model=SignalWindow)
async def get_eeg_window(window_idx: int):
    """Get a specific EEG signal window."""
    X, y = load_signal_data('eeg')
    idx  = window_idx % len(X)
    return SignalWindow(
        modality='eeg',
        window_idx=idx,
        signal=X[idx].tolist(),
        label=int(y[idx]),
        is_anomaly=bool(y[idx] == 1),
        sample_rate=256
    )


@router.get("/emg/{window_idx}", response_model=SignalWindow)
async def get_emg_window(window_idx: int):
    """Get a specific EMG signal window."""
    X, y = load_signal_data('emg')
    idx  = window_idx % len(X)
    return SignalWindow(
        modality='emg',
        window_idx=idx,
        signal=X[idx].tolist(),
        label=int(y[idx]),
        is_anomaly=bool(y[idx] == 1),
        sample_rate=256
    )


@router.get("/stats")
async def get_signal_stats():
    """Get dataset statistics."""
    stats = {}
    for modality in ['ecg', 'eeg', 'emg']:
        try:
            X, y = load_signal_data(modality)
            stats[modality] = {
                'total_windows': len(X),
                'normal':        int(np.sum(y == 0)),
                'anomaly':       int(np.sum(y == 1)),
                'shape':         list(X.shape)
            }
        except:
            stats[modality] = {'error': 'Data not loaded'}
    return stats