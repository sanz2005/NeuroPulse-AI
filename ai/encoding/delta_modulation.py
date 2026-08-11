"""
Temporal Delta Modulation Spike Encoder for NeuroPulse AI
Primary encoder for ECG and EEG signals.
Fires a spike when signal change exceeds threshold.
"""

import numpy as np
import torch


def delta_modulation_encode(signal: np.ndarray,
                             threshold: float = 0.05,
                             padding: bool = True) -> np.ndarray:
    """
    Encode a continuous signal into spike trains using temporal delta modulation.

    A spike (1) is fired when the signal increases by more than threshold.
    A negative spike (-1) is fired when signal decreases by more than threshold.
    For SNN input we use only positive spikes (0 or 1).

    Args:
        signal:    Input signal [samples] or [channels, samples]
        threshold: Firing threshold (default 0.05)
        padding:   Pad output to same length as input

    Returns:
        spikes:    Binary spike train, same shape as input
    """
    if signal.ndim == 1:
        return _encode_1d(signal, threshold)
    elif signal.ndim == 2:
        # Apply to each channel independently
        spikes = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            spikes[i] = _encode_1d(signal[i], threshold)
        return spikes
    else:
        raise ValueError(f"Expected 1D or 2D signal, got {signal.ndim}D")


def _encode_1d(signal: np.ndarray, threshold: float) -> np.ndarray:
    """
    Delta modulation encoding for a single 1D signal.
    """
    spikes    = np.zeros(len(signal), dtype=np.float32)
    reference = signal[0]  # Running reference level

    for i in range(1, len(signal)):
        delta = signal[i] - reference
        if delta > threshold:
            spikes[i]  = 1.0
            reference  = signal[i]
        elif delta < -threshold:
            spikes[i]  = 0.0  # Negative change — no spike (inhibitory ignored)
            reference  = signal[i]
        # else: no spike, reference unchanged

    return spikes


def delta_modulation_encode_batch(signals: np.ndarray,
                                   threshold: float = 0.05) -> np.ndarray:
    """
    Batch encode signals into spike trains.

    Args:
        signals:   [batch, samples] or [batch, channels, samples]
        threshold: Firing threshold

    Returns:
        spikes:    Same shape as signals, binary
    """
    if signals.ndim == 2:
        # [batch, samples]
        spikes = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            spikes[i] = _encode_1d(signals[i], threshold)
        return spikes

    elif signals.ndim == 3:
        # [batch, channels, samples]
        spikes = np.zeros_like(signals)
        for b in range(signals.shape[0]):
            for c in range(signals.shape[1]):
                spikes[b, c] = _encode_1d(signals[b, c], threshold)
        return spikes

    else:
        raise ValueError(f"Expected 2D or 3D batch, got {signals.ndim}D")


def encode_to_tensor(signals: np.ndarray,
                     threshold: float = 0.05,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Encode signals and return as PyTorch tensor.

    Args:
        signals: numpy array [batch, samples] or [batch, channels, samples]
        threshold: firing threshold
        device: 'cpu' or 'cuda'

    Returns:
        torch.Tensor of spike trains, same shape as input
    """
    spikes = delta_modulation_encode_batch(signals, threshold)
    return torch.FloatTensor(spikes).to(device)


def adaptive_threshold(signal: np.ndarray,
                        base_threshold: float = 0.05,
                        window_size: int = 50) -> np.ndarray:
    """
    Adaptive delta modulation — threshold adjusts based on
    local signal variance. Better for non-stationary biosignals.

    Args:
        signal:          1D signal [samples]
        base_threshold:  Minimum threshold
        window_size:     Window for local variance computation

    Returns:
        spikes: binary spike train [samples]
    """
    spikes    = np.zeros(len(signal), dtype=np.float32)
    reference = signal[0]

    for i in range(1, len(signal)):
        # Compute local std for adaptive threshold
        start = max(0, i - window_size)
        local_std = np.std(signal[start:i]) + 1e-8
        threshold = base_threshold * local_std * 10

        delta = signal[i] - reference
        if abs(delta) > threshold:
            spikes[i]  = 1.0 if delta > 0 else 0.0
            reference  = signal[i]

    return spikes


if __name__ == "__main__":
    # ── Quick test ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Generate a test ECG-like signal
    t      = np.linspace(0, 2, 512)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 5 * t)
    signal = (signal - signal.mean()) / signal.std()

    # Encode
    spikes = delta_modulation_encode(signal, threshold=0.05)

    print(f"Signal shape:  {signal.shape}")
    print(f"Spikes shape:  {spikes.shape}")
    print(f"Spike count:   {int(spikes.sum())}")
    print(f"Sparsity:      {(1 - spikes.mean())*100:.1f}%")

    # Test batch encoding
    batch  = np.stack([signal] * 8)  # [8, 512]
    spikes_batch = delta_modulation_encode_batch(batch, threshold=0.05)
    print(f"\nBatch shape:   {spikes_batch.shape}")

    # Test tensor conversion
    tensor = encode_to_tensor(batch, threshold=0.05)
    print(f"Tensor shape:  {tensor.shape}")
    print(f"Tensor dtype:  {tensor.dtype}")

    print("\nDelta modulation encoding test passed!")