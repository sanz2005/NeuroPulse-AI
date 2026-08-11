"""
Rate Encoding Spike Encoder for NeuroPulse AI
Used primarily for EMG signals.
Spike probability proportional to signal amplitude.
"""

import numpy as np
import torch


def rate_encode(signal: np.ndarray,
                timesteps: int = 100,
                threshold: float = 0.5) -> np.ndarray:
    """
    Rate encoding — higher amplitude = higher spike probability.

    Args:
        signal:    Input signal [samples] or [channels, samples]
                   Values should be normalized between 0 and 1
        timesteps: Number of timesteps for encoding
        threshold: Spike firing threshold

    Returns:
        spikes:    Binary spike train [timesteps] or [channels, timesteps]
    """
    # Normalize to [0, 1]
    signal = _normalize_0_1(signal)

    if signal.ndim == 1:
        return _rate_encode_1d(signal, timesteps, threshold)
    elif signal.ndim == 2:
        spikes = np.zeros((signal.shape[0], timesteps), dtype=np.float32)
        for i in range(signal.shape[0]):
            spikes[i] = _rate_encode_1d(signal[i], timesteps, threshold)
        return spikes
    else:
        raise ValueError(f"Expected 1D or 2D signal, got {signal.ndim}D")


def _normalize_0_1(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [0, 1] range.
    """
    min_val = signal.min()
    max_val = signal.max()
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)


def _rate_encode_1d(signal: np.ndarray,
                    timesteps: int,
                    threshold: float) -> np.ndarray:
    """
    Rate encode a single 1D signal.
    Resamples signal to timesteps length,
    then fires spike based on amplitude vs random threshold.
    """
    # Resample signal to timesteps
    indices     = np.linspace(0, len(signal) - 1, timesteps).astype(int)
    resampled   = signal[indices]

    # Generate random numbers and compare with signal amplitude
    random_vals = np.random.uniform(0, 1, timesteps)
    spikes      = (resampled > random_vals).astype(np.float32)

    return spikes


def rate_encode_batch(signals: np.ndarray,
                      timesteps: int = 100,
                      threshold: float = 0.5) -> np.ndarray:
    """
    Batch rate encoding.

    Args:
        signals:   [batch, samples] or [batch, channels, samples]
        timesteps: Output timesteps
        threshold: Firing threshold

    Returns:
        spikes: [batch, timesteps] or [batch, channels, timesteps]
    """
    if signals.ndim == 2:
        # [batch, samples] → [batch, timesteps]
        spikes = np.zeros((signals.shape[0], timesteps), dtype=np.float32)
        for i in range(signals.shape[0]):
            spikes[i] = _rate_encode_1d(
                _normalize_0_1(signals[i]), timesteps, threshold
            )
        return spikes

    elif signals.ndim == 3:
        # [batch, channels, samples] → [batch, channels, timesteps]
        spikes = np.zeros(
            (signals.shape[0], signals.shape[1], timesteps),
            dtype=np.float32
        )
        for b in range(signals.shape[0]):
            for c in range(signals.shape[1]):
                spikes[b, c] = _rate_encode_1d(
                    _normalize_0_1(signals[b, c]), timesteps, threshold
                )
        return spikes

    else:
        raise ValueError(f"Expected 2D or 3D batch, got {signals.ndim}D")


def encode_to_tensor(signals: np.ndarray,
                     timesteps: int = 100,
                     threshold: float = 0.5,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Rate encode and return as PyTorch tensor.

    Args:
        signals:   numpy array
        timesteps: output timesteps
        threshold: firing threshold
        device:    'cpu' or 'cuda'

    Returns:
        torch.Tensor of spike trains
    """
    spikes = rate_encode_batch(signals, timesteps, threshold)
    return torch.FloatTensor(spikes).to(device)


def poisson_encode(signal: np.ndarray,
                   timesteps: int = 100,
                   max_rate: float = 100.0,
                   dt: float = 0.001) -> np.ndarray:
    """
    Poisson rate encoding — biologically realistic version.
    Spike probability at each timestep follows Poisson process.

    Args:
        signal:    Normalized signal [0, 1]
        timesteps: Number of timesteps
        max_rate:  Maximum firing rate in Hz
        dt:        Timestep duration in seconds

    Returns:
        spikes: binary spike train [timesteps]
    """
    signal    = _normalize_0_1(signal)
    indices   = np.linspace(0, len(signal) - 1, timesteps).astype(int)
    resampled = signal[indices]

    # Firing probability = rate * dt
    firing_prob = resampled * max_rate * dt
    firing_prob = np.clip(firing_prob, 0, 1)

    # Sample spikes from Poisson process
    spikes = (np.random.uniform(0, 1, timesteps) < firing_prob).astype(np.float32)
    return spikes


def poisson_encode_batch(signals: np.ndarray,
                          timesteps: int = 100,
                          max_rate: float = 100.0,
                          dt: float = 0.001) -> np.ndarray:
    """
    Batch Poisson encoding.

    Args:
        signals: [batch, samples] or [batch, channels, samples]

    Returns:
        spikes: same batch dims, last dim = timesteps
    """
    if signals.ndim == 2:
        spikes = np.zeros((signals.shape[0], timesteps), dtype=np.float32)
        for i in range(signals.shape[0]):
            spikes[i] = poisson_encode(signals[i], timesteps, max_rate, dt)
        return spikes

    elif signals.ndim == 3:
        spikes = np.zeros(
            (signals.shape[0], signals.shape[1], timesteps),
            dtype=np.float32
        )
        for b in range(signals.shape[0]):
            for c in range(signals.shape[1]):
                spikes[b, c] = poisson_encode(
                    signals[b, c], timesteps, max_rate, dt
                )
        return spikes
    else:
        raise ValueError(f"Expected 2D or 3D signals, got {signals.ndim}D")


if __name__ == "__main__":
    # ── Quick test ─────────────────────────────────────────────────────────────
    np.random.seed(42)

    # Test 1D
    signal = np.random.randn(256)
    spikes = rate_encode(signal, timesteps=100)
    print(f"1D Rate encoding:")
    print(f"  Input shape:  {signal.shape}")
    print(f"  Output shape: {spikes.shape}")
    print(f"  Spike count:  {int(spikes.sum())}")
    print(f"  Sparsity:     {(1 - spikes.mean())*100:.1f}%")

    # Test batch 2D
    batch  = np.random.randn(32, 256)
    spikes = rate_encode_batch(batch, timesteps=100)
    print(f"\nBatch 2D Rate encoding:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {spikes.shape}")

    # Test batch 3D (EMG: batch x channels x samples)
    batch_3d = np.random.randn(32, 12, 256)
    spikes_3d = rate_encode_batch(batch_3d, timesteps=100)
    print(f"\nBatch 3D Rate encoding:")
    print(f"  Input shape:  {batch_3d.shape}")
    print(f"  Output shape: {spikes_3d.shape}")

    # Test Poisson encoding
    spikes_p = poisson_encode_batch(batch, timesteps=100)
    print(f"\nPoisson encoding:")
    print(f"  Output shape: {spikes_p.shape}")
    print(f"  Avg sparsity: {(1 - spikes_p.mean())*100:.1f}%")

    # Test tensor conversion
    tensor = encode_to_tensor(batch, timesteps=100)
    print(f"\nTensor output: {tensor.shape}, dtype: {tensor.dtype}")

    print("\nRate encoding test passed!")