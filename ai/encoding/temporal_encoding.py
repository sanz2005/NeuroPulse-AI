"""
Temporal/Latency Spike Encoder for NeuroPulse AI
Encodes signal amplitude as spike timing (latency coding).
Higher amplitude = earlier spike.
Used as auxiliary encoder for feature-rich signal regions.
"""

import numpy as np
import torch


def temporal_encode(signal: np.ndarray,
                    timesteps: int = 100,
                    normalize: bool = True) -> np.ndarray:
    """
    Latency encoding — stronger stimulus fires earlier.

    Each sample in the signal maps to exactly ONE spike
    at a timestep inversely proportional to its amplitude.
    High amplitude → early spike (low timestep index)
    Low amplitude  → late spike (high timestep index)

    Args:
        signal:    1D signal [samples] or 2D [channels, samples]
        timesteps: Total number of timesteps
        normalize: Normalize signal before encoding

    Returns:
        spikes: [timesteps] or [channels, timesteps] binary array
    """
    if normalize:
        signal = _normalize_0_1(signal)

    if signal.ndim == 1:
        return _temporal_encode_1d(signal, timesteps)
    elif signal.ndim == 2:
        spikes = np.zeros((signal.shape[0], timesteps), dtype=np.float32)
        for i in range(signal.shape[0]):
            spikes[i] = _temporal_encode_1d(signal[i], timesteps)
        return spikes
    else:
        raise ValueError(f"Expected 1D or 2D signal, got {signal.ndim}D")


def _normalize_0_1(signal: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    min_val = signal.min()
    max_val = signal.max()
    if max_val - min_val == 0:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - min_val) / (max_val - min_val)).astype(np.float32)


def _temporal_encode_1d(signal: np.ndarray,
                         timesteps: int) -> np.ndarray:
    """
    Temporal/latency encoding for a single 1D signal.

    Maps each sample to a spike time:
      spike_time = (1 - amplitude) * (timesteps - 1)

    Then creates a binary spike train with 1s at spike times.
    """
    spikes = np.zeros(timesteps, dtype=np.float32)

    # Subsample signal to fit timesteps
    if len(signal) > timesteps:
        indices = np.linspace(0, len(signal) - 1, timesteps).astype(int)
        signal  = signal[indices]

    for i, amplitude in enumerate(signal):
        # Map amplitude to spike time (inverse relationship)
        spike_time = int((1.0 - float(amplitude)) * (timesteps - 1))
        spike_time = max(0, min(timesteps - 1, spike_time))
        spikes[spike_time] = 1.0

    return spikes


def temporal_encode_batch(signals: np.ndarray,
                           timesteps: int = 100,
                           normalize: bool = True) -> np.ndarray:
    """
    Batch temporal encoding.

    Args:
        signals:   [batch, samples] or [batch, channels, samples]
        timesteps: Output timesteps
        normalize: Normalize before encoding

    Returns:
        spikes: [batch, timesteps] or [batch, channels, timesteps]
    """
    if signals.ndim == 2:
        # [batch, samples] → [batch, timesteps]
        spikes = np.zeros(
            (signals.shape[0], timesteps), dtype=np.float32
        )
        for i in range(signals.shape[0]):
            sig = _normalize_0_1(signals[i]) if normalize else signals[i]
            spikes[i] = _temporal_encode_1d(sig, timesteps)
        return spikes

    elif signals.ndim == 3:
        # [batch, channels, samples] → [batch, channels, timesteps]
        spikes = np.zeros(
            (signals.shape[0], signals.shape[1], timesteps),
            dtype=np.float32
        )
        for b in range(signals.shape[0]):
            for c in range(signals.shape[1]):
                sig = _normalize_0_1(signals[b, c]) if normalize else signals[b, c]
                spikes[b, c] = _temporal_encode_1d(sig, timesteps)
        return spikes

    else:
        raise ValueError(f"Expected 2D or 3D signals, got {signals.ndim}D")


def encode_to_tensor(signals: np.ndarray,
                     timesteps: int = 100,
                     normalize: bool = True,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Temporal encode and return as PyTorch tensor.

    Args:
        signals:   numpy array
        timesteps: output timesteps
        normalize: normalize before encoding
        device:    'cpu' or 'cuda'

    Returns:
        torch.Tensor of spike trains
    """
    spikes = temporal_encode_batch(signals, timesteps, normalize)
    return torch.FloatTensor(spikes).to(device)


def population_encode(signal: np.ndarray,
                       n_neurons: int = 10,
                       timesteps: int = 100) -> np.ndarray:
    """
    Population temporal encoding.
    Multiple neurons with different preferred stimuli
    encode the signal as a population spike pattern.

    Args:
        signal:    1D normalized signal [samples]
        n_neurons: Number of neurons in population
        timesteps: Output timesteps

    Returns:
        spikes: [n_neurons, timesteps]
    """
    signal    = _normalize_0_1(signal)
    spikes    = np.zeros((n_neurons, timesteps), dtype=np.float32)

    # Each neuron has a preferred amplitude (tuning curve center)
    preferred = np.linspace(0, 1, n_neurons)
    sigma     = 1.0 / n_neurons  # Tuning width

    # Subsample signal
    indices   = np.linspace(0, len(signal) - 1, timesteps).astype(int)
    resampled = signal[indices]

    for n in range(n_neurons):
        for t in range(timesteps):
            # Gaussian tuning curve
            response   = np.exp(
                -((resampled[t] - preferred[n]) ** 2) / (2 * sigma ** 2)
            )
            # Fire if response exceeds threshold
            if response > 0.5:
                spikes[n, t] = 1.0

    return spikes


if __name__ == "__main__":
    # ── Quick test ─────────────────────────────────────────────────────────────
    np.random.seed(42)

    # Test 1D
    signal = np.random.randn(256)
    spikes = temporal_encode(signal, timesteps=100)
    print(f"1D Temporal encoding:")
    print(f"  Input shape:  {signal.shape}")
    print(f"  Output shape: {spikes.shape}")
    print(f"  Spike count:  {int(spikes.sum())}")

    # Test 2D (channels)
    signal_2d = np.random.randn(18, 256)
    spikes_2d = temporal_encode(signal_2d, timesteps=100)
    print(f"\n2D Temporal encoding:")
    print(f"  Input shape:  {signal_2d.shape}")
    print(f"  Output shape: {spikes_2d.shape}")

    # Test batch 2D
    batch  = np.random.randn(32, 256)
    spikes = temporal_encode_batch(batch, timesteps=100)
    print(f"\nBatch 2D Temporal encoding:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {spikes.shape}")

    # Test batch 3D
    batch_3d = np.random.randn(32, 18, 256)
    spikes_3d = temporal_encode_batch(batch_3d, timesteps=100)
    print(f"\nBatch 3D Temporal encoding:")
    print(f"  Input shape:  {batch_3d.shape}")
    print(f"  Output shape: {spikes_3d.shape}")

    # Test population encoding
    signal_1d = np.random.randn(256)
    pop_spikes = population_encode(signal_1d, n_neurons=10, timesteps=100)
    print(f"\nPopulation encoding:")
    print(f"  Output shape: {pop_spikes.shape}")

    # Test tensor
    tensor = encode_to_tensor(batch, timesteps=100)
    print(f"\nTensor: {tensor.shape}, dtype: {tensor.dtype}")

    print("\nTemporal encoding test passed!")