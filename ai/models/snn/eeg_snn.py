"""
EEG Spiking Neural Network for NeuroPulse AI
Dataset: CHB-MIT Scalp EEG
Task: Seizure Detection (Ictal vs Interictal)
Architecture: 4-layer Adaptive LIF SNN
Input: [batch, channels=18, timesteps=256] spike trains
Output: [batch, 2] class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.snn.lif_neuron import LIFLayer, AdaptiveLIFNeuron
from ai.models.snn.surrogate import atan_spike


class EEGSpatialEncoder(nn.Module):
    """
    Spatial encoder for multi-channel EEG.
    Combines information across EEG channels using
    learned spatial weights before temporal SNN processing.

    Args:
        n_channels:  Number of EEG channels (18)
        n_features:  Output spatial features
    """

    def __init__(self, n_channels: int = 18, n_features: int = 64):
        super(EEGSpatialEncoder, self).__init__()

        self.spatial = nn.Sequential(
            nn.Linear(n_channels, n_features),
            nn.BatchNorm1d(n_features),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, timesteps]
        Returns:
            [batch, timesteps, n_features]
        """
        batch, channels, timesteps = x.shape
        # Reshape to apply spatial mixing per timestep
        x = x.permute(0, 2, 1)          # [batch, timesteps, channels]
        x = x.reshape(-1, channels)      # [batch*timesteps, channels]
        x = self.spatial(x)              # [batch*timesteps, n_features]
        x = x.reshape(batch, timesteps, -1)  # [batch, timesteps, n_features]
        return x


class EEGSNN(nn.Module):
    """
    4-layer Adaptive LIF SNN for EEG seizure detection.

    Architecture:
        Spatial Encoder (18 ch → 64) →
        Adaptive LIF Layer 1 (64 → 128) →
        Adaptive LIF Layer 2 (128 → 64) →
        Adaptive LIF Layer 3 (64 → 32) →
        Readout (32 → 2)

    Uses Adaptive LIF neurons which adjust threshold
    dynamically — better suited for EEG burst patterns.

    Args:
        n_channels:  EEG channels (18)
        input_size:  Spatial feature size after encoding
        hidden1:     Hidden layer 1 size
        hidden2:     Hidden layer 2 size
        hidden3:     Hidden layer 3 size
        n_classes:   Output classes (2: normal/seizure)
        timesteps:   SNN simulation timesteps
        beta:        Membrane decay
        threshold:   Base spike threshold
        dropout:     Dropout rate
    """

    def __init__(self,
                 n_channels: int = 18,
                 input_size: int = 64,
                 hidden1: int = 128,
                 hidden2: int = 64,
                 hidden3: int = 32,
                 n_classes: int = 2,
                 timesteps: int = 64,
                 beta: float = 0.9,
                 threshold: float = 1.0,
                 dropout: float = 0.2):
        super(EEGSNN, self).__init__()

        self.timesteps  = timesteps
        self.input_size = input_size
        self.n_channels = n_channels

        # Spatial encoder — combines EEG channels
        self.spatial_encoder = EEGSpatialEncoder(
            n_channels=n_channels,
            n_features=input_size
        )

        # Adaptive LIF layers — better for EEG dynamics
        self.layer1 = LIFLayer(input_size, hidden1,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout,
                                adaptive=True)

        self.layer2 = LIFLayer(hidden1, hidden2,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout,
                                adaptive=True)

        self.layer3 = LIFLayer(hidden2, hidden3,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout,
                                adaptive=True)

        self.layer4 = LIFLayer(hidden3, hidden3,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout,
                                adaptive=True)

        # Readout
        self.readout = nn.Linear(hidden3, n_classes)

    def reset_states(self):
        """Reset all adaptive LIF membrane states."""
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.layer3.reset_state()
        self.layer4.reset_state()

    def forward(self, x: torch.Tensor, record_spikes: bool = False) -> tuple:
        """
        Forward pass over EEG timesteps.

        Args:
            x: [batch, channels, window_size]
            record_spikes: if True, also records per-timestep, per-layer
                mean spike rates for diagnostic/visualization purposes.
                This is OFF by default because calling .item() every
                timestep forces a blocking GPU->CPU sync 64 times per
                layer per batch, which was the dominant cause of slow
                training. Only enable this for standalone inspection of
                a single model / small batch, never during training.

        Returns:
            logits:       [batch, n_classes]
            spike_record: dict with observability metrics
        """
        batch_size = x.shape[0]
        self.reset_states()

        # Spatial encoding — mix channels
        # x: [batch, channels, window_size] → [batch, window_size, input_size]
        x_spatial = self.spatial_encoder(x)

        # Accumulate spikes
        spike_accum1 = torch.zeros(batch_size, self.layer1.linear.out_features).to(x.device)
        spike_accum2 = torch.zeros(batch_size, self.layer2.linear.out_features).to(x.device)
        spike_accum3 = torch.zeros(batch_size, self.layer3.linear.out_features).to(x.device)
        spike_accum4 = torch.zeros(batch_size, self.layer4.linear.out_features).to(x.device)

        spike_record = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
        }

        # Process each timestep
        n_timesteps = min(self.timesteps, x_spatial.shape[1])

        for t in range(n_timesteps):
            x_t = x_spatial[:, t, :]  # [batch, input_size]

            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)
            s3, _ = self.layer3(s2)
            s4, _ = self.layer4(s3)

            spike_accum1 += s1
            spike_accum2 += s2
            spike_accum3 += s3
            spike_accum4 += s4

            if record_spikes:
                spike_record['layer1'].append(s1.detach().mean().item())
                spike_record['layer2'].append(s2.detach().mean().item())
                spike_record['layer3'].append(s3.detach().mean().item())
                spike_record['layer4'].append(s4.detach().mean().item())

        # Readout
        logits = self.readout(spike_accum4 / n_timesteps)

        spike_record['total_spikes'] = (
            spike_accum1.sum() + spike_accum2.sum() +
            spike_accum3.sum() + spike_accum4.sum()
        ).item()
        spike_record['sparsity'] = 1.0 - (
            spike_accum4.mean() / n_timesteps
        ).item()

        return logits, spike_record

    def get_spike_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get temporal spike attribution map for XAI.

        Args:
            x: [batch, channels, window_size]

        Returns:
            attribution: [batch, timesteps]
        """
        batch_size = x.shape[0]
        self.reset_states()

        x_spatial    = self.spatial_encoder(x)
        attributions = []
        n_timesteps  = min(self.timesteps, x_spatial.shape[1])

        for t in range(n_timesteps):
            x_t = x_spatial[:, t, :]
            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)
            s3, _ = self.layer3(s2)
            s4, _ = self.layer4(s3)
            attributions.append(s4.mean(dim=1, keepdim=True))

        return torch.cat(attributions, dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    n_channels = 18
    window_size = 256

    print("Testing EEG SNN Model...")
    model = EEGSNN(
        n_channels=n_channels,
        input_size=64,
        hidden1=128,
        hidden2=64,
        hidden3=32,
        n_classes=2,
        timesteps=64,
        beta=0.9,
        threshold=1.0,
        dropout=0.2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    x = torch.randn(batch_size, n_channels, window_size)
    logits, spike_record = model(x, record_spikes=True)

    print(f"  Input shape:      {x.shape}")
    print(f"  Output shape:     {logits.shape}")
    print(f"  Total spikes:     {spike_record['total_spikes']:.0f}")
    print(f"  Sparsity:         {spike_record['sparsity']*100:.1f}%")

    attribution = model.get_spike_attribution(x)
    print(f"  Attribution shape: {attribution.shape}")

    labels = torch.randint(0, 2, (batch_size,))
    loss   = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"  Loss:             {loss.item():.4f}")

    print("\nEEG SNN model test passed!")