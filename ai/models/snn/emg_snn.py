"""
EMG Spiking Neural Network for NeuroPulse AI
Dataset: NinaPro DB2
Task: Muscle Anomaly Detection (Normal vs Anomaly)
Architecture: 2-layer LIF SNN with channel attention
Input: [batch, channels=12, window_size=256]
Output: [batch, 2] class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.snn.lif_neuron import LIFLayer, LIFNeuron
from ai.models.snn.surrogate import atan_spike


class EMGChannelAttention(nn.Module):
    """
    Channel attention module for EMG.
    Learns which of the 12 EMG channels are most
    relevant for anomaly detection.

    Args:
        n_channels: Number of EMG channels (12)
        reduction:  Reduction ratio for bottleneck
    """

    def __init__(self, n_channels: int = 12, reduction: int = 4):
        super(EMGChannelAttention, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            x: [batch, channels, window_size] — channel weighted
        """
        # Global average pooling across time
        avg = x.mean(dim=2)           # [batch, channels]
        weights = self.attention(avg)  # [batch, channels]
        weights = weights.unsqueeze(2) # [batch, channels, 1]
        return x * weights             # broadcast multiply


class EMGTemporalEncoder(nn.Module):
    """
    Temporal encoder — flattens channel x time into feature vector.
    Uses 1D convolution to extract local temporal patterns
    before feeding into SNN layers.

    Args:
        n_channels:  EMG channels (12)
        window_size: Samples per window (256)
        out_features: Feature vector size
    """

    def __init__(self,
                 n_channels: int = 12,
                 window_size: int = 256,
                 out_features: int = 128):
        super(EMGTemporalEncoder, self).__init__()

        self.conv = nn.Sequential(
            # Local temporal patterns
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),  # 256 → 64

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),  # 64 → 16
        )

        # Calculate conv output size
        conv_out = 64 * (window_size // 16)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, out_features),
            nn.BatchNorm1d(out_features),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            features: [batch, out_features]
        """
        x = self.conv(x)                    # [batch, 64, 16]
        x = x.flatten(1)                    # [batch, 64*16]
        x = self.fc(x)                      # [batch, out_features]
        return x


class EMGSNN(nn.Module):
    """
    2-layer LIF SNN for EMG muscle anomaly detection.

    Architecture:
        Channel Attention →
        Temporal Encoder (Conv1D → FC) →
        LIF Layer 1 (128 → 64) →
        LIF Layer 2 (64 → 32) →
        Readout (32 → 2)

    Args:
        n_channels:   EMG channels (12)
        window_size:  Samples per window (256)
        hidden1:      Hidden layer 1 size
        hidden2:      Hidden layer 2 size
        n_classes:    Output classes (2)
        timesteps:    SNN simulation timesteps
        beta:         Membrane decay
        threshold:    Spike threshold
        dropout:      Dropout rate
    """

    def __init__(self,
                 n_channels: int = 12,
                 window_size: int = 256,
                 hidden1: int = 64,
                 hidden2: int = 32,
                 n_classes: int = 2,
                 timesteps: int = 32,
                 beta: float = 0.9,
                 threshold: float = 1.0,
                 dropout: float = 0.2):
        super(EMGSNN, self).__init__()

        self.timesteps   = timesteps
        self.n_channels  = n_channels
        self.window_size = window_size

        # Channel attention
        self.channel_attention = EMGChannelAttention(
            n_channels=n_channels
        )

        # Temporal encoder
        self.temporal_encoder = EMGTemporalEncoder(
            n_channels=n_channels,
            window_size=window_size,
            out_features=128
        )

        # SNN layers
        self.layer1 = LIFLayer(128, hidden1,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout)

        self.layer2 = LIFLayer(hidden1, hidden2,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout)

        # Readout
        self.readout = nn.Linear(hidden2, n_classes)

    def reset_states(self):
        """Reset LIF membrane states."""
        self.layer1.reset_state()
        self.layer2.reset_state()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: [batch, channels, window_size]

        Returns:
            logits:       [batch, n_classes]
            spike_record: observability metrics dict
        """
        batch_size = x.shape[0]
        self.reset_states()

        # Channel attention
        x = self.channel_attention(x)

        # Temporal encoding — extract features once
        features = self.temporal_encoder(x)  # [batch, 128]

        # Accumulate spikes over timesteps
        spike_accum1 = torch.zeros(
            batch_size, self.layer1.linear.out_features
        ).to(x.device)
        spike_accum2 = torch.zeros(
            batch_size, self.layer2.linear.out_features
        ).to(x.device)

        spike_record = {
            'layer1': [],
            'layer2': [],
        }

        # Add small noise per timestep to simulate temporal dynamics
        for t in range(self.timesteps):
            noise = torch.randn_like(features) * 0.01
            x_t   = features + noise

            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)

            spike_accum1 += s1
            spike_accum2 += s2

            spike_record['layer1'].append(s1.detach().mean().item())
            spike_record['layer2'].append(s2.detach().mean().item())

        # Readout
        logits = self.readout(spike_accum2 / self.timesteps)

        spike_record['total_spikes'] = (
            spike_accum1.sum() + spike_accum2.sum()
        ).item()
        spike_record['sparsity'] = 1.0 - (
            spike_accum2.mean() / self.timesteps
        ).item()

        return logits, spike_record

    def get_spike_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get temporal spike attribution for XAI.

        Args:
            x: [batch, channels, window_size]

        Returns:
            attribution: [batch, timesteps]
        """
        batch_size = x.shape[0]
        self.reset_states()

        x        = self.channel_attention(x)
        features = self.temporal_encoder(x)

        attributions = []

        for t in range(self.timesteps):
            noise = torch.randn_like(features) * 0.01
            x_t   = features + noise
            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)
            attributions.append(s2.mean(dim=1, keepdim=True))

        return torch.cat(attributions, dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size  = 4
    n_channels  = 12
    window_size = 256

    print("Testing EMG SNN Model...")
    model = EMGSNN(
        n_channels=n_channels,
        window_size=window_size,
        hidden1=64,
        hidden2=32,
        n_classes=2,
        timesteps=32,
        beta=0.9,
        threshold=1.0,
        dropout=0.2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    x = torch.randn(batch_size, n_channels, window_size)
    logits, spike_record = model(x)

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

    print("\nEMG SNN model test passed!")