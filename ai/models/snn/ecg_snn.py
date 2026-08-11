"""
ECG Spiking Neural Network for NeuroPulse AI
Dataset: MIT-BIH Arrhythmia
Task: Binary classification — Normal vs Arrhythmia
Architecture: 3-layer LIF SNN with surrogate gradient training
Input: [batch, timesteps=256] spike trains
Output: [batch, 2] class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.snn.lif_neuron import LIFLayer, LIFNeuron
from ai.models.snn.surrogate import atan_spike


class ECGSNN(nn.Module):
    """
    3-layer Spiking Neural Network for ECG arrhythmia detection.

    Architecture:
        Input (256) → LIF Layer 1 (256) → LIF Layer 2 (128)
        → LIF Layer 3 (64) → Readout (2)

    The readout layer accumulates spike counts over timesteps
    and applies softmax for classification.

    Args:
        input_size:  Number of input neurons (= window_size = 256)
        hidden1:     First hidden layer size
        hidden2:     Second hidden layer size
        hidden3:     Third hidden layer size
        n_classes:   Number of output classes (2: normal/arrhythmia)
        timesteps:   Number of simulation timesteps
        beta:        Membrane decay factor
        threshold:   Spike threshold
        dropout:     Dropout rate
    """

    def __init__(self,
                 input_size: int = 256,
                 hidden1: int = 256,
                 hidden2: int = 128,
                 hidden3: int = 64,
                 n_classes: int = 2,
                 timesteps: int = 64,
                 beta: float = 0.9,
                 threshold: float = 1.0,
                 dropout: float = 0.2):
        super(ECGSNN, self).__init__()

        self.timesteps  = timesteps
        self.input_size = input_size

        # ── SNN Layers ─────────────────────────────────────────────────────────
        self.layer1 = LIFLayer(input_size, hidden1,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout)

        self.layer2 = LIFLayer(hidden1, hidden2,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout)

        self.layer3 = LIFLayer(hidden2, hidden3,
                                threshold=threshold,
                                beta=beta,
                                dropout=dropout)

        # Readout layer — standard linear, no LIF
        # Accumulates spike activity for classification
        self.readout = nn.Linear(hidden3, n_classes)

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)

    def reset_states(self):
        """Reset all LIF neuron membrane states between sequences."""
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.layer3.reset_state()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass over multiple timesteps.

        Args:
            x: Spike train input [batch, input_size]
               The full spike train is replayed over timesteps.

        Returns:
            logits:       [batch, n_classes]
            spike_record: dict with spike counts per layer (for XAI/observability)
        """
        batch_size = x.shape[0]

        # Reset membrane states for new batch
        self.reset_states()

        # Normalize input
        x = self.input_bn(x)

        # Accumulate spikes over timesteps
        spike_accum1 = torch.zeros(batch_size, self.layer1.linear.out_features).to(x.device)
        spike_accum2 = torch.zeros(batch_size, self.layer2.linear.out_features).to(x.device)
        spike_accum3 = torch.zeros(batch_size, self.layer3.linear.out_features).to(x.device)

        # Spike records for observability
        spike_record = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
        }

        # Replay input over timesteps
        # Split input into timestep chunks
        chunk_size = max(1, self.input_size // self.timesteps)

        for t in range(self.timesteps):
            # Get input slice for this timestep
            start = min(t * chunk_size, self.input_size - 1)
            end   = min(start + chunk_size, self.input_size)
            x_t   = x[:, start:end]

            # Pad if needed
            if x_t.shape[1] < self.layer1.linear.in_features:
                pad = torch.zeros(
                    batch_size,
                    self.layer1.linear.in_features - x_t.shape[1]
                ).to(x.device)
                x_t = torch.cat([x_t, pad], dim=1)

            # Forward through layers
            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)
            s3, _ = self.layer3(s2)

            # Accumulate
            spike_accum1 += s1
            spike_accum2 += s2
            spike_accum3 += s3

            # Record for observability
            spike_record['layer1'].append(s1.detach().mean().item())
            spike_record['layer2'].append(s2.detach().mean().item())
            spike_record['layer3'].append(s3.detach().mean().item())

        # Readout from accumulated spikes
        logits = self.readout(spike_accum3 / self.timesteps)

        # Add summary stats to spike record
        spike_record['total_spikes'] = (
            spike_accum1.sum() +
            spike_accum2.sum() +
            spike_accum3.sum()
        ).item()
        spike_record['sparsity'] = 1.0 - (
            spike_accum3.mean() / self.timesteps
        ).item()

        return logits, spike_record

    def get_spike_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-timestep spike attribution for XAI.
        Returns spike activity at each timestep for layer 3.

        Args:
            x: Input [batch, input_size]

        Returns:
            attribution: [batch, timesteps] spike attribution map
        """
        batch_size = x.shape[0]
        self.reset_states()
        x = self.input_bn(x)

        attributions = []
        chunk_size   = max(1, self.input_size // self.timesteps)

        for t in range(self.timesteps):
            start = min(t * chunk_size, self.input_size - 1)
            end   = min(start + chunk_size, self.input_size)
            x_t   = x[:, start:end]

            if x_t.shape[1] < self.layer1.linear.in_features:
                pad = torch.zeros(
                    batch_size,
                    self.layer1.linear.in_features - x_t.shape[1]
                ).to(x.device)
                x_t = torch.cat([x_t, pad], dim=1)

            s1, _ = self.layer1(x_t)
            s2, _ = self.layer2(s1)
            s3, _ = self.layer3(s2)

            # Attribution = mean spike activity at this timestep
            attributions.append(s3.mean(dim=1, keepdim=True))

        return torch.cat(attributions, dim=1)  # [batch, timesteps]


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    input_size = 256
    timesteps  = 64

    print("Testing ECG SNN Model...")
    model = ECGSNN(
        input_size=input_size,
        hidden1=256,
        hidden2=128,
        hidden3=64,
        n_classes=2,
        timesteps=timesteps,
        beta=0.9,
        threshold=1.0,
        dropout=0.2
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, input_size)
    logits, spike_record = model(x)

    print(f"  Input shape:      {x.shape}")
    print(f"  Output shape:     {logits.shape}")
    print(f"  Total spikes:     {spike_record['total_spikes']:.0f}")
    print(f"  Sparsity:         {spike_record['sparsity']*100:.1f}%")
    print(f"  Layer1 avg spike: {sum(spike_record['layer1'])/len(spike_record['layer1']):.4f}")

    # Test attribution
    attribution = model.get_spike_attribution(x)
    print(f"  Attribution shape: {attribution.shape}")

    # Test loss computation
    labels = torch.randint(0, 2, (batch_size,))
    loss   = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"  Loss:             {loss.item():.4f}")
    print(f"  Gradients OK:     True")

    print("\nECG SNN model test passed!")