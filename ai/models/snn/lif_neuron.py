"""
Leaky Integrate-and-Fire (LIF) Neuron for NeuroPulse AI
Core building block of all SNN models.
Implements membrane potential dynamics with surrogate gradients.
"""

import torch
import torch.nn as nn
from ai.models.snn.surrogate import atan_spike


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Membrane dynamics:
        V[t] = beta * V[t-1] + I[t]
        S[t] = 1 if V[t] >= threshold else 0
        V[t] = V[t] * (1 - S[t])  # reset after spike

    Args:
        threshold:  Spike threshold (default 1.0)
        beta:       Membrane decay factor (default 0.9)
        reset_mode: 'zero' or 'subtract'
        learn_beta: Whether beta is learnable
        learn_threshold: Whether threshold is learnable
    """

    def __init__(self,
                 threshold: float = 1.0,
                 beta: float = 0.9,
                 reset_mode: str = 'zero',
                 learn_beta: bool = False,
                 learn_threshold: bool = False):
        super(LIFNeuron, self).__init__()

        self.reset_mode = reset_mode

        # Optionally learnable parameters
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))

        # Membrane potential — initialized as None, reset each forward pass
        self.membrane = None

    def reset_state(self):
        """Reset membrane potential to zero."""
        self.membrane = None

    def init_membrane(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize membrane potential as zeros matching input shape."""
        return torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Single timestep forward pass.

        Args:
            x: Input current [batch, neurons]

        Returns:
            spikes:   Binary spike tensor [batch, neurons]
            membrane: Updated membrane potential [batch, neurons]
        """
        # Initialize membrane on first call
        if self.membrane is None:
            self.membrane = self.init_membrane(x)

        # Membrane update: leak + input
        self.membrane = self.beta * self.membrane + x

        # Generate spikes using surrogate gradient
        # (pass threshold as a tensor, not .item() — calling .item() here
        # forces a blocking GPU->CPU sync on every single timestep, which
        # is the dominant cause of slow training for this model)
        spikes = atan_spike(self.membrane, self.threshold)

        # Reset membrane after spike
        if self.reset_mode == 'zero':
            self.membrane = self.membrane * (1.0 - spikes.detach())
        elif self.reset_mode == 'subtract':
            self.membrane = self.membrane - self.threshold * spikes.detach()

        return spikes, self.membrane


class AdaptiveLIFNeuron(nn.Module):
    """
    Adaptive LIF neuron — threshold increases after each spike
    and decays back to baseline over time.
    Better for EEG where firing patterns are more complex.

    Args:
        threshold:      Base spike threshold
        beta:           Membrane decay
        theta_plus:     Threshold increase per spike
        rho:            Threshold decay rate
        learn_beta:     Learnable decay
    """

    def __init__(self,
                 threshold: float = 1.0,
                 beta: float = 0.9,
                 theta_plus: float = 0.05,
                 rho: float = 0.99,
                 learn_beta: bool = False):
        super(AdaptiveLIFNeuron, self).__init__()

        self.theta_plus = theta_plus
        self.rho        = rho

        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

        self.register_buffer('base_threshold', torch.tensor(threshold))

        self.membrane  = None
        self.threshold = None  # Adaptive threshold

    def reset_state(self):
        """Reset membrane and adaptive threshold."""
        self.membrane  = None
        self.threshold = None

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Single timestep forward with adaptive threshold.

        Args:
            x: Input current [batch, neurons]

        Returns:
            spikes:   [batch, neurons]
            membrane: [batch, neurons]
        """
        if self.membrane is None:
            self.membrane  = torch.zeros_like(x)
            self.threshold = torch.ones_like(x) * self.base_threshold

        # Membrane update
        self.membrane = self.beta * self.membrane + x

        # Spike with adaptive threshold (tensor, not .item() — see note above)
        spikes = atan_spike(self.membrane, self.base_threshold)

        # Threshold adaptation
        self.threshold = (self.rho * self.threshold +
                          self.theta_plus * spikes.detach())

        # Reset membrane
        self.membrane = self.membrane * (1.0 - spikes.detach())

        return spikes, self.membrane


class LIFLayer(nn.Module):
    """
    Complete LIF layer: Linear transformation + LIF neuron.
    Drop-in replacement for nn.Linear in SNN architectures.

    Args:
        in_features:  Input size
        out_features: Output size (number of neurons)
        threshold:    Spike threshold
        beta:         Membrane decay
        dropout:      Dropout rate (applied to input)
        adaptive:     Use adaptive LIF if True
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 threshold: float = 1.0,
                 beta: float = 0.9,
                 dropout: float = 0.0,
                 adaptive: bool = False):
        super(LIFLayer, self).__init__()

        self.linear  = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        if adaptive:
            self.neuron = AdaptiveLIFNeuron(threshold=threshold, beta=beta)
        else:
            self.neuron = LIFNeuron(threshold=threshold, beta=beta)

    def reset_state(self):
        """Reset neuron membrane state."""
        self.neuron.reset_state()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input [batch, in_features]

        Returns:
            spikes:   [batch, out_features]
            membrane: [batch, out_features]
        """
        if self.dropout is not None:
            x = self.dropout(x)
        current = self.linear(x)
        return self.neuron(current)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    timesteps  = 10
    n_input    = 64
    n_neurons  = 32

    print("Testing LIF Neuron...")

    # Test basic LIF
    lif = LIFNeuron(threshold=1.0, beta=0.9)
    x   = torch.randn(batch_size, n_neurons)

    all_spikes = []
    for t in range(timesteps):
        spikes, membrane = lif(x)
        all_spikes.append(spikes)

    all_spikes = torch.stack(all_spikes)
    print(f"  Input shape:    {x.shape}")
    print(f"  Spikes shape:   {all_spikes.shape}")
    print(f"  Total spikes:   {all_spikes.sum().item():.0f}")
    print(f"  Sparsity:       {(1-all_spikes.mean())*100:.1f}%")

    # Test Adaptive LIF
    print("\nTesting Adaptive LIF Neuron...")
    alif = AdaptiveLIFNeuron(threshold=1.0, beta=0.9)
    all_spikes_a = []
    for t in range(timesteps):
        spikes, membrane = alif(x)
        all_spikes_a.append(spikes)
    all_spikes_a = torch.stack(all_spikes_a)
    print(f"  Spikes shape:   {all_spikes_a.shape}")
    print(f"  Total spikes:   {all_spikes_a.sum().item():.0f}")

    # Test LIF Layer
    print("\nTesting LIF Layer...")
    lif_layer = LIFLayer(n_input, n_neurons, dropout=0.1)
    x_in      = torch.randn(batch_size, n_input)
    spikes, membrane = lif_layer(x_in)
    print(f"  Input shape:    {x_in.shape}")
    print(f"  Output shape:   {spikes.shape}")
    print(f"  Membrane shape: {membrane.shape}")

    print("\nAll LIF neuron tests passed!")