"""
Surrogate Gradient Functions for NeuroPulse AI
Standard backpropagation fails through binary spike function
because its derivative is zero everywhere except at threshold.
Surrogate gradients replace the true derivative during backprop.
"""

import torch
import torch.nn as nn


# ── ATan Surrogate (default — most stable) ────────────────────────────────────
class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient.
    Forward:  Heaviside step function (spike if membrane > threshold)
    Backward: Smooth ATan approximation of derivative
    Reference: Fang et al. SpikingJelly (Science Advances 2023)
    """
    @staticmethod
    def forward(ctx, membrane: torch.Tensor,
                threshold: float = 1.0,
                alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.alpha     = alpha
        # Spike if membrane potential exceeds threshold
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane,  = ctx.saved_tensors
        threshold  = ctx.threshold
        alpha      = ctx.alpha
        # Surrogate derivative: alpha / (2 * (1 + (pi*alpha*(m-th)/2)^2))
        grad = alpha / (2.0 * (
            1.0 + (torch.pi * alpha * (membrane - threshold) / 2.0) ** 2
        ))
        return grad_output * grad, None, None


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient.
    Simpler and faster than ATan, slightly less stable.
    """
    @staticmethod
    def forward(ctx, membrane: torch.Tensor,
                threshold: float = 1.0,
                alpha: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.alpha     = alpha
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha     = ctx.alpha
        # Surrogate: alpha / (1 + alpha * |membrane - threshold|)^2
        abs_diff = torch.abs(membrane - threshold)
        grad     = alpha / ((1.0 + alpha * abs_diff) ** 2)
        return grad_output * grad, None, None


class SuperSpikeSurrogate(torch.autograd.Function):
    """
    SuperSpike surrogate gradient.
    Based on Zenke & Ganguli (2018).
    """
    @staticmethod
    def forward(ctx, membrane: torch.Tensor,
                threshold: float = 1.0,
                beta: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.beta      = beta
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane, = ctx.saved_tensors
        threshold = ctx.threshold
        beta      = ctx.beta
        grad      = 1.0 / ((beta * torch.abs(membrane - threshold) + 1.0) ** 2)
        return grad_output * grad, None, None


# ── Convenience wrapper functions ─────────────────────────────────────────────
def atan_spike(membrane: torch.Tensor,
               threshold: float = 1.0,
               alpha: float = 2.0) -> torch.Tensor:
    """Apply ATan surrogate spike function."""
    return ATanSurrogate.apply(membrane, threshold, alpha)


def fast_sigmoid_spike(membrane: torch.Tensor,
                        threshold: float = 1.0,
                        alpha: float = 25.0) -> torch.Tensor:
    """Apply Fast Sigmoid surrogate spike function."""
    return FastSigmoidSurrogate.apply(membrane, threshold, alpha)


def superspike(membrane: torch.Tensor,
               threshold: float = 1.0,
               beta: float = 10.0) -> torch.Tensor:
    """Apply SuperSpike surrogate spike function."""
    return SuperSpikeSurrogate.apply(membrane, threshold, beta)


# ── Surrogate selector ────────────────────────────────────────────────────────
def get_surrogate(name: str = 'atan'):
    """
    Get surrogate function by name.

    Args:
        name: 'atan', 'fast_sigmoid', 'superspike'

    Returns:
        surrogate function
    """
    surrogates = {
        'atan':         atan_spike,
        'fast_sigmoid': fast_sigmoid_spike,
        'superspike':   superspike,
    }
    if name not in surrogates:
        raise ValueError(f"Unknown surrogate: {name}. "
                         f"Choose from {list(surrogates.keys())}")
    return surrogates[name]


if __name__ == "__main__":
    # ── Quick test ─────────────────────────────────────────────────────────────
    torch.manual_seed(42)

    # Test membrane potentials
    membrane = torch.randn(4, 64, requires_grad=True)

    print("Testing surrogate gradient functions...\n")

    # ATan surrogate
    spikes = atan_spike(membrane, threshold=1.0, alpha=2.0)
    loss   = spikes.sum()
    loss.backward()
    print(f"ATan surrogate:")
    print(f"  Spikes shape:    {spikes.shape}")
    print(f"  Spike count:     {spikes.sum().item():.0f}")
    print(f"  Gradient shape:  {membrane.grad.shape}")
    print(f"  Gradient range:  [{membrane.grad.min():.4f}, "
          f"{membrane.grad.max():.4f}]")

    # Reset gradient
    membrane.grad = None

    # Fast sigmoid
    spikes2 = fast_sigmoid_spike(membrane, threshold=1.0, alpha=25.0)
    loss2   = spikes2.sum()
    loss2.backward()
    print(f"\nFast Sigmoid surrogate:")
    print(f"  Spike count:     {spikes2.sum().item():.0f}")
    print(f"  Gradient range:  [{membrane.grad.min():.4f}, "
          f"{membrane.grad.max():.4f}]")

    membrane.grad = None

    # SuperSpike
    spikes3 = superspike(membrane, threshold=1.0, beta=10.0)
    loss3   = spikes3.sum()
    loss3.backward()
    print(f"\nSuperSpike surrogate:")
    print(f"  Spike count:     {spikes3.sum().item():.0f}")
    print(f"  Gradient range:  [{membrane.grad.min():.4f}, "
          f"{membrane.grad.max():.4f}]")

    print("\nAll surrogate gradient tests passed!")