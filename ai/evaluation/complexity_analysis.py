"""
Computational Complexity Analysis for NeuroPulse AI
====================================================
Computes parameter count, theoretical FLOPs/operations, model size on
disk, peak memory footprint, and measured inference runtime for every
model x modality combination (12 total: SNN/CNN/LSTM/Transformer x
ECG/EEG/EMG).

This does NOT require completed training — it works on model
architecture alone (random-initialized weights are fine for FLOPs,
parameter count, and runtime; only accuracy-dependent metrics need a
trained checkpoint, which this script does not compute).

Run: python -m ai.evaluation.complexity_analysis
Output: outputs/plots/13_complexity_analysis.png
        outputs/tables/complexity_analysis.csv
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from ai.models.snn.ecg_snn import ECGSNN
from ai.models.snn.eeg_snn import EEGSNN
from ai.models.snn.emg_snn import EMGSNN
from ai.models.baselines.cnn_model import ECGCNN, EEGCNN, EMGCNN
from ai.models.baselines.lstm_model import ECGLSTM, EEGLSTM, EMGLSTM
from ai.models.baselines.transformer_model import ECGTransformer, EEGTransformer, EMGTransformer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TIMING_RUNS = 100  # repeated forward passes for stable latency estimate
BATCH_SIZE_FOR_TIMING = 1  # single-sample inference latency (edge-deployment realistic)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Model size on disk if saved as float32 state_dict."""
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    return (n_params + n_buffers) * 4 / (1024 ** 2)  # 4 bytes per float32


class FlopCounter:
    """
    Runtime FLOP counter using forward hooks. This correctly handles the
    SNN's repeated per-timestep layer invocations automatically — each
    call to a hooked layer (e.g. once per timestep inside the LIF loop)
    adds to the running total, so no separate "multiply by timesteps"
    bookkeeping is needed.
    """
    def __init__(self):
        self.total_flops = 0
        self.hooks = []

    def _linear_hook(self, module, inp, out):
        in_features = module.in_features
        out_features = module.out_features
        batch_elements = inp[0].numel() // in_features
        self.total_flops += 2 * batch_elements * in_features * out_features

    def _conv1d_hook(self, module, inp, out):
        out_elements = out.numel()
        k = module.kernel_size[0]
        in_channels_per_group = module.in_channels // module.groups
        self.total_flops += 2 * out_elements * in_channels_per_group * k

    def _lstm_hook(self, module, inp, out):
        x = inp[0]
        seq_len = x.shape[1] if module.batch_first else x.shape[0]
        batch = x.shape[0] if module.batch_first else x.shape[1]
        directions = 2 if module.bidirectional else 1
        flops = 0
        input_size = module.input_size
        for _ in range(module.num_layers):
            flops += 8 * module.hidden_size * (input_size + module.hidden_size) * seq_len * directions
            input_size = module.hidden_size * directions
        self.total_flops += flops * batch

    def register(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, nn.Conv1d):
                self.hooks.append(m.register_forward_hook(self._conv1d_hook))
            elif isinstance(m, nn.LSTM):
                self.hooks.append(m.register_forward_hook(self._lstm_hook))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def measure_flops(model: nn.Module, x_batched: torch.Tensor) -> int:
    """Single forward pass with FLOP-counting hooks attached. x_batched
    must already include the batch dimension."""
    counter = FlopCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad():
        _ = model(x_batched)
    counter.remove()
    return counter.total_flops


def measure_runtime(model: nn.Module, x: torch.Tensor, device: str) -> dict:
    """Measure wall-clock latency over repeated runs (mean +/- std)."""
    model.eval()
    model.to(device)
    x = x.to(device)

    with torch.no_grad():
        # Warm-up (avoid first-call overhead skewing results)
        for _ in range(10):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]

        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(N_TIMING_RUNS):
            start = time.perf_counter()
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'latency_mean_ms': float(np.mean(times)),
        'latency_std_ms':  float(np.std(times)),
        'latency_p95_ms':  float(np.percentile(times, 95)),
    }


def measure_peak_memory(model: nn.Module, x: torch.Tensor, device: str) -> float:
    """Peak memory footprint (MB) during a single forward+backward pass."""
    model.to(device)
    x = x.to(device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model.train()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()

        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 ** 2)
    else:
        # CPU peak memory tracking requires external profiling (e.g. memory_profiler);
        # report None so the table clearly shows this was not measured on CPU rather
        # than silently reporting a misleading number.
        return None


def analyze_model(name: str, model: nn.Module, input_shape: tuple, is_snn: bool,
                   snn_timesteps: int = None) -> dict:
    n_params = count_parameters(model)
    size_mb = model_size_mb(model)

    # Move model to device ONCE here, and move x to device ONCE here too —
    # previously each of measure_runtime/measure_peak_memory/measure_flops
    # called .to(device) on a locally-scoped variable, which moved the
    # model (mutating it in place, so it "stuck") but did NOT mutate the
    # caller's x (a .to() call returns a new tensor, it doesn't move the
    # original in place) — so after the first call, the model lived on
    # GPU while x remained on CPU for every subsequent call. Doing it once
    # here, consistently, fixes the device-mismatch crashes.
    model.to(DEVICE)
    x = torch.randn(BATCH_SIZE_FOR_TIMING, *input_shape).to(DEVICE)

    runtime = measure_runtime(model, x, DEVICE)
    flops = measure_flops(model, x)

    # BatchNorm layers (present in some baseline architectures) require
    # more than 1 sample per channel when the model is in .train() mode
    # (needed here to also measure backward-pass memory) — batch_size=1
    # fails with "Expected more than 1 value per channel". Use a slightly
    # larger batch just for this specific measurement.
    x_for_mem = torch.randn(max(2, BATCH_SIZE_FOR_TIMING), *input_shape).to(DEVICE)
    peak_mem = measure_peak_memory(model, x_for_mem, DEVICE)

    result = {
        'model': name,
        'parameters': n_params,
        'model_size_mb': round(size_mb, 3),
        'flops': flops,
        **runtime,
        'peak_memory_mb': round(peak_mem, 2) if peak_mem is not None else None,
        'is_snn': is_snn,
    }

    if is_snn and snn_timesteps is not None:
        result['snn_timesteps'] = snn_timesteps

    return result


def build_model_registry():
    """
    Returns {modality: [(name, model, input_shape, is_snn, timesteps), ...]}

    IMPORTANT: input_shape is the shape WITHOUT the batch dimension, and
    must match what each architecture actually expects — this differs
    across CNN (needs a channel dim for Conv1d), Transformer, and LSTM
    (sequence-major) even within the same modality. Shapes below are
    based on each model file's docstring; verify against your actual
    saved config before trusting the results, and adjust any constructor
    keyword argument that doesn't match your local model definitions.
    """
    registry = {
        'ECG': [
            ('SNN',         ECGSNN(input_size=256, hidden1=256, hidden2=128, hidden3=64,
                                    n_classes=2, timesteps=64, beta=0.9, threshold=1.0, dropout=0.2),
             (256,), True, 64),
            ('CNN',         ECGCNN(input_size=256, n_classes=2, dropout=0.3), (1, 256), False, None),
            ('LSTM',        ECGLSTM(input_size=1, hidden_size=128, num_layers=2,
                                     n_classes=2, dropout=0.3), (256,), False, None),
            ('Transformer', ECGTransformer(input_size=256, d_model=64, n_heads=4,
                                            n_layers=2, dim_feedforward=128, n_classes=2), (256,), False, None),
        ],
        'EEG': [
            ('SNN',         EEGSNN(n_channels=18, input_size=64, hidden1=128, hidden2=64, hidden3=32,
                                    n_classes=2, timesteps=64, beta=0.9, threshold=1.0, dropout=0.2),
             (18, 256), True, 64),
            ('CNN',         EEGCNN(n_channels=18, window_size=256, n_classes=2, dropout=0.3),
             (18, 256), False, None),
            ('LSTM',        EEGLSTM(n_channels=18, hidden_size=128, num_layers=2,
                                     n_classes=2, dropout=0.3), (18, 256), False, None),
            ('Transformer', EEGTransformer(n_channels=18, window_size=256, d_model=64,
                                            n_heads=4, n_layers=3, n_classes=2, dropout=0.1),
             (18, 256), False, None),
        ],
        'EMG': [
            ('SNN',         EMGSNN(n_channels=12, window_size=256, hidden1=64, hidden2=32,
                                    n_classes=2, timesteps=32, beta=0.9, dropout=0.2),
             (12, 256), True, 32),
            ('CNN',         EMGCNN(n_channels=12, window_size=256, n_classes=2, dropout=0.3),
             (12, 256), False, None),
            ('LSTM',        EMGLSTM(n_channels=12, hidden_size=64, num_layers=2,
                                     n_classes=2, dropout=0.3), (12, 256), False, None),
            ('Transformer', EMGTransformer(n_channels=12, window_size=256, d_model=32,
                                            n_heads=4, n_layers=2, n_classes=2), (12, 256), False, None),
        ],
    }
    return registry


def main():
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Timing runs per model: {N_TIMING_RUNS}\n")

    registry = build_model_registry()
    all_results = []

    for modality, models in registry.items():
        print(f"=== {modality} ===")
        for name, model, input_shape, is_snn, timesteps in models:
            try:
                result = analyze_model(name, model, input_shape, is_snn, timesteps)
                result['modality'] = modality
                all_results.append(result)
                print(f"  {name:12s} | params: {result['parameters']:>10,} | "
                      f"FLOPs: {result['flops']:>12,} | "
                      f"size: {result['model_size_mb']:>7.3f} MB | "
                      f"latency: {result['latency_mean_ms']:>7.3f}"
                      f"+/-{result['latency_std_ms']:.3f} ms")
            except Exception as e:
                print(f"  {name:12s} | FAILED: {e}")
                print(f"    (check that constructor args in build_model_registry() "
                      f"match your actual model definitions)")

    # Save raw results
    with open('outputs/tables/complexity_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save CSV for direct import into the paper's LaTeX table
    import csv
    csv_path = 'outputs/tables/complexity_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'modality', 'model', 'parameters', 'flops', 'model_size_mb',
            'latency_mean_ms', 'latency_std_ms', 'latency_p95_ms', 'peak_memory_mb'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    print(f"\nSaved: {csv_path}")
    print("Saved: outputs/tables/complexity_analysis.json")

    # ── Plot: parameters vs latency, sized by memory, colored by modality ──
    try:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle('NeuroPulse AI — Computational Complexity Analysis', fontsize=14, fontweight='bold')

        colors = {'ECG': '#2dd4bf', 'EEG': '#a78bfa', 'EMG': '#f59e0b'}
        markers = {'SNN': 'o', 'CNN': 's', 'LSTM': '^', 'Transformer': 'D'}

        ax = axes[0]
        for r in all_results:
            ax.scatter(r['parameters'], r['latency_mean_ms'],
                       color=colors[r['modality']], marker=markers[r['model']],
                       s=120, edgecolors='white', linewidths=0.5,
                       label=f"{r['modality']}-{r['model']}")
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Latency (ms)')
        ax.set_xscale('log')
        ax.set_title('Parameters vs. Inference Latency')
        ax.grid(alpha=0.2)

        ax = axes[1]
        model_order = ['SNN', 'CNN', 'LSTM', 'Transformer']
        x_pos = np.arange(len(model_order))
        width = 0.25
        for i, modality in enumerate(['ECG', 'EEG', 'EMG']):
            vals = []
            for m in model_order:
                match = next((r['model_size_mb'] for r in all_results
                             if r['modality'] == modality and r['model'] == m), None)
                vals.append(match if match is not None else 0)
            ax.bar(x_pos + i * width, vals, width, label=modality, color=colors[modality])
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(model_order)
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size by Architecture')
        ax.legend()
        ax.grid(alpha=0.2, axis='y')

        plt.tight_layout()
        plt.savefig('outputs/plots/13_complexity_analysis.png', dpi=150, facecolor='#0d1117')
        print("Saved: outputs/plots/13_complexity_analysis.png")
    except ImportError:
        print("matplotlib not available — skipping plot generation (CSV/JSON still saved)")


if __name__ == "__main__":
    main()