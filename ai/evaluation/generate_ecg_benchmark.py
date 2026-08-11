"""
Generate ECG Benchmark Results from saved checkpoints.
No retraining needed — loads saved .pth files directly.
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.append('.')

from ai.models.snn.ecg_snn import ECGSNN
from ai.models.baselines.cnn_model import ECGCNN
from ai.models.baselines.lstm_model import ECGLSTM
from ai.models.baselines.transformer_model import ECGTransformer
from ai.training.trainer import Trainer

CONFIG = {
    'data_dir':  'data/processed/ecg',
    'save_dir':  'ai/saved_models',
    'device':    'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
}


def load_test_data():
    X_test = np.load(f"{CONFIG['data_dir']}/test/X.npy")
    y_test = np.load(f"{CONFIG['data_dir']}/test/y.npy")
    print(f"Test data: {X_test.shape} | {np.bincount(y_test)}")
    return X_test, y_test


def estimate_energy(model, X_sample, device, is_snn=False):
    model.eval()
    x = torch.FloatTensor(X_sample[:1]).to(device)
    start = time.time()
    with torch.no_grad():
        output = model(x)
    latency_ms = (time.time() - start) * 1000

    if is_snn:
        _, spike_record = output
        total_spikes = spike_record.get('total_spikes', 0)
        energy_mj    = total_spikes * 0.1e-9 * 1000
        sparsity     = spike_record.get('sparsity', 0)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        energy_mj    = total_params * 4.6e-9 * 1000
        sparsity     = 0.0

    return {
        'latency_ms': round(latency_ms, 3),
        'energy_mj':  round(energy_mj, 6),
        'sparsity':   round(sparsity, 4)
    }


def evaluate_model(model, model_name, X_test, y_test):
    """Load checkpoint and evaluate."""
    trainer = Trainer(
        model=model,
        model_name=model_name,
        device=CONFIG['device']
    )
    trainer.load_checkpoint()
    metrics = trainer.evaluate(X_test, y_test,
                                batch_size=CONFIG['batch_size'])
    return metrics


def main():
    print("=" * 60)
    print("Generating ECG Benchmark Results")
    print("=" * 60)

    X_test, y_test = load_test_data()
    results = {}

    # ECG SNN
    print("\nEvaluating ECG-SNN...")
    model = ECGSNN(input_size=256, hidden1=256, hidden2=128,
                    hidden3=64, n_classes=2, timesteps=64)
    metrics = evaluate_model(model, 'ecg_snn', X_test, y_test)
    energy  = estimate_energy(model, X_test, CONFIG['device'], is_snn=True)
    results['ECG-SNN'] = {**metrics, **energy}

    # ECG CNN
    print("\nEvaluating ECG-CNN...")
    model = ECGCNN(input_size=256, n_classes=2)
    metrics = evaluate_model(model, 'ecg_cnn', X_test, y_test)
    energy  = estimate_energy(model, X_test, CONFIG['device'], is_snn=False)
    results['ECG-CNN'] = {**metrics, **energy}

    # ECG LSTM
    print("\nEvaluating ECG-LSTM...")
    model = ECGLSTM(input_size=1, hidden_size=128, num_layers=2, n_classes=2)
    metrics = evaluate_model(model, 'ecg_lstm', X_test, y_test)
    energy  = estimate_energy(model, X_test, CONFIG['device'], is_snn=False)
    results['ECG-LSTM'] = {**metrics, **energy}

    # ECG Transformer
    print("\nEvaluating ECG-Transformer...")
    model = ECGTransformer(input_size=256, d_model=64,
                            n_heads=4, n_layers=2, n_classes=2)
    metrics = evaluate_model(model, 'ecg_transformer', X_test, y_test)
    energy  = estimate_energy(model, X_test, CONFIG['device'], is_snn=False)
    results['ECG-Transformer'] = {**metrics, **energy}

    # Print table
    print("\n" + "=" * 60)
    print("ECG MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} "
          f"{'Latency(ms)':>12} {'Energy(mJ)':>12} {'Sparsity':>10}")
    print("-" * 74)
    for model_name, m in results.items():
        print(f"{model_name:<20} "
              f"{m['accuracy']:>10.4f} "
              f"{m['f1']:>10.4f} "
              f"{m['latency_ms']:>12.3f} "
              f"{m['energy_mj']:>12.6f} "
              f"{m.get('sparsity', 0):>10.4f}")

    # Save
    path = os.path.join(CONFIG['save_dir'], 'ecg_benchmark_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {path}")
    return results


if __name__ == "__main__":
    main()