"""
ECG Training Pipeline for NeuroPulse AI
Trains ECG-SNN and CNN/LSTM/Transformer baselines
on MIT-BIH Arrhythmia Dataset.
Run: python ai/training/train_ecg.py
"""

import os
import sys
import numpy as np
import torch
import json
import time

sys.path.append('.')

from ai.models.snn.ecg_snn import ECGSNN
from ai.models.baselines.cnn_model import ECGCNN
from ai.models.baselines.lstm_model import ECGLSTM
from ai.models.baselines.transformer_model import ECGTransformer
from ai.training.trainer import Trainer

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':     'data/processed/ecg',
    'save_dir':     'ai/saved_models',
    'batch_size':   64,
    'epochs':       50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience':     10,
    'device':      'cuda:0',
}


def load_data():
    """Load preprocessed ECG data."""
    print("Loading ECG data...")
    X_train = np.load(f"{CONFIG['data_dir']}/train/X.npy")
    y_train = np.load(f"{CONFIG['data_dir']}/train/y.npy")
    X_val   = np.load(f"{CONFIG['data_dir']}/val/X.npy")
    y_val   = np.load(f"{CONFIG['data_dir']}/val/y.npy")
    X_test  = np.load(f"{CONFIG['data_dir']}/test/X.npy")
    y_test  = np.load(f"{CONFIG['data_dir']}/test/y.npy")

    print(f"  Train: {X_train.shape} | {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}   | {np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}  | {np.bincount(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def estimate_energy(model, X_sample, device, is_snn=False):
    """
    Estimate inference energy consumption in mJ.
    SNN energy based on spike count × synaptic energy.
    ANN energy based on MAC operations.
    """
    model.eval()
    x = torch.FloatTensor(X_sample[:1]).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(x)
    latency_ms = (time.time() - start) * 1000

    if is_snn:
        # SNN energy = spike_count × E_syn (0.1 pJ per spike)
        _, spike_record = output
        total_spikes = spike_record.get('total_spikes', 0)
        energy_mj    = total_spikes * 0.1e-9 * 1000  # pJ → mJ
        sparsity     = spike_record.get('sparsity', 0)
    else:
        # ANN energy = total MACs × E_mac (4.6 pJ per MAC)
        total_params = sum(p.numel() for p in model.parameters())
        energy_mj    = total_params * 4.6e-9 * 1000
        sparsity     = 0.0

    return {
        'latency_ms': round(latency_ms, 3),
        'energy_mj':  round(energy_mj, 6),
        'sparsity':   round(sparsity, 4)
    }


def train_all_models():
    """Train all ECG models and compare results."""
    print("=" * 60)
    print("NeuroPulse AI — ECG Training Pipeline")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = {}

    # ── 1. Train ECG SNN ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training ECG-SNN (Spiking Neural Network)")
    print("=" * 60)

    snn_model = ECGSNN(
        input_size=256,
        hidden1=256,
        hidden2=128,
        hidden3=64,
        n_classes=2,
        timesteps=64,
        beta=0.9,
        threshold=1.0,
        dropout=0.2
    )

    snn_trainer = Trainer(
        model=snn_model,
        model_name='ecg_snn',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    snn_history = snn_trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    snn_metrics = snn_trainer.evaluate(X_test, y_test)
    snn_energy  = estimate_energy(snn_model, X_test,
                                   CONFIG['device'], is_snn=True)

    results['ECG-SNN'] = {**snn_metrics, **snn_energy}

    # ── 2. Train CNN Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training ECG-CNN Baseline")
    print("=" * 60)

    cnn_model = ECGCNN(input_size=256, n_classes=2, dropout=0.3)

    cnn_trainer = Trainer(
        model=cnn_model,
        model_name='ecg_cnn',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    cnn_trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    cnn_metrics = cnn_trainer.evaluate(X_test, y_test)
    cnn_energy  = estimate_energy(cnn_model, X_test,
                                   CONFIG['device'], is_snn=False)

    results['ECG-CNN'] = {**cnn_metrics, **cnn_energy}

    # ── 3. Train LSTM Baseline ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training ECG-LSTM Baseline")
    print("=" * 60)

    lstm_model = ECGLSTM(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        n_classes=2,
        dropout=0.3
    )

    lstm_trainer = Trainer(
        model=lstm_model,
        model_name='ecg_lstm',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    lstm_trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    lstm_metrics = lstm_trainer.evaluate(X_test, y_test)
    lstm_energy  = estimate_energy(lstm_model, X_test,
                                    CONFIG['device'], is_snn=False)

    results['ECG-LSTM'] = {**lstm_metrics, **lstm_energy}

    # ── 4. Train Transformer Baseline ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training ECG-Transformer Baseline")
    print("=" * 60)

    trans_model = ECGTransformer(
        input_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        dropout=0.1
    )

    trans_trainer = Trainer(
        model=trans_model,
        model_name='ecg_transformer',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    trans_trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    trans_metrics = trans_trainer.evaluate(X_test, y_test)
    trans_energy  = estimate_energy(trans_model, X_test,
                                     CONFIG['device'], is_snn=False)

    results['ECG-Transformer'] = {**trans_metrics, **trans_energy}

    # ── Print Comparison Table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ECG MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} "
          f"{'Latency(ms)':>12} {'Energy(mJ)':>12} {'Sparsity':>10}")
    print("-" * 74)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['latency_ms']:>12.3f} "
            f"{metrics['energy_mj']:>12.6f} "
            f"{metrics.get('sparsity', 0):>10.4f}"
        )

    # Save results
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    results_path = os.path.join(
        CONFIG['save_dir'], 'ecg_benchmark_results.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return results


if __name__ == "__main__":
    results = train_all_models()