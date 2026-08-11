"""
EEG Training Pipeline for NeuroPulse AI
Trains EEG-SNN and CNN/LSTM/Transformer baselines
on CHB-MIT Scalp EEG Dataset.
Run: python ai/training/train_eeg.py
"""

import os
import sys
import gc
import numpy as np
import torch
import json
import time

sys.path.append('.')

from ai.models.snn.eeg_snn import EEGSNN
from ai.models.baselines.cnn_model import EEGCNN
from ai.models.baselines.lstm_model import EEGLSTM
from ai.models.baselines.transformer_model import EEGTransformer
from ai.training.trainer import Trainer

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':      'data/processed/eeg',
    'save_dir':      'ai/saved_models',
    # 480,070 train samples / 32 = 15,002 batches/epoch — far too many Python-level
    # loop iterations for the SNN's 64-timestep simulation, which is the dominant
    # remaining cost after fixing the .item() sync bug. 256 cuts this to ~1,875
    # batches/epoch (~8x fewer), which should give a roughly proportional speedup
    # since GPU utilization per batch also improves with fewer, larger batches.
    # If you hit a CUDA out-of-memory error, step back down to 128 first.
    'batch_size':    256,
    'epochs':        50,
    # Mildly increased alongside batch_size (not fully linearly scaled 8x,
    # since the ~154x class weight already applied for imbalance makes very
    # large effective gradients risky — tune down if you see loss spikes/NaN).
    'learning_rate': 2e-3,
    'weight_decay':  1e-4,
    'patience':      10,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
    'n_channels':    18,
    'window_size':   256,
}


def load_data():
    """Load preprocessed EEG data."""
    print("Loading EEG data...")
    X_train = np.load(f"{CONFIG['data_dir']}/train/X.npy")
    y_train = np.load(f"{CONFIG['data_dir']}/train/y.npy")
    X_val   = np.load(f"{CONFIG['data_dir']}/val/X.npy")
    y_val   = np.load(f"{CONFIG['data_dir']}/val/y.npy")
    X_test  = np.load(f"{CONFIG['data_dir']}/test/X.npy")
    y_test  = np.load(f"{CONFIG['data_dir']}/test/y.npy")

    print(f"  Train: {X_train.shape} | {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}   | {np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}  | {np.bincount(y_test)}")

    # Class imbalance is handled via class-weighted loss in Trainer
    # (see trainer.py compute_class_weights / train()), not via
    # materialized array upsampling. Upsampling 1.5K seizure windows to
    # ~20% of ~480K normal windows requires holding several full-dataset
    # copies in RAM simultaneously (the source array, the upsampled
    # seizure array, and the concatenated result), which is what caused
    # the previous out-of-memory crash. Class-weighted loss achieves the
    # same imbalance-correction goal without duplicating any data.
    return X_train, y_train, X_val, y_val, X_test, y_test


def upsample_minority(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Upsample minority class (seizure) to handle
    severe class imbalance in EEG data.
    Seizure windows are very rare (~0.47%).
    """
    normal_idx  = np.where(y == 0)[0]
    seizure_idx = np.where(y == 1)[0]

    if len(seizure_idx) == 0:
        return X, y

    # Upsample seizure to 20% of normal
    target_seizure = max(len(seizure_idx),
                         int(len(normal_idx) * 0.2))
    repeat_times   = target_seizure // len(seizure_idx) + 1
    seizure_idx_up = np.tile(seizure_idx, repeat_times)[:target_seizure]

    # Add small noise to upsampled seizure windows
    X_seizure_up = X[seizure_idx_up].copy()
    X_seizure_up += np.random.randn(*X_seizure_up.shape) * 0.01

    X_balanced = np.concatenate([X[normal_idx], X_seizure_up], axis=0)
    y_balanced = np.concatenate([y[normal_idx],
                                  np.ones(target_seizure, dtype=np.int64)])

    # Shuffle
    idx = np.random.permutation(len(X_balanced))
    return X_balanced[idx], y_balanced[idx]


def estimate_energy(model, X_sample, device, is_snn=False):
    """Estimate inference energy in mJ."""
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


def train_all_models():
    """Train all EEG models and compare results."""
    print("=" * 60)
    print("NeuroPulse AI — EEG Training Pipeline")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = {}

    # ── 1. Train EEG SNN ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training EEG-SNN (Adaptive LIF)")
    print("=" * 60)

    snn_model = EEGSNN(
        n_channels=CONFIG['n_channels'],
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

    snn_trainer = Trainer(
        model=snn_model,
        model_name='eeg_snn',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    snn_trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    snn_metrics = snn_trainer.evaluate(X_test, y_test)
    snn_energy  = estimate_energy(snn_model, X_test,
                                   CONFIG['device'], is_snn=True)
    results['EEG-SNN'] = {**snn_metrics, **snn_energy}

    # Free this model's GPU/CPU memory before training the next one —
    # without this, all four trainers (and their internal copies of
    # X_train as torch tensors) stay alive simultaneously until the end
    # of this function, multiplying peak memory usage by 4x.
    del snn_model, snn_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. Train CNN Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training EEG-CNN Baseline")
    print("=" * 60)

    cnn_model = EEGCNN(
        n_channels=CONFIG['n_channels'],
        window_size=CONFIG['window_size'],
        n_classes=2,
        dropout=0.3
    )

    cnn_trainer = Trainer(
        model=cnn_model,
        model_name='eeg_cnn',
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
    results['EEG-CNN'] = {**cnn_metrics, **cnn_energy}

    del cnn_model, cnn_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. Train LSTM Baseline ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training EEG-LSTM Baseline")
    print("=" * 60)

    lstm_model = EEGLSTM(
        n_channels=CONFIG['n_channels'],
        hidden_size=128,
        num_layers=2,
        n_classes=2,
        dropout=0.3
    )

    lstm_trainer = Trainer(
        model=lstm_model,
        model_name='eeg_lstm',
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
    results['EEG-LSTM'] = {**lstm_metrics, **lstm_energy}

    del lstm_model, lstm_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 4. Train Transformer Baseline ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training EEG-Transformer Baseline")
    print("=" * 60)

    trans_model = EEGTransformer(
        n_channels=CONFIG['n_channels'],
        window_size=CONFIG['window_size'],
        d_model=64,
        n_heads=4,
        n_layers=3,
        n_classes=2,
        dropout=0.1
    )

    trans_trainer = Trainer(
        model=trans_model,
        model_name='eeg_transformer',
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
    results['EEG-Transformer'] = {**trans_metrics, **trans_energy}

    del trans_model, trans_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Print Comparison Table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EEG MODEL COMPARISON")
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
    results_path = os.path.join(
        CONFIG['save_dir'], 'eeg_benchmark_results.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return results


if __name__ == "__main__":
    results = train_all_models()