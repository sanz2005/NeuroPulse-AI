"""
Advanced Evaluation for NeuroPulse AI
======================================
Loads saved checkpoints for every trained model x modality combination
and generates:
  - ROC curves (with AUC) overlaid per modality
  - Precision-Recall curves (with AP) overlaid per modality
  - Calibration / reliability diagrams
  - Confusion matrices per model
  - Robustness-to-noise curves (accuracy/F1 vs. injected Gaussian noise)

Only models with an existing checkpoint at ai/saved_models/{name}_best.pth
are evaluated — others are skipped with a printed warning, so this script
is safe to run at any point (e.g. right after ECG/EMG finish, before EEG
retraining completes).

Run: python -m ai.evaluation.advanced_evaluation
Output: outputs/plots/09_roc_curves.png
        outputs/plots/10_pr_curves.png
        outputs/plots/11_calibration.png
        outputs/plots/14_confusion_matrices.png
        outputs/plots/15_noise_robustness.png
        outputs/tables/advanced_metrics.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

from ai.models.snn.ecg_snn import ECGSNN
from ai.models.snn.eeg_snn import EEGSNN
from ai.models.snn.emg_snn import EMGSNN
from ai.models.baselines.cnn_model import ECGCNN, EEGCNN, EMGCNN
from ai.models.baselines.lstm_model import ECGLSTM, EEGLSTM, EMGLSTM
from ai.models.baselines.transformer_model import ECGTransformer, EEGTransformer, EMGTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'ai/saved_models'
DATA_DIR = 'data/processed'
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  # std of injected Gaussian noise

COLORS = {'SNN': '#2dd4bf', 'CNN': '#a78bfa', 'LSTM': '#f59e0b', 'Transformer': '#f87171'}


def build_model(modality: str, arch: str) -> nn.Module:
    """Construct the (untrained) architecture matching the checkpoint to load into.
    Keep these constructor args in sync with train_{modality}.py."""
    builders = {
        ('ECG', 'snn'):         lambda: ECGSNN(input_size=256, hidden1=256, hidden2=128, hidden3=64,
                                                n_classes=2, timesteps=64, beta=0.9, threshold=1.0, dropout=0.2),
        ('ECG', 'cnn'):         lambda: ECGCNN(input_size=256, n_classes=2, dropout=0.3),
        ('ECG', 'lstm'):        lambda: ECGLSTM(input_size=1, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3),
        ('ECG', 'transformer'): lambda: ECGTransformer(input_size=256, d_model=64, n_heads=4, n_layers=2,
                                                         dim_feedforward=128, n_classes=2),
        ('EEG', 'snn'):         lambda: EEGSNN(n_channels=18, input_size=64, hidden1=128, hidden2=64, hidden3=32,
                                                n_classes=2, timesteps=64, beta=0.9, threshold=1.0, dropout=0.2),
        ('EEG', 'cnn'):         lambda: EEGCNN(n_channels=18, window_size=256, n_classes=2, dropout=0.3),
        ('EEG', 'lstm'):        lambda: EEGLSTM(n_channels=18, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3),
        ('EEG', 'transformer'): lambda: EEGTransformer(n_channels=18, window_size=256, d_model=64, n_heads=4,
                                                         n_layers=3, n_classes=2, dropout=0.1),
        ('EMG', 'snn'):         lambda: EMGSNN(n_channels=12, window_size=256, hidden1=64, hidden2=32,
                                                n_classes=2, timesteps=32, beta=0.9, dropout=0.2),
        ('EMG', 'cnn'):         lambda: EMGCNN(n_channels=12, window_size=256, n_classes=2, dropout=0.3),
        ('EMG', 'lstm'):        lambda: EMGLSTM(n_channels=12, hidden_size=64, num_layers=2, n_classes=2, dropout=0.3),
        ('EMG', 'transformer'): lambda: EMGTransformer(n_channels=12, window_size=256, d_model=32, n_heads=4,
                                                         n_layers=2, n_classes=2),
    }
    key = (modality, arch)
    if key not in builders:
        raise ValueError(f"Unknown (modality, arch) pair: {key}")
    return builders[key]()


def load_checkpoint(modality: str, arch: str) -> nn.Module:
    name = f"{modality.lower()}_{arch}"
    path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    if not os.path.exists(path):
        return None
    model = build_model(modality, arch)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def load_test_data(modality: str):
    test_dir = os.path.join(DATA_DIR, modality.lower(), 'test')
    X = np.load(os.path.join(test_dir, 'X.npy'))
    y = np.load(os.path.join(test_dir, 'y.npy'))
    return X, y


def get_predictions(model: nn.Module, X: np.ndarray, batch_size: int = 256, noise_std: float = 0.0):
    """Returns predicted probabilities for the positive class."""
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            if noise_std > 0:
                batch = batch + np.random.normal(0, noise_std, batch.shape).astype(np.float32)
            x = torch.FloatTensor(batch).to(DEVICE)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


def evaluate_all():
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)

    modalities = ['ECG', 'EEG', 'EMG']
    architectures = ['snn', 'cnn', 'lstm', 'transformer']
    arch_display = {'snn': 'SNN', 'cnn': 'CNN', 'lstm': 'LSTM', 'transformer': 'Transformer'}

    all_metrics = {}
    loaded = {}  # (modality, arch) -> (model, X_test, y_test, y_prob)

    for modality in modalities:
        try:
            X_test, y_test = load_test_data(modality)
        except FileNotFoundError:
            print(f"[{modality}] test data not found at {DATA_DIR}/{modality.lower()}/test — skipping modality.")
            continue

        for arch in architectures:
            model = load_checkpoint(modality, arch)
            if model is None:
                print(f"[{modality}-{arch}] no checkpoint found — skipping.")
                continue
            y_prob = get_predictions(model, X_test)
            loaded[(modality, arch)] = (model, X_test, y_test, y_prob)
            print(f"[{modality}-{arch}] loaded and evaluated ({len(y_test)} test samples).")

    if not loaded:
        print("\nNo checkpoints found — nothing to evaluate. Train at least one model first.")
        return

    plt.style.use('dark_background')

    # ── ROC curves ───────────────────────────────────────────────────────
    present_modalities = sorted(set(m for m, a in loaded))
    fig, axes = plt.subplots(1, len(present_modalities), figsize=(6 * len(present_modalities), 5.5))
    if len(present_modalities) == 1:
        axes = [axes]
    fig.suptitle('ROC Curves — SNN vs. Baselines', fontsize=14, fontweight='bold')

    for ax, modality in zip(axes, present_modalities):
        for arch in architectures:
            if (modality, arch) not in loaded:
                continue
            _, _, y_test, y_prob = loaded[(modality, arch)]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            all_metrics.setdefault(modality, {}).setdefault(arch_display[arch], {})['roc_auc'] = float(roc_auc)
            ax.plot(fpr, tpr, color=COLORS[arch_display[arch]], linewidth=2,
                    label=f'{arch_display[arch]} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(modality)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('outputs/plots/09_roc_curves.png', dpi=150, facecolor='#0d1117')
    print("Saved: outputs/plots/09_roc_curves.png")
    plt.close()

    # ── Precision-Recall curves ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(present_modalities), figsize=(6 * len(present_modalities), 5.5))
    if len(present_modalities) == 1:
        axes = [axes]
    fig.suptitle('Precision-Recall Curves — SNN vs. Baselines', fontsize=14, fontweight='bold')

    for ax, modality in zip(axes, present_modalities):
        for arch in architectures:
            if (modality, arch) not in loaded:
                continue
            _, _, y_test, y_prob = loaded[(modality, arch)]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            all_metrics.setdefault(modality, {}).setdefault(arch_display[arch], {})['average_precision'] = float(ap)
            ax.plot(recall, precision, color=COLORS[arch_display[arch]], linewidth=2,
                    label=f'{arch_display[arch]} (AP={ap:.3f})')
        base_rate = np.mean(loaded[(modality, next(a for a in architectures if (modality, a) in loaded))][2])
        ax.axhline(base_rate, color='white', linestyle='--', alpha=0.3, linewidth=1, label=f'Baseline ({base_rate:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(modality)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('outputs/plots/10_pr_curves.png', dpi=150, facecolor='#0d1117')
    print("Saved: outputs/plots/10_pr_curves.png")
    plt.close()

    # ── Calibration diagrams ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(present_modalities), figsize=(6 * len(present_modalities), 5.5))
    if len(present_modalities) == 1:
        axes = [axes]
    fig.suptitle('Calibration (Reliability) Diagrams', fontsize=14, fontweight='bold')

    for ax, modality in zip(axes, present_modalities):
        for arch in architectures:
            if (modality, arch) not in loaded:
                continue
            _, _, y_test, y_prob = loaded[(modality, arch)]
            try:
                frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
                ax.plot(mean_pred, frac_pos, 'o-', color=COLORS[arch_display[arch]],
                        label=arch_display[arch], markersize=4)
            except ValueError:
                pass  # too few positive samples for the requested number of bins
        ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(modality)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('outputs/plots/11_calibration.png', dpi=150, facecolor='#0d1117')
    print("Saved: outputs/plots/11_calibration.png")
    plt.close()

    # ── Confusion matrices ──────────────────────────────────────────────
    n_present = len(loaded)
    n_cols = 4
    n_rows = int(np.ceil(n_present / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for idx, ((modality, arch), (_, _, y_test, y_prob)) in enumerate(loaded.items()):
        ax = axes[idx // n_cols, idx % n_cols]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap='viridis')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] < cm.max() / 2 else 'black', fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{modality}-{arch_display[arch]}', fontsize=10)

    for idx in range(n_present, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/plots/14_confusion_matrices.png', dpi=150, facecolor='#0d1117')
    print("Saved: outputs/plots/14_confusion_matrices.png")
    plt.close()

    # ── Noise robustness ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(present_modalities), figsize=(6 * len(present_modalities), 5.5))
    if len(present_modalities) == 1:
        axes = [axes]
    fig.suptitle('Robustness to Gaussian Noise', fontsize=14, fontweight='bold')

    for ax, modality in zip(axes, present_modalities):
        for arch in architectures:
            if (modality, arch) not in loaded:
                continue
            model, X_test, y_test, _ = loaded[(modality, arch)]
            accs = []
            for noise_std in NOISE_LEVELS:
                y_prob_noisy = get_predictions(model, X_test, noise_std=noise_std)
                y_pred_noisy = (y_prob_noisy >= 0.5).astype(int)
                acc = np.mean(y_pred_noisy == y_test)
                accs.append(acc)
            all_metrics.setdefault(modality, {}).setdefault(arch_display[arch], {})['noise_robustness'] = \
                dict(zip([str(n) for n in NOISE_LEVELS], [float(a) for a in accs]))
            ax.plot(NOISE_LEVELS, accs, 'o-', color=COLORS[arch_display[arch]], label=arch_display[arch])
        ax.set_xlabel('Injected Gaussian Noise (std)')
        ax.set_ylabel('Accuracy')
        ax.set_title(modality)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('outputs/plots/15_noise_robustness.png', dpi=150, facecolor='#0d1117')
    print("Saved: outputs/plots/15_noise_robustness.png")
    plt.close()

    with open('outputs/tables/advanced_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved: outputs/tables/advanced_metrics.json")


if __name__ == "__main__":
    evaluate_all()