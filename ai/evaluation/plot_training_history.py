"""
Training History Visualization for NeuroPulse AI
Plots loss curves and accuracy curves for all trained models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs/plots', exist_ok=True)

BG    = '#0D1117'
GRID  = '#21262D'
TEXT  = '#E6EDF3'
GREEN = '#00D4AA'
RED   = '#EF4444'
BLUE  = '#3B82F6'
AMBER = '#F59E0B'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.edgecolor':   GRID,
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'text.color':       TEXT,
    'grid.color':       GRID,
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})

MODELS = [
    ('ecg_snn',         'ECG-SNN',         GREEN),
    ('ecg_cnn',         'ECG-CNN',          '#7F77DD'),
    ('ecg_lstm',        'ECG-LSTM',         AMBER),
    ('ecg_transformer', 'ECG-Transformer',  BLUE),
    ('eeg_snn',         'EEG-SNN',          GREEN),
    ('eeg_cnn',         'EEG-CNN',          '#7F77DD'),
    ('eeg_lstm',        'EEG-LSTM',         AMBER),
    ('eeg_transformer', 'EEG-Transformer',  BLUE),
    ('emg_snn',         'EMG-SNN',          GREEN),
    ('emg_cnn',         'EMG-CNN',          '#7F77DD'),
    ('emg_lstm',        'EMG-LSTM',         AMBER),
    ('emg_transformer', 'EMG-Transformer',  BLUE),
]


def load_history(model_name):
    path = f'ai/saved_models/{model_name}_history.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════
# PLOT 1 — ECG Training Curves
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG)
fig.suptitle('ECG Model Training History',
             fontsize=16, color=TEXT, fontweight='bold')

ecg_models = [m for m in MODELS if m[0].startswith('ecg')]

for i, (model_name, label, color) in enumerate(ecg_models):
    history = load_history(model_name)
    if not history:
        continue

    # Loss plot
    ax_loss = axes[0, i]
    ax_loss.plot(history['train_losses'],
                 color=color, linewidth=1.5, label='Train')
    ax_loss.plot(history['val_losses'],
                 color=RED, linewidth=1.5,
                 linestyle='--', label='Val')
    ax_loss.set_title(f'{label} — Loss',
                      color=color, fontsize=10)
    ax_loss.set_xlabel('Epoch', fontsize=8)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # Accuracy plot
    ax_acc = axes[1, i]
    ax_acc.plot(history['train_accs'],
                color=color, linewidth=1.5, label='Train')
    ax_acc.plot(history['val_accs'],
                color=RED, linewidth=1.5,
                linestyle='--', label='Val')
    ax_acc.set_title(f'{label} — Accuracy',
                     color=color, fontsize=10)
    ax_acc.set_xlabel('Epoch', fontsize=8)
    ax_acc.set_ylabel('Accuracy', fontsize=8)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = history.get('best_epoch', 0)
    best_acc   = history.get('best_val_acc', 0)
    ax_acc.axvline(x=best_epoch, color='white',
                   linestyle=':', alpha=0.5)
    ax_acc.text(best_epoch, best_acc,
                f'Best\n{best_acc:.3f}',
                color='white', fontsize=7,
                ha='left', va='bottom')

plt.tight_layout()
plt.savefig('outputs/plots/09_ecg_training_history.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: outputs/plots/09_ecg_training_history.png")

# ══════════════════════════════════════════════════════
# PLOT 2 — EEG Training Curves
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG)
fig.suptitle('EEG Model Training History',
             fontsize=16, color=TEXT, fontweight='bold')

eeg_models = [m for m in MODELS if m[0].startswith('eeg')]

for i, (model_name, label, color) in enumerate(eeg_models):
    history = load_history(model_name)
    if not history:
        continue

    ax_loss = axes[0, i]
    ax_loss.plot(history['train_losses'],
                 color=color, linewidth=1.5, label='Train')
    ax_loss.plot(history['val_losses'],
                 color=RED, linewidth=1.5,
                 linestyle='--', label='Val')
    ax_loss.set_title(f'{label} — Loss',
                      color=color, fontsize=10)
    ax_loss.set_xlabel('Epoch', fontsize=8)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc = axes[1, i]
    ax_acc.plot(history['train_accs'],
                color=color, linewidth=1.5, label='Train')
    ax_acc.plot(history['val_accs'],
                color=RED, linewidth=1.5,
                linestyle='--', label='Val')
    ax_acc.set_title(f'{label} — Accuracy',
                     color=color, fontsize=10)
    ax_acc.set_xlabel('Epoch', fontsize=8)
    ax_acc.set_ylabel('Accuracy', fontsize=8)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/10_eeg_training_history.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: outputs/plots/10_eeg_training_history.png")

# ══════════════════════════════════════════════════════
# PLOT 3 — EMG Training Curves
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG)
fig.suptitle('EMG Model Training History',
             fontsize=16, color=TEXT, fontweight='bold')

emg_models = [m for m in MODELS if m[0].startswith('emg')]

for i, (model_name, label, color) in enumerate(emg_models):
    history = load_history(model_name)
    if not history:
        continue

    ax_loss = axes[0, i]
    ax_loss.plot(history['train_losses'],
                 color=color, linewidth=1.5, label='Train')
    ax_loss.plot(history['val_losses'],
                 color=RED, linewidth=1.5,
                 linestyle='--', label='Val')
    ax_loss.set_title(f'{label} — Loss',
                      color=color, fontsize=10)
    ax_loss.set_xlabel('Epoch', fontsize=8)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc = axes[1, i]
    ax_acc.plot(history['train_accs'],
                color=color, linewidth=1.5, label='Train')
    ax_acc.plot(history['val_accs'],
                color=RED, linewidth=1.5,
                linestyle='--', label='Val')
    ax_acc.set_title(f'{label} — Accuracy',
                     color=color, fontsize=10)
    ax_acc.set_xlabel('Epoch', fontsize=8)
    ax_acc.set_ylabel('Accuracy', fontsize=8)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/11_emg_training_history.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: outputs/plots/11_emg_training_history.png")

# ══════════════════════════════════════════════════════
# PLOT 4 — Final Comparison Radar Chart
# ══════════════════════════════════════════════════════
import json

with open('ai/saved_models/ecg_benchmark_results.json') as f:
    ecg_r = json.load(f)
with open('ai/saved_models/eeg_benchmark_results.json') as f:
    eeg_r = json.load(f)
with open('ai/saved_models/emg_benchmark_results.json') as f:
    emg_r = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                          subplot_kw=dict(polar=True),
                          facecolor=BG)
fig.suptitle('NeuroPulse AI — Model Performance Radar',
             fontsize=16, color=TEXT, fontweight='bold')

categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]

for ax, (title, results, color) in zip(axes, [
    ('ECG', ecg_r, '#00D4AA'),
    ('EEG', eeg_r, '#7F77DD'),
    ('EMG', emg_r, '#F59E0B'),
]):
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        color=TEXT, size=7)
    ax.grid(color=GRID, alpha=0.5)

    plot_colors = [GREEN, '#7F77DD', AMBER, BLUE]

    for i, (model_name, metrics) in enumerate(results.items()):
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['auc']
        ]
        values += values[:1]
        c = plot_colors[i % len(plot_colors)]
        ax.plot(angles, values, 'o-',
                linewidth=2, color=c,
                label=model_name.split('-')[1])
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_title(title, color=color, size=12,
                 pad=15, fontweight='bold')
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.3, 1.1),
              fontsize=8)

plt.tight_layout()
plt.savefig('outputs/plots/12_radar_comparison.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: outputs/plots/12_radar_comparison.png")

print("\nAll training history plots generated!")
print("Total plots in outputs/plots/: 12")