"""
NeuroPulse AI — Data Visualization Script
Generates presentation-ready plots from preprocessed data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Output folder for plots ────────────────────────────────────────────────────
os.makedirs('outputs/plots', exist_ok=True)

# ── Load preprocessed data ─────────────────────────────────────────────────────
print("Loading preprocessed data...")
ecg_X = np.load('data/processed/ecg/train/X.npy')
ecg_y = np.load('data/processed/ecg/train/y.npy')
eeg_X = np.load('data/processed/eeg/train/X.npy')
eeg_y = np.load('data/processed/eeg/train/y.npy')
emg_X = np.load('data/processed/emg/train/X.npy')
emg_y = np.load('data/processed/emg/train/y.npy')
print(f"ECG: {ecg_X.shape} | EEG: {eeg_X.shape} | EMG: {emg_X.shape}")

# ── Color scheme ───────────────────────────────────────────────────────────────
ECG_COLOR   = '#00D4AA'
EEG_COLOR   = '#7F77DD'
EMG_COLOR   = '#F59E0B'
ALERT_COLOR = '#EF4444'
BG_COLOR    = '#0D1117'
GRID_COLOR  = '#21262D'
TEXT_COLOR  = '#E6EDF3'

plt.rcParams.update({
    'figure.facecolor':  BG_COLOR,
    'axes.facecolor':    BG_COLOR,
    'axes.edgecolor':    GRID_COLOR,
    'axes.labelcolor':   TEXT_COLOR,
    'xtick.color':       TEXT_COLOR,
    'ytick.color':       TEXT_COLOR,
    'text.color':        TEXT_COLOR,
    'grid.color':        GRID_COLOR,
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Multi-Modal Biosignal Overview
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Plot 1: Multi-Modal Biosignal Overview...")

fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle('NeuroPulse AI — Multi-Modal Biosignal Overview',
             fontsize=16, color=TEXT_COLOR, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

# ECG Normal
ax1 = fig.add_subplot(gs[0, 0])
normal_idx = np.where(ecg_y == 0)[0][0]
ax1.plot(ecg_X[normal_idx], color=ECG_COLOR, linewidth=1.5)
ax1.set_title('ECG — Normal Beat', color=ECG_COLOR, fontsize=11, pad=8)
ax1.set_xlabel('Samples (256Hz)', fontsize=9)
ax1.set_ylabel('Amplitude (normalized)', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color=GRID_COLOR, linewidth=0.5)

# ECG Arrhythmia
ax2 = fig.add_subplot(gs[0, 1])
anomaly_idx = np.where(ecg_y == 1)[0][0]
ax2.plot(ecg_X[anomaly_idx], color=ALERT_COLOR, linewidth=1.5)
ax2.set_title('ECG — Arrhythmia Detected', color=ALERT_COLOR, fontsize=11, pad=8)
ax2.set_xlabel('Samples (256Hz)', fontsize=9)
ax2.set_ylabel('Amplitude (normalized)', fontsize=9)
ax2.grid(True, alpha=0.3)

# EEG Interictal
ax3 = fig.add_subplot(gs[1, 0])
eeg_normal_idx = np.where(eeg_y == 0)[0][0]
for ch in range(min(5, eeg_X.shape[1])):
    offset = ch * 2
    ax3.plot(eeg_X[eeg_normal_idx, ch] + offset,
             color=EEG_COLOR, linewidth=0.8, alpha=0.8)
ax3.set_title('EEG — Normal (Interictal)', color=EEG_COLOR, fontsize=11, pad=8)
ax3.set_xlabel('Samples (256Hz)', fontsize=9)
ax3.set_ylabel('Channels (offset)', fontsize=9)
ax3.grid(True, alpha=0.3)

# EEG Seizure
ax4 = fig.add_subplot(gs[1, 1])
eeg_seizure_indices = np.where(eeg_y == 1)[0]
if len(eeg_seizure_indices) > 0:
    eeg_seizure_idx = eeg_seizure_indices[0]
    for ch in range(min(5, eeg_X.shape[1])):
        offset = ch * 2
        ax4.plot(eeg_X[eeg_seizure_idx, ch] + offset,
                 color=ALERT_COLOR, linewidth=0.8, alpha=0.8)
    ax4.set_title('EEG — Seizure Activity', color=ALERT_COLOR, fontsize=11, pad=8)
else:
    ax4.text(0.5, 0.5, 'No seizure samples\nin training set',
             ha='center', va='center', color=TEXT_COLOR, transform=ax4.transAxes)
    ax4.set_title('EEG — Seizure Activity', color=ALERT_COLOR, fontsize=11, pad=8)
ax4.set_xlabel('Samples (256Hz)', fontsize=9)
ax4.set_ylabel('Channels (offset)', fontsize=9)
ax4.grid(True, alpha=0.3)

# EMG Normal
ax5 = fig.add_subplot(gs[2, 0])
emg_normal_idx = np.where(emg_y == 0)[0][0]
for ch in range(min(4, emg_X.shape[1])):
    offset = ch * 3
    ax5.plot(emg_X[emg_normal_idx, ch] + offset,
             color=EMG_COLOR, linewidth=0.8, alpha=0.8)
ax5.set_title('EMG — Normal Muscle Activity', color=EMG_COLOR, fontsize=11, pad=8)
ax5.set_xlabel('Samples (256Hz)', fontsize=9)
ax5.set_ylabel('Channels (offset)', fontsize=9)
ax5.grid(True, alpha=0.3)

# EMG Anomaly
ax6 = fig.add_subplot(gs[2, 1])
emg_anomaly_idx = np.where(emg_y == 1)[0][0]
for ch in range(min(4, emg_X.shape[1])):
    offset = ch * 3
    ax6.plot(emg_X[emg_anomaly_idx, ch] + offset,
             color=ALERT_COLOR, linewidth=0.8, alpha=0.8)
ax6.set_title('EMG — Anomaly Detected', color=ALERT_COLOR, fontsize=11, pad=8)
ax6.set_xlabel('Samples (256Hz)', fontsize=9)
ax6.set_ylabel('Channels (offset)', fontsize=9)
ax6.grid(True, alpha=0.3)

plt.savefig('outputs/plots/01_biosignal_overview.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  Saved: outputs/plots/01_biosignal_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Dataset Statistics
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Plot 2: Dataset Statistics...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG_COLOR)
fig.suptitle('NeuroPulse AI — Dataset Distribution',
             fontsize=16, color=TEXT_COLOR, fontweight='bold')

datasets = [
    ('ECG\nMIT-BIH', ecg_y, ECG_COLOR),
    ('EEG\nCHB-MIT', eeg_y, EEG_COLOR),
    ('EMG\nNinaPro', emg_y, EMG_COLOR),
]

for ax, (name, labels, color) in zip(axes, datasets):
    normal  = np.sum(labels == 0)
    anomaly = np.sum(labels == 1)
    total   = len(labels)

    bars = ax.bar(['Normal', 'Anomaly'],
                  [normal, anomaly],
                  color=[color, ALERT_COLOR],
                  width=0.5, edgecolor=GRID_COLOR, linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, [normal, anomaly]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total*0.01,
                f'{val:,}\n({val/total*100:.1f}%)',
                ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=10, fontweight='bold')

    ax.set_title(name, color=color, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Number of Samples', fontsize=10)
    ax.set_ylim(0, max(normal, anomaly) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.98, 0.98, f'Total: {total:,}',
            transform=ax.transAxes,
            ha='right', va='top',
            color=TEXT_COLOR, fontsize=9,
            bbox=dict(boxstyle='round', facecolor=GRID_COLOR, alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/plots/02_dataset_distribution.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  Saved: outputs/plots/02_dataset_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Spike Train Visualization
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Plot 3: Spike Train Visualization...")

import sys
sys.path.append('.')
from ai.encoding.delta_modulation import delta_modulation_encode
from ai.encoding.rate_encoding import rate_encode_batch

fig, axes = plt.subplots(4, 1, figsize=(16, 12), facecolor=BG_COLOR)
fig.suptitle('NeuroPulse AI — Spike Train Encoding Visualization',
             fontsize=16, color=TEXT_COLOR, fontweight='bold')

# Raw ECG signal
ecg_sample = ecg_X[normal_idx]
axes[0].plot(ecg_sample, color=ECG_COLOR, linewidth=1.5)
axes[0].set_title('Raw ECG Signal (Normal Beat)', color=ECG_COLOR, fontsize=11)
axes[0].set_ylabel('Amplitude', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, len(ecg_sample))

# ECG Spike train
ecg_spikes = delta_modulation_encode(ecg_sample, threshold=0.05)
spike_times = np.where(ecg_spikes > 0)[0]
axes[1].vlines(spike_times, 0, 1,
               color=ECG_COLOR, linewidth=1.2, alpha=0.8)
axes[1].set_title(
    f'ECG Spike Train — Delta Modulation '
    f'(Spikes: {len(spike_times)} | '
    f'Sparsity: {(1-ecg_spikes.mean())*100:.1f}%)',
    color=ECG_COLOR, fontsize=11
)
axes[1].set_ylabel('Spike', fontsize=9)
axes[1].set_ylim(-0.1, 1.3)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, len(ecg_spikes))

# Raw EEG (single channel)
eeg_sample = eeg_X[eeg_normal_idx, 0]
axes[2].plot(eeg_sample, color=EEG_COLOR, linewidth=1.2)
axes[2].set_title('Raw EEG Signal (Channel 1 — Interictal)',
                  color=EEG_COLOR, fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, len(eeg_sample))

# EEG Spike train
eeg_spikes  = delta_modulation_encode(eeg_sample, threshold=0.05)
spike_times = np.where(eeg_spikes > 0)[0]
axes[3].vlines(spike_times, 0, 1,
               color=EEG_COLOR, linewidth=1.2, alpha=0.8)
axes[3].set_title(
    f'EEG Spike Train — Delta Modulation '
    f'(Spikes: {len(spike_times)} | '
    f'Sparsity: {(1-eeg_spikes.mean())*100:.1f}%)',
    color=EEG_COLOR, fontsize=11
)
axes[3].set_ylabel('Spike', fontsize=9)
axes[3].set_ylim(-0.1, 1.3)
axes[3].set_xlabel('Samples', fontsize=9)
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, len(eeg_spikes))

plt.tight_layout()
plt.savefig('outputs/plots/03_spike_train_visualization.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  Saved: outputs/plots/03_spike_train_visualization.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Summary Stats Card
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Plot 4: Summary Stats Card...")

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.axis('off')

fig.suptitle('NeuroPulse AI — Dataset Summary',
             fontsize=18, color=TEXT_COLOR, fontweight='bold', y=0.95)

stats = [
    ['Modality',  'Dataset',   'Total',    'Normal',   'Anomaly',  'Train',   'Val',     'Test'],
    ['ECG',       'MIT-BIH',   '109,438',  '75,011',   '34,427',  '76,608',  '10,943',  '21,887'],
    ['EEG',       'CHB-MIT',   '28,792',   '28,656',   '136',     '20,155',  '2,879',   '5,758'],
    ['EMG',       'NinaPro',   '57,316',   '37,228',   '20,088',  '40,122',  '5,731',   '11,463'],
    ['TOTAL',     '3 Sources', '195,546',  '140,895',  '54,651',  '136,885', '19,553',  '39,108'],
]

colors_row = [
    ['#1A56DB'] * 8,
    [BG_COLOR, '#0D2137', ECG_COLOR+'33', ECG_COLOR+'22', ALERT_COLOR+'22',
     ECG_COLOR+'11', ECG_COLOR+'11', ECG_COLOR+'11'],
    [BG_COLOR, '#0D2137', EEG_COLOR+'33', EEG_COLOR+'22', ALERT_COLOR+'22',
     EEG_COLOR+'11', EEG_COLOR+'11', EEG_COLOR+'11'],
    [BG_COLOR, '#0D2137', EMG_COLOR+'33', EMG_COLOR+'22', ALERT_COLOR+'22',
     EMG_COLOR+'11', EMG_COLOR+'11', EMG_COLOR+'11'],
    ['#1D6F42'] * 8,
]

table = ax.table(
    cellText=stats[1:],
    colLabels=stats[0],
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.05, 1.0, 0.85]
)

table.auto_set_font_size(False)
table.set_fontsize(12)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(colors_row[row][col] if row < len(colors_row) else BG_COLOR)
    cell.set_text_props(color=TEXT_COLOR, fontweight='bold' if row == 0 else 'normal')
    cell.set_edgecolor(GRID_COLOR)
    cell.set_linewidth(1)

plt.savefig('outputs/plots/04_dataset_summary.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  Saved: outputs/plots/04_dataset_summary.png")

print("\n" + "="*50)
print("All plots generated successfully!")
print("Location: outputs/plots/")
print("="*50)
print("\nFiles ready for presentation:")
print("  01_biosignal_overview.png     — ECG/EEG/EMG signal comparisons")
print("  02_dataset_distribution.png   — Class distribution bar charts")
print("  03_spike_train_visualization.png — Raw signal vs spike trains")
print("  04_dataset_summary.png        — Complete dataset stats table")