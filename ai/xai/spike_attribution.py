"""
Spike Attribution XAI Module for NeuroPulse AI
Explains SNN predictions using spike activity analysis.
Shows WHICH parts of the biosignal caused the anomaly detection.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('.')


class SpikeAttributionAnalyzer:
    """
    Analyzes spike attribution for SNN models.
    Uses gradient-based and activation-based methods
    to explain which signal regions caused anomaly detection.

    Args:
        model:      Trained SNN model
        model_name: Name for saving outputs
        device:     'cpu' or 'cuda'
    """

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 device: str = 'cpu'):
        self.model      = model.to(device)
        self.model_name = model_name
        self.device     = device
        self.model.eval()

        os.makedirs('outputs/xai', exist_ok=True)

    def get_spike_attribution(self,
                               x: np.ndarray) -> np.ndarray:
        """
        Get temporal spike attribution map.
        Shows spike activity at each timestep.

        Args:
            x: Input signal [samples] or [channels, samples]

        Returns:
            attribution: [timesteps] normalized attribution map
        """
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            attribution = self.model.get_spike_attribution(x_tensor)

        attribution = attribution.cpu().numpy().squeeze()

        # Normalize to [0, 1]
        if attribution.max() > attribution.min():
            attribution = ((attribution - attribution.min()) /
                           (attribution.max() - attribution.min()))

        return attribution

    def get_gradient_attribution(self,
                                  x: np.ndarray,
                                  target_class: int = 1) -> np.ndarray:
        """
        Gradient-based attribution using backpropagation.
        Computes gradient of target class score w.r.t. input.

        Args:
            x:            Input signal
            target_class: Class to explain (1 = anomaly)

        Returns:
            attribution: Gradient magnitude map
        """
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        x_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(x_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Get gradient magnitude
        gradients  = x_tensor.grad.cpu().numpy().squeeze()
        attribution = np.abs(gradients)

        # Normalize
        if attribution.max() > attribution.min():
            attribution = ((attribution - attribution.min()) /
                           (attribution.max() - attribution.min()))

        return attribution

    def analyze_batch(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       n_samples: int = 5) -> dict:
        """
        Analyze multiple samples and return attribution maps.

        Args:
            X:         Input signals [batch, ...]
            y:         Labels [batch]
            n_samples: Number of samples to analyze

        Returns:
            results: dict with normal and anomaly attributions
        """
        normal_idx  = np.where(y == 0)[0][:n_samples]
        anomaly_idx = np.where(y == 1)[0][:n_samples]

        results = {
            'normal_attributions':  [],
            'anomaly_attributions': [],
            'normal_signals':       [],
            'anomaly_signals':      [],
        }

        for idx in normal_idx:
            attr = self.get_spike_attribution(X[idx])
            results['normal_attributions'].append(attr)
            results['normal_signals'].append(X[idx])

        for idx in anomaly_idx:
            attr = self.get_spike_attribution(X[idx])
            results['anomaly_attributions'].append(attr)
            results['anomaly_signals'].append(X[idx])

        return results

    def plot_ecg_attribution(self,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              n_samples: int = 3):
        """
        Plot ECG signal with spike attribution heatmap overlay.
        Shows which parts of ECG signal triggered anomaly detection.
        """
        BG    = '#0D1117'
        TEXT  = '#E6EDF3'
        GRID  = '#21262D'
        GREEN = '#00D4AA'
        RED   = '#EF4444'

        plt.rcParams.update({
            'figure.facecolor': BG,
            'axes.facecolor':   BG,
            'axes.edgecolor':   GRID,
            'axes.labelcolor':  TEXT,
            'xtick.color':      TEXT,
            'ytick.color':      TEXT,
            'text.color':       TEXT,
        })

        # Get samples
        normal_idx  = np.where(y_test == 0)[0][:n_samples]
        anomaly_idx = np.where(y_test == 1)[0][:n_samples]

        fig = plt.figure(figsize=(18, 12), facecolor=BG)
        fig.suptitle(
            f'NeuroPulse AI — {self.model_name} Spike Attribution (XAI)',
            fontsize=14, color=TEXT, fontweight='bold'
        )

        n_rows = n_samples * 2
        gs     = gridspec.GridSpec(n_rows, 2, figure=fig,
                                    hspace=0.5, wspace=0.3)

        for i, (idx, label, color, title) in enumerate([
            *[(idx, 0, GREEN, 'Normal') for idx in normal_idx],
            *[(idx, 1, RED, 'Anomaly') for idx in anomaly_idx]
        ]):
            row = i
            if row >= n_rows:
                break

            signal      = X_test[idx]
            attribution = self.get_spike_attribution(signal)

            # Handle 2D signals (EEG/EMG)
            if signal.ndim == 2:
                signal_1d = signal[0]
            else:
                signal_1d = signal

            # Resample attribution to signal length
            from scipy.signal import resample
            attr_resampled = resample(attribution, len(signal_1d))
            attr_resampled = np.clip(attr_resampled, 0, 1)

            # Signal plot
            ax_sig = fig.add_subplot(gs[row, 0])
            ax_sig.plot(signal_1d, color=color, linewidth=1.2)
            ax_sig.set_title(f'{title} — Signal',
                             color=color, fontsize=9)
            ax_sig.set_ylabel('Amplitude', fontsize=8)
            ax_sig.grid(True, alpha=0.3)
            ax_sig.set_xlim(0, len(signal_1d))

            # Attribution heatmap
            ax_attr = fig.add_subplot(gs[row, 1])

            # Create heatmap overlay
            x_pos  = np.arange(len(attr_resampled))
            ax_attr.fill_between(
                x_pos, attr_resampled,
                alpha=0.7,
                color=RED if label == 1 else GREEN
            )
            ax_attr.plot(attr_resampled,
                         color=RED if label == 1 else GREEN,
                         linewidth=1.0)
            ax_attr.set_title(
                f'{title} — Spike Attribution Map',
                color=RED if label == 1 else GREEN,
                fontsize=9
            )
            ax_attr.set_ylabel('Attribution', fontsize=8)
            ax_attr.set_ylim(0, 1.1)
            ax_attr.grid(True, alpha=0.3)
            ax_attr.set_xlim(0, len(attr_resampled))

            # Highlight high attribution regions
            threshold = 0.7
            high_attr = attr_resampled > threshold
            for j in range(len(high_attr)):
                if high_attr[j]:
                    ax_attr.axvspan(j, j+1,
                                    alpha=0.3,
                                    color=RED,
                                    linewidth=0)

        save_path = f'outputs/xai/{self.model_name}_attribution.png'
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_temporal_heatmap(self,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               n_samples: int = 10):
        """
        Plot temporal heatmap showing attribution
        across multiple samples — rows = samples,
        cols = timesteps, color = attribution strength.
        """
        BG   = '#0D1117'
        TEXT = '#E6EDF3'
        GRID = '#21262D'

        # Get equal samples of each class
        normal_idx  = np.where(y_test == 0)[0][:n_samples]
        anomaly_idx = np.where(y_test == 1)[0][:n_samples]

        all_idx    = np.concatenate([normal_idx, anomaly_idx])
        all_labels = np.concatenate([
            np.zeros(len(normal_idx)),
            np.ones(len(anomaly_idx))
        ])

        # Compute attributions
        attributions = []
        for idx in all_idx:
            attr = self.get_spike_attribution(X_test[idx])
            # Normalize length to 64
            from scipy.signal import resample
            attr = resample(attr, 64)
            attr = np.clip(attr, 0, 1)
            attributions.append(attr)

        attributions = np.array(attributions)

        fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
        fig.suptitle(
            f'NeuroPulse AI — Temporal Spike Attribution Heatmap',
            fontsize=14, color=TEXT, fontweight='bold'
        )

        im = ax.imshow(
            attributions,
            aspect='auto',
            cmap='RdYlGn_r',
            vmin=0, vmax=1
        )

        # Add class labels on y-axis
        y_labels = [
            f"{'Normal' if l == 0 else 'Anomaly'} {i}"
            for i, l in enumerate(all_labels)
        ]
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8, color=TEXT)
        ax.set_xlabel('Timestep', fontsize=10, color=TEXT)
        ax.set_title('Red = High Attribution (Anomaly Region)',
                     color=TEXT, fontsize=10)

        # Separator line between normal and anomaly
        ax.axhline(y=len(normal_idx) - 0.5,
                   color='white', linewidth=2, linestyle='--')
        ax.text(65, len(normal_idx)/2, 'NORMAL',
                color='#00D4AA', fontsize=9,
                va='center', ha='left')
        ax.text(65, len(normal_idx) + len(anomaly_idx)/2,
                'ANOMALY', color='#EF4444', fontsize=9,
                va='center', ha='left')

        plt.colorbar(im, ax=ax, label='Attribution Strength')

        save_path = f'outputs/xai/{self.model_name}_heatmap.png'
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    import numpy as np
    sys.path.append('.')

    from ai.models.snn.ecg_snn import ECGSNN
    from ai.models.snn.eeg_snn import EEGSNN
    from ai.models.snn.emg_snn import EMGSNN

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('outputs/xai', exist_ok=True)

    print("Generating XAI Attribution Maps...")
    print(f"Device: {device}")

    # ── ECG XAI ────────────────────────────────────────────────────────────────
    print("\n1. ECG Spike Attribution...")
    X_test = np.load('data/processed/ecg/test/X.npy')
    y_test = np.load('data/processed/ecg/test/y.npy')

    ecg_model = ECGSNN(input_size=256, hidden1=256,
                        hidden2=128, hidden3=64,
                        n_classes=2, timesteps=64)

    checkpoint = torch.load('ai/saved_models/ecg_snn_best.pth',
                             map_location=device)
    ecg_model.load_state_dict(checkpoint['model_state_dict'])
    ecg_model.eval()

    ecg_xai = SpikeAttributionAnalyzer(ecg_model, 'ECG_SNN', device)
    ecg_xai.plot_ecg_attribution(X_test, y_test, n_samples=3)
    ecg_xai.plot_temporal_heatmap(X_test, y_test, n_samples=8)
    print("ECG XAI complete!")

    # ── EEG XAI ────────────────────────────────────────────────────────────────
    print("\n2. EEG Spike Attribution...")
    X_test_eeg = np.load('data/processed/eeg/test/X.npy')
    y_test_eeg = np.load('data/processed/eeg/test/y.npy')

    eeg_model = EEGSNN(n_channels=18, input_size=64,
                        hidden1=128, hidden2=64, hidden3=32,
                        n_classes=2, timesteps=64)

    checkpoint = torch.load('ai/saved_models/eeg_snn_best.pth',
                             map_location=device)
    eeg_model.load_state_dict(checkpoint['model_state_dict'])
    eeg_model.eval()

    eeg_xai = SpikeAttributionAnalyzer(eeg_model, 'EEG_SNN', device)
    eeg_xai.plot_ecg_attribution(X_test_eeg, y_test_eeg, n_samples=3)
    eeg_xai.plot_temporal_heatmap(X_test_eeg, y_test_eeg, n_samples=8)
    print("EEG XAI complete!")

    # ── EMG XAI ────────────────────────────────────────────────────────────────
    print("\n3. EMG Spike Attribution...")
    X_test_emg = np.load('data/processed/emg/test/X.npy')
    y_test_emg = np.load('data/processed/emg/test/y.npy')

    emg_model = EMGSNN(n_channels=12, window_size=256,
                        hidden1=64, hidden2=32,
                        n_classes=2, timesteps=32)

    checkpoint = torch.load('ai/saved_models/emg_snn_best.pth',
                             map_location=device)
    emg_model.load_state_dict(checkpoint['model_state_dict'])
    emg_model.eval()

    emg_xai = SpikeAttributionAnalyzer(emg_model, 'EMG_SNN', device)
    emg_xai.plot_ecg_attribution(X_test_emg, y_test_emg, n_samples=3)
    emg_xai.plot_temporal_heatmap(X_test_emg, y_test_emg, n_samples=8)
    print("EMG XAI complete!")

    print("\nAll XAI outputs generated!")
    print("Location: outputs/xai/")