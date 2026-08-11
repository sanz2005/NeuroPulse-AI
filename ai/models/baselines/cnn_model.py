"""
CNN Baseline Model for NeuroPulse AI
Used for benchmarking against SNN models.
Implements 1D CNN for ECG, EEG, and EMG classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    """
    1D CNN baseline for ECG arrhythmia detection.
    Input:  [batch, 1, 256] — 1 channel, 256 samples
    Output: [batch, 2]
    """

    def __init__(self,
                 input_size: int = 256,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(ECGCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 → 128

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 → 64

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 → 32

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # → 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 256] — will be reshaped to [batch, 1, 256]
        Returns:
            logits: [batch, n_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 256]
        x = self.features(x)
        x = self.classifier(x)
        return x


class EEGCNN(nn.Module):
    """
    2D CNN baseline for EEG seizure detection.
    Treats EEG as [channels x time] 2D input.
    Input:  [batch, 18, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 18,
                 window_size: int = 256,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(EEGCNN, self).__init__()

        self.features = nn.Sequential(
            # Temporal convolutions
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),

            # Spatial convolutions (across channels)
            nn.Conv2d(32, 64, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # time: 256 → 64

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # time: 64 → 16

            nn.AdaptiveAvgPool2d((1, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            logits: [batch, n_classes]
        """
        x = x.unsqueeze(1)  # [batch, 1, channels, window_size]
        x = self.features(x)
        x = self.classifier(x)
        return x


class EMGCNN(nn.Module):
    """
    1D CNN baseline for EMG anomaly detection.
    Input:  [batch, 12, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 12,
                 window_size: int = 256,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(EMGCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 → 128

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 → 64

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            logits: [batch, n_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 8

    # Test ECG CNN
    print("Testing ECG CNN...")
    model = ECGCNN(input_size=256, n_classes=2)
    x = torch.randn(batch_size, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EEG CNN
    print("\nTesting EEG CNN...")
    model = EEGCNN(n_channels=18, window_size=256, n_classes=2)
    x = torch.randn(batch_size, 18, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EMG CNN
    print("\nTesting EMG CNN...")
    model = EMGCNN(n_channels=12, window_size=256, n_classes=2)
    x = torch.randn(batch_size, 12, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    print("\nAll CNN baseline tests passed!")