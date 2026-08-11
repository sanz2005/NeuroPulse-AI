"""
LSTM Baseline Model for NeuroPulse AI
Bidirectional LSTM for ECG, EEG, and EMG classification.
Used for benchmarking against SNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGLSTM(nn.Module):
    """
    Bidirectional LSTM baseline for ECG arrhythmia detection.
    Input:  [batch, 256] — treated as sequence of 256 timesteps
    Output: [batch, 2]
    """

    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(ECGLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 256]
        Returns:
            logits: [batch, n_classes]
        """
        # Reshape to sequence
        # [batch, 256] → [batch, 256, 1]
        x = x.unsqueeze(2)

        # LSTM forward
        # output: [batch, seq_len, hidden*2]
        output, _ = self.lstm(x)

        # Attention mechanism
        attn_weights = self.attention(output)       # [batch, seq, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context      = (output * attn_weights).sum(dim=1)  # [batch, hidden*2]

        # Classification
        logits = self.classifier(context)
        return logits


class EEGLSTM(nn.Module):
    """
    Bidirectional LSTM for EEG seizure detection.
    Input:  [batch, 18, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 18,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(EEGLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Spatial mixing across channels
        self.spatial_fc = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.LayerNorm(64),
            nn.ELU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
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
        # x: [batch, 18, 256] → [batch, 256, 18]
        x = x.permute(0, 2, 1)

        # Spatial mixing per timestep
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)           # [batch*seq, channels]
        x = self.spatial_fc(x)          # [batch*seq, 64]
        x = x.reshape(batch, seq, -1)   # [batch, seq, 64]

        # LSTM
        output, _ = self.lstm(x)        # [batch, seq, hidden*2]

        # Attention
        attn_weights = self.attention(output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context      = (output * attn_weights).sum(dim=1)

        logits = self.classifier(context)
        return logits


class EMGLSTM(nn.Module):
    """
    Bidirectional LSTM for EMG anomaly detection.
    Input:  [batch, 12, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 12,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 n_classes: int = 2,
                 dropout: float = 0.3):
        super(EMGLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Channel projection
        self.channel_proj = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.LayerNorm(32),
            nn.ELU()
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            logits: [batch, n_classes]
        """
        # [batch, 12, 256] → [batch, 256, 12]
        x = x.permute(0, 2, 1)

        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = self.channel_proj(x)
        x = x.reshape(batch, seq, -1)

        output, (hn, _) = self.lstm(x)

        # Use final hidden state
        # hn: [num_layers*2, batch, hidden]
        # Take last layer, both directions
        forward_h  = hn[-2]   # [batch, hidden]
        backward_h = hn[-1]   # [batch, hidden]
        context    = torch.cat([forward_h, backward_h], dim=1)

        logits = self.classifier(context)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 8

    # Test ECG LSTM
    print("Testing ECG LSTM...")
    model = ECGLSTM(hidden_size=128, num_layers=2, n_classes=2)
    x     = torch.randn(batch_size, 256)
    out   = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EEG LSTM
    print("\nTesting EEG LSTM...")
    model = EEGLSTM(n_channels=18, hidden_size=128, n_classes=2)
    x     = torch.randn(batch_size, 18, 256)
    out   = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EMG LSTM
    print("\nTesting EMG LSTM...")
    model = EMGLSTM(n_channels=12, hidden_size=64, n_classes=2)
    x     = torch.randn(batch_size, 12, 256)
    out   = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    print("\nAll LSTM baseline tests passed!")