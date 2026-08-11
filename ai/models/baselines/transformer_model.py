"""
Transformer Baseline Model for NeuroPulse AI
Transformer encoder for ECG, EEG, and EMG classification.
Used for benchmarking against SNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds position information to input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encoding matrix
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ECGTransformer(nn.Module):
    """
    Transformer encoder baseline for ECG arrhythmia detection.
    Input:  [batch, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 input_size: int = 256,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dim_feedforward: int = 128,
                 n_classes: int = 2,
                 dropout: float = 0.1):
        super(ECGTransformer, self).__init__()

        self.d_model = d_model

        # Project each timestep value to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=input_size + 1,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 256]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size = x.shape[0]

        # [batch, 256] → [batch, 256, 1] → [batch, 256, d_model]
        x = x.unsqueeze(2)
        x = self.input_projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 257, d_model]

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)  # [batch, 257, d_model]

        # Use CLS token output for classification
        cls_output = x[:, 0, :]  # [batch, d_model]

        logits = self.classifier(cls_output)
        return logits


class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG seizure detection.
    Input:  [batch, 18, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 18,
                 window_size: int = 256,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dim_feedforward: int = 128,
                 n_classes: int = 2,
                 dropout: float = 0.1):
        super(EEGTransformer, self).__init__()

        self.d_model = d_model

        # Project channels to d_model
        self.channel_projection = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=window_size + 1,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
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
        batch_size = x.shape[0]

        # [batch, 18, 256] → [batch, 256, 18]
        x = x.permute(0, 2, 1)

        # Project channels → d_model
        # [batch, 256, 18] → [batch, 256, d_model]
        x = self.channel_projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding + transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # CLS token output
        cls_output = x[:, 0, :]
        logits     = self.classifier(cls_output)
        return logits


class EMGTransformer(nn.Module):
    """
    Transformer encoder for EMG anomaly detection.
    Input:  [batch, 12, 256]
    Output: [batch, 2]
    """

    def __init__(self,
                 n_channels: int = 12,
                 window_size: int = 256,
                 d_model: int = 32,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dim_feedforward: int = 64,
                 n_classes: int = 2,
                 dropout: float = 0.1):
        super(EMGTransformer, self).__init__()

        self.d_model = d_model

        # Channel projection
        self.channel_projection = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=window_size + 1,
            dropout=dropout
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, window_size]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size = x.shape[0]

        # [batch, 12, 256] → [batch, 256, 12]
        x = x.permute(0, 2, 1)

        # Project channels
        x = self.channel_projection(x)

        # CLS token + positional encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # CLS output
        cls_output = x[:, 0, :]
        logits     = self.classifier(cls_output)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 8

    # Test ECG Transformer
    print("Testing ECG Transformer...")
    model  = ECGTransformer(input_size=256, d_model=64,
                             n_heads=4, n_layers=2, n_classes=2)
    x      = torch.randn(batch_size, 256)
    out    = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EEG Transformer
    print("\nTesting EEG Transformer...")
    model  = EEGTransformer(n_channels=18, window_size=256,
                             d_model=64, n_heads=4, n_layers=3, n_classes=2)
    x      = torch.randn(batch_size, 18, 256)
    out    = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    # Test EMG Transformer
    print("\nTesting EMG Transformer...")
    model  = EMGTransformer(n_channels=12, window_size=256,
                             d_model=32, n_heads=4, n_layers=2, n_classes=2)
    x      = torch.randn(batch_size, 12, 256)
    out    = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    print("\nAll Transformer baseline tests passed!")