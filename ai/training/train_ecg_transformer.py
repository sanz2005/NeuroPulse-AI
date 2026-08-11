import numpy as np
import torch

from ai.models.baselines.transformer_model import ECGTransformer
from ai.training.trainer import Trainer

CONFIG = {
    'data_dir': 'data/processed/ecg',
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 10,
    'device': 'cuda:0',
}

def load_data():
    X_train = np.load(f"{CONFIG['data_dir']}/train/X.npy")
    y_train = np.load(f"{CONFIG['data_dir']}/train/y.npy")

    X_val = np.load(f"{CONFIG['data_dir']}/val/X.npy")
    y_val = np.load(f"{CONFIG['data_dir']}/val/y.npy")

    X_test = np.load(f"{CONFIG['data_dir']}/test/X.npy")
    y_test = np.load(f"{CONFIG['data_dir']}/test/y.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_transformer():
    print("=" * 60)
    print("Training ECG Transformer Only")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    model = ECGTransformer(
        input_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        dropout=0.1
    )

    trainer = Trainer(
        model=model,
        model_name='ecg_transformer',
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        patience=CONFIG['patience']
    )

    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    metrics = trainer.evaluate(X_test, y_test)

    print("\nTransformer Results:")
    print(metrics)

    return metrics


if __name__ == "__main__":
    train_transformer()