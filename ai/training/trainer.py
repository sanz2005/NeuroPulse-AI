"""
Base Trainer Class for NeuroPulse AI
Handles training loop, validation, checkpointing,
early stopping, and metric tracking for all models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from tqdm import tqdm
import json


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training if validation loss doesn't improve
    for 'patience' consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0
        return self.should_stop


class Trainer:
    """
    Base trainer for all NeuroPulse AI models.
    Works with both SNN and baseline models.

    Args:
        model:        PyTorch model to train
        model_name:   Name for saving checkpoints
        device:       'cpu' or 'cuda'
        learning_rate: Optimizer learning rate
        weight_decay:  L2 regularization
        patience:      Early stopping patience
    """

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 10):

        self.model      = model.to(device)
        self.model_name = model_name
        self.device     = device

        # Optimizer — Adam works well for both SNN and ANN
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Loss function — weighted for class imbalance
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience)

        # Tracking
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []
        self.best_val_f1  = 0.0
        self.best_epoch   = 0

        # Save directory
        self.save_dir = 'ai/saved_models'
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_class_weights(self, labels: np.ndarray, weight_power: float = 1.0) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        Higher weight for minority class (anomaly).

        Args:
            weight_power: dampening exponent applied to the raw
                inverse-frequency weights (weight ** weight_power).
                1.0 = full inverse-frequency weighting (current default,
                unchanged behavior). Try 0.5 (sqrt-dampened) ONLY if F1
                is still exactly 0.0 after ~10 epochs — an extremely
                large weight ratio (this dataset's is ~154x) can
                occasionally make training unstable enough that the
                model never escapes majority-only prediction; dampening
                reduces that ratio to ~12x while still strongly favoring
                the minority class.
        """
        classes, counts = np.unique(labels, return_counts=True)
        total           = len(labels)
        weights         = (total / (len(classes) * counts)) ** weight_power
        weights         = torch.FloatTensor(weights).to(self.device)
        return weights

    def prepare_dataloader(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            batch_size: int = 64,
                            shuffle: bool = True,
                            weighted_sampling: bool = False) -> DataLoader:
        """
        Create DataLoader from numpy arrays.

        Args:
            weighted_sampling: if True, uses a WeightedRandomSampler so
                minority-class samples are drawn (with replacement) much
                more often per epoch than their raw frequency — this is
                the memory-safe equivalent of oversampling (no duplicate
                arrays are ever materialized), and directly increases how
                often the optimizer actually sees a minority-class
                gradient signal, which class-weighted loss alone was not
                sufficient to achieve for this dataset's ~300x raw count
                imbalance. When True, `shuffle` is ignored (a sampler and
                shuffle=True cannot both be passed to DataLoader).
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset  = TensorDataset(X_tensor, y_tensor)
        use_pin_memory = (self.device.type == 'cuda') if hasattr(self.device, 'type') else str(self.device) == 'cuda'

        if weighted_sampling:
            classes, counts = np.unique(y, return_counts=True)
            class_weight_map = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
            sample_weights = np.array([class_weight_map[label] for label in y], dtype=np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=0,
                              pin_memory=use_pin_memory)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=0,
                          pin_memory=use_pin_memory)

    def train_epoch(self, dataloader: DataLoader) -> tuple:
        """
        Single training epoch.

        Returns:
            avg_loss: float
            accuracy: float
        """
        self.model.train()
        total_loss = 0.0
        all_preds  = []
        all_labels = []

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass — handle both SNN (returns tuple) and ANN
            output = self.model(batch_X)
            if isinstance(output, tuple):
                logits = output[0]  # SNN returns (logits, spike_record)
            else:
                logits = output

            loss = self.criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping — important for SNN stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def validate_epoch(self, dataloader: DataLoader) -> tuple:
        """
        Single validation epoch.

        Returns:
            avg_loss: float
            accuracy: float
            metrics:  dict with precision, recall, f1
        """
        self.model.eval()
        total_loss = 0.0
        all_preds  = []
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_X)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss  = total_loss / len(dataloader)
        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds,
                                     zero_division=0)
        recall    = recall_score(all_labels, all_preds,
                                  zero_division=0)
        f1        = f1_score(all_labels, all_preds,
                              zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        metrics = {
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'auc':       auc
        }

        return avg_loss, accuracy, metrics

    def save_checkpoint(self, epoch: int, val_acc: float, val_f1: float):
        """Save model checkpoint."""
        path = os.path.join(
            self.save_dir,
            f'{self.model_name}_best.pth'
        )
        torch.save({
            'epoch':      epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc':    val_acc,
            'val_f1':     val_f1,
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
        }, path)

    def load_checkpoint(self) -> int:
        """Load best checkpoint. Returns epoch number."""
        path = os.path.join(
            self.save_dir,
            f'{self.model_name}_best.pth'
        )
        if not os.path.exists(path):
            return 0
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"(val_acc: {checkpoint['val_acc']:.4f})")
        return checkpoint['epoch']

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 64) -> dict:
        """
        Full training loop.

        Args:
            X_train:    Training features
            y_train:    Training labels
            X_val:      Validation features
            y_val:      Validation labels
            epochs:     Maximum training epochs
            batch_size: Batch size

        Returns:
            history: Training history dict
        """
        # Compute class weights for imbalanced data
        # weight_power=0.0 (neutral/no extra loss weighting) because
        # weighted_sampling=True (see train_loader below) is now the
        # primary imbalance-correction mechanism — it already exposes the
        # model to a roughly balanced class ratio every batch by
        # resampling minority examples with replacement. Combining that
        # WITH a strong loss weight (154x or even the dampened 12x) would
        # double-correct and likely push the model to over-predict the
        # minority class, hurting precision. If recall is still too low
        # after trying weighted_sampling alone, raise this back toward
        # 0.3-0.5 gradually rather than jumping to 1.0.
        weights = self.compute_class_weights(y_train, weight_power=0.0)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Create dataloaders
        train_loader = self.prepare_dataloader(
            X_train, y_train, batch_size, shuffle=True, weighted_sampling=True
        )
        val_loader = self.prepare_dataloader(
            X_val, y_val, batch_size, shuffle=False
        )

        print(f"\nTraining {self.model_name}")
        print(f"  Device:     {self.device}")
        print(f"  Train size: {len(X_train)}")
        print(f"  Val size:   {len(X_val)}")
        print(f"  Epochs:     {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Class weights: {weights.cpu().numpy()}")
        print("-" * 50)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, metrics = self.validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Save best model — selected by F1, not accuracy. With this
            # dataset's ~300x class imbalance, a trivial "always predict
            # majority" model already scores ~99.6% validation accuracy,
            # so selecting the "best" checkpoint by accuracy could silently
            # keep an epoch that never learned to detect the minority
            # class over one that genuinely did.
            if metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['f1']
                self.best_epoch  = epoch
                self.save_checkpoint(epoch, val_acc, metrics['f1'])

            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Precision: {metrics['precision']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Best val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"  Total time:  {total_time:.1f}s")

        history = {
            'model_name':   self.model_name,
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
            'train_accs':   self.train_accs,
            'val_accs':     self.val_accs,
            'best_val_f1':  self.best_val_f1,
            'best_epoch':   self.best_epoch,
            'total_time':   total_time
        }

        # Save history
        history_path = os.path.join(
            self.save_dir,
            f'{self.model_name}_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 batch_size: int = 64) -> dict:
        """
        Evaluate model on test set.

        Returns:
            metrics: Complete evaluation metrics dict
        """
        # Load best checkpoint
        self.load_checkpoint()

        test_loader = self.prepare_dataloader(
            X_test, y_test, batch_size, shuffle=False
        )

        _, _, metrics = self.validate_epoch(test_loader)

        print(f"\nTest Results for {self.model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        return metrics