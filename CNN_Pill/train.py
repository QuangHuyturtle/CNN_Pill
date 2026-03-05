"""
Training script for Pill Image Classification with EfficientNetV2
Uses random 80/20 split on all data (recommended for ePillID dataset)
Supports ensemble training for improved performance
"""

import os
# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import yaml
import argparse
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet_pill import create_model, get_model_info
from utils.dataset import DataManager


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.losses = []
        self.top5_correct = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with a batch.

        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels
            loss: Batch loss value
        """
        self.losses.append(loss)

        # Top-1 accuracy
        _, predicted = outputs.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()

        # Top-5 accuracy
        _, top5_predicted = outputs.topk(5, dim=1)
        top5_correct = top5_predicted.eq(targets.view(-1, 1).expand_as(top5_predicted))
        self.top5_correct += top5_correct.any(dim=1).sum().item()

    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics."""
        loss = np.mean(self.losses) if self.losses else 0.0
        acc1 = 100.0 * self.correct / self.total if self.total > 0 else 0.0
        acc5 = 100.0 * self.top5_correct / self.total if self.total > 0 else 0.0

        return {
            'loss': loss,
            'acc1': acc1,
            'acc5': acc5
        }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'max'  # 'max' for accuracy, 'min' for loss
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


class PillClassifierTrainer:
    """
    Trainer class for pill image classification.

    Args:
        model: The neural network model
        device: Device to train on (cuda/cpu)
        num_classes: Number of output classes
        learning_rate: Initial learning rate
        use_class_weights: Use weighted loss for imbalanced classes
        max_grad_norm: Maximum gradient norm for clipping
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 960,
        learning_rate: float = 1e-3,
        use_class_weights: bool = True,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 0.5
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.use_class_weights = use_class_weights
        self.max_grad_norm = max_grad_norm

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

        # Metrics
        self.train_metrics = MetricsTracker(num_classes)
        self.val_metrics = MetricsTracker(num_classes)

        # Training state
        self.current_epoch = 0
        self.best_acc1 = 0.0
        self.best_acc5 = 0.0

    def setup_optimizer(
        self,
        backbone_lr_multiplier: float = 0.1,
        optimizer_type: str = 'adamw',
        weight_decay: float = 1e-4
    ):
        """Setup optimizer with differential learning rates."""
        param_groups = self.model.get_param_groups(
            lr=self.learning_rate,
            backbone_lr_multiplier=backbone_lr_multiplier
        )

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def setup_scheduler(
        self,
        scheduler_type: str = 'cosine',
        num_epochs: int = 50,
        warmup_epochs: int = 5
    ):
        """Setup learning rate scheduler."""
        if scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            self.scheduler = None

    def train_epoch(self, train_loader, class_weights: Optional[torch.Tensor] = None):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        # Update loss with class weights if provided
        if class_weights is not None and self.use_class_weights:
            self.criterion.weight = class_weights.to(self.device)
        else:
            self.criterion.weight = None

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability (configurable)
            max_grad_norm = getattr(self, 'max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(outputs, labels, loss.item())

            # Update progress bar
            metrics = self.train_metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc1': f"{metrics['acc1']:.2f}%"
            })

        return self.train_metrics.get_metrics()

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()

        # Check if val_loader has batches
        num_batches = len(val_loader)
        if num_batches == 0:
            print("Warning: Validation loader is empty!")
            return {'loss': 0.0, 'acc1': 0.0, 'acc5': 0.0}

        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch} [Val]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Debug info for first batch
            if batch_idx == 0:
                print(f"Val batch 0: outputs shape={outputs.shape}, labels shape={labels.shape}")
                print(f"Val batch 0: labels range=[{labels.min().item()}, {labels.max().item()}]")

            loss = self.criterion(outputs, labels)

            # Update metrics
            self.val_metrics.update(outputs, labels, loss.item())

            # Update progress bar
            metrics = self.val_metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc1': f"{metrics['acc1']:.2f}%"
            })

        return self.val_metrics.get_metrics()

    def train_fold(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        fold_idx: int,
        save_dir: str,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        writer: Optional[SummaryWriter] = None,
        unfreeze_epoch: int = 10,
        start_epoch: int = 0,
        early_stopping_patience: int = 7,
        early_stopping_delta: float = 0.001
    ):
        """
        Train a single fold with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            fold_idx: Current fold index
            save_dir: Directory to save checkpoints
            num_classes: Number of classes for this fold
            class_weights: Optional class weights for loss function
            writer: TensorBoard writer
            unfreeze_epoch: Epoch to unfreeze backbone
            early_stopping_patience: Patience for early stopping
            early_stopping_delta: Minimum improvement for early stopping
        """
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*50}")

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_delta,
            mode='max'
        )

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Unfreeze backbone at specified epoch
            if epoch == unfreeze_epoch:
                print(f"\nUnfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()

            # Train
            train_metrics = self.train_epoch(train_loader, class_weights)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['acc1'])
            elif self.scheduler is not None:
                self.scheduler.step()

            # Log to tensorboard
            if writer is not None:
                for metric_name, value in train_metrics.items():
                    writer.add_scalar(f'Fold{fold_idx}/train_{metric_name}', value, epoch)
                for metric_name, value in val_metrics.items():
                    writer.add_scalar(f'Fold{fold_idx}/val_{metric_name}', value, epoch)
                writer.add_scalar(f'Fold{fold_idx}/lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc1: {train_metrics['acc1']:.2f}%, Acc5: {train_metrics['acc5']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc1: {val_metrics['acc1']:.2f}%, Acc5: {val_metrics['acc5']:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            is_best = val_metrics['acc1'] > self.best_acc1
            if is_best:
                self.best_acc1 = val_metrics['acc1']
                self.best_acc5 = val_metrics['acc5']

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc1': self.best_acc1,
                    'best_acc5': self.best_acc5,
                    'fold_idx': fold_idx,
                    'num_classes': num_classes
                }

                save_path = os.path.join(save_dir, f'best_fold{fold_idx}.pth')
                torch.save(checkpoint, save_path)
                print(f"  Saved best model: Acc1={self.best_acc1:.2f}%, Acc5={self.best_acc5:.2f}%")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(save_dir, f'checkpoint_fold{fold_idx}_epoch{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path)

            # Early stopping check
            if early_stopping(val_metrics['acc1']):
                print(f"\n{'='*50}")
                print(f"EARLY STOPPING at epoch {epoch}")
                print(f"Best validation accuracy: {early_stopping.best_score:.2f}%")
                print(f"No improvement for {early_stopping.counter} epochs")
                print(f"{'='*50}")
                break

        print(f"\nFold {fold_idx} completed!")
        print(f"Best Val - Acc1: {self.best_acc1:.2f}%, Acc5: {self.best_acc5:.2f}%")

        return self.best_acc1, self.best_acc5


def train_random_split(
    config: Dict,
    data_manager: DataManager,
    save_dir: str,
    device: torch.device,
    resume_path: Optional[str] = None,
    ensemble_seed: Optional[int] = None
):
    """
    Train using random split on all data (recommended for ePillID dataset).

    Args:
        config: Training configuration dict
        data_manager: DataManager instance
        save_dir: Directory to save results
        device: Training device
        resume_path: Path to checkpoint to resume from (optional)
        ensemble_seed: Random seed for ensemble training (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'='*60}")
    print("TRAINING ON ALL DATA (Random 80/20 Split)")
    print(f"{'='*60}")
    if ensemble_seed is not None:
        print(f"Ensemble mode - Seed: {ensemble_seed}")

    # Get data loaders for all data with random split
    train_loader, val_loader, num_classes = data_manager.get_all_data_loaders(
        augmentation=config['data']['augmentation'],
        train_ratio=0.8,
        random_seed=42 if ensemble_seed is None else ensemble_seed
    )

    # Get class weights for all data
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = data_manager.get_all_class_weights(num_classes)

    # Create model
    model = create_model(
        num_classes=num_classes,
        model_size=config['model']['size'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout'],
        dropout_head_extra=config['model'].get('dropout_head_extra', 0.15),
        freeze_backbone=config['model']['freeze_backbone']
    )

    # Print model info
    info = get_model_info(model)
    print(f"\nModel: {info['model_name']}")
    print(f"Parameters: {info['trainable_params']:,} trainable / {info['total_params']:,} total")
    print(f"Model size: {info['model_size_mb']:.2f} MB")

    # Create trainer
    weight_decay = config['model'].get('weight_decay', 0.01)
    trainer = PillClassifierTrainer(
        model=model,
        device=device,
        num_classes=num_classes,
        learning_rate=config['training']['learning_rate'],
        use_class_weights=config['training']['use_class_weights'],
        label_smoothing=config['training']['label_smoothing'],
        max_grad_norm=config['training'].get('max_grad_norm', 0.5)
    )

    trainer.setup_optimizer(
        backbone_lr_multiplier=config['training']['backbone_lr_multiplier'],
        optimizer_type=config['training']['optimizer'],
        weight_decay=weight_decay
    )

    trainer.setup_scheduler(
        scheduler_type=config['training']['scheduler'],
        num_epochs=config['training']['num_epochs'],
        warmup_epochs=config['training']['warmup_epochs']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path:
        print(f"\nLoading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
        if 'best_acc1' in checkpoint:
            trainer.best_acc1 = checkpoint['best_acc1']
            trainer.best_acc5 = checkpoint['best_acc5']
            print(f"Previous best - Acc1: {trainer.best_acc1:.2f}%, Acc5: {trainer.best_acc5:.2f}%")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs', 'train'))

    # Train model
    best_acc1, best_acc5 = trainer.train_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        fold_idx=0,  # Single model, not a fold
        save_dir=save_dir,
        num_classes=num_classes,
        class_weights=class_weights,
        writer=writer,
        unfreeze_epoch=config['training']['unfreeze_epoch'],
        start_epoch=start_epoch,
        early_stopping_patience=config['training'].get('early_stopping_patience', 7),
        early_stopping_delta=config['training'].get('early_stopping_delta', 0.001)
    )

    writer.close()

    # Print final results
    print(f"\n{'='*60}")
    print("TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Best Top-1 Accuracy: {best_acc1:.2f}%")
    print(f"Best Top-5 Accuracy: {best_acc5:.2f}%")

    # Save results
    results_path = os.path.join(save_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump({
            'best_acc1': float(best_acc1),
            'best_acc5': float(best_acc5),
            'num_classes': num_classes
        }, f)

    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Pill Image Classifier with Ensemble Support')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--ensemble', action='store_true',
                        help='Enable ensemble training (train multiple models)')
    parser.add_argument('--num_models', type=int, default=3,
                        help='Number of models for ensemble training (default: 3)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999],
                        help='Random seeds for each model (default: 42 123 999)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single model training (default: 42)')
    args = parser.parse_args()

    # Load or create config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {
                'size': 's',  # s, m, or l
                'pretrained': True,
                'dropout': 0.3,
                'freeze_backbone': True
            },
            'data': {
                'batch_size': 32,
                'num_workers': 0,  # Set = 0 for Windows
                'augmentation': True
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 1e-3,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'backbone_lr_multiplier': 0.1,
                'warmup_epochs': 5,
                'unfreeze_epoch': 10,
                'use_class_weights': True,
                'label_smoothing': 0.1,
                'gradient_clip': 1.0
            }
        }

        # Save default config
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
        print(f"Created default config at {args.config}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data manager
    data_manager = DataManager(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Ensemble training: train multiple models with different seeds
    if args.ensemble:
        num_models = args.num_models
        seeds = args.seeds[:num_models]  # Take first N seeds
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING: {num_models} models with seeds {seeds}")
        print(f"{'='*60}")

        ensemble_results = []

        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{num_models} (seed={seed})")
            print(f"{'='*60}")

            # Create save directory for this model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_dir = os.path.join(args.save_dir, f'run_{timestamp}_model{i+1}')
            os.makedirs(model_save_dir, exist_ok=True)

            # Train this model
            train_random_split(config, data_manager, model_save_dir, device,
                                resume_path=None, ensemble_seed=seed)

            # Load results to track best model
            results_path = os.path.join(model_save_dir, 'results.yaml')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = yaml.safe_load(f)
                    ensemble_results.append({
                        'model_idx': i+1,
                        'seed': seed,
                        'best_acc1': results.get('best_acc1', 0),
                        'best_acc5': results.get('best_acc5', 0),
                        'save_dir': model_save_dir
                    })

        # Print ensemble summary
        print(f"\n{'='*60}")
        print("ENSEMBLE TRAINING SUMMARY")
        print(f"{'='*60}")
        for result in ensemble_results:
            print(f"Model {result['model_idx']} (seed={result['seed']}): "
                  f"Val Acc1={result['best_acc1']:.2f}%, Acc5={result['best_acc5']:.2f}%")

        # Find best model
        best_model = max(ensemble_results, key=lambda x: x['best_acc1'])
        print(f"\nBest model: Model {best_model['model_idx']} (seed={best_model['seed']})")
        print(f"Best Val Acc1: {best_model['best_acc1']:.2f}%")
        print(f"Use checkpoint: {best_model['save_dir']}/best_fold0.pth")
    else:
        # Single model training (original logic)
        # Create save directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, f'run_{timestamp}')

        # Resume from checkpoint if specified
        if args.resume:
            if not os.path.exists(args.resume):
                print(f"Error: Checkpoint not found: {args.resume}")
                return
            # Extract directory from resume path for save_dir
            save_dir = os.path.dirname(args.resume)
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"Saving to: {save_dir}")

        # Train with random 80/20 split
        print(f"Training with random 80/20 split on all data (seed={args.seed})")
        train_random_split(config, data_manager, save_dir, device, resume_path=args.resume, ensemble_seed=args.seed)

if __name__ == '__main__':
    main()
