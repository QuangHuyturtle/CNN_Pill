"""
Plot Training Metrics from TensorBoard Logs
Visualizes train, validation, and test accuracy over epochs
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse TensorBoard event files and extract metrics.

    Args:
        log_dir: Directory containing TensorBoard logs

    Returns:
        Dictionary mapping metric names to (step, value) tuples
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get available tags (metric names)
    tags = event_acc.Tags()['scalars']

    metrics = {}
    for tag in tags:
        # Extract scalar events
        scalar_events = event_acc.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in scalar_events]

    return metrics


def plot_train_loss(
    train_loss: List[Tuple[int, float]],
    save_path: str = None,
    show_plot: bool = True
):
    """
    Plot train loss curve over epochs.

    Args:
        train_loss: List of (epoch, train_loss) tuples
        save_path: Optional path to save the figure
        show_plot: Whether to display plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 12

    # Extract epochs and values
    train_epochs, train_loss_values = zip(*train_loss) if train_loss else ([], [])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot train loss
    ax.plot(train_epochs, train_loss_values,
            marker='o', markersize=5, linewidth=2.5,
            label='Train Loss', color='#6366f1', alpha=0.9)

    # Labels and title
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=13)
    ax.set_title('Training Process: Loss over Epochs',
                fontweight='bold', fontsize=15, pad=15)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
             fontsize=11, edgecolor='black', facecolor='white')

    # Grid and limits
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_ylim(bottom=0)

    # Add final loss annotation
    if len(train_loss_values) > 0:
        final_train_loss = train_loss_values[-1]
        min_loss = min(train_loss_values)
        min_idx = np.argmin(train_loss_values)
        min_epoch = train_epochs[min_idx]
        loss_text = f'Final: {final_train_loss:.4f}\nBest: {min_loss:.4f} (Epoch {int(min_epoch)})'

        ax.text(0.02, 0.98, loss_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=10, fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Train Loss plot saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def plot_accuracy_comparison(
    train_acc: List[Tuple[int, float]],
    val_acc: List[Tuple[int, float]],
    test_acc: List[Tuple[int, float]] = None,
    save_path: str = None,
    show_plot: bool = True
):
    """
    Plot train, validation, and test accuracy curves for comparison.

    Args:
        train_acc: List of (epoch, train_accuracy) tuples
        val_acc: List of (epoch, val_accuracy) tuples
        test_acc: Optional list of (epoch, test_accuracy) tuples
        save_path: Optional path to save the figure
        show_plot: Whether to display plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 12

    # Extract epochs and values
    train_epochs, train_acc_values = zip(*train_acc) if train_acc else ([], [])
    val_epochs, val_acc_values = zip(*val_acc) if val_acc else ([], [])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot train accuracy
    ax.plot(train_epochs, train_acc_values,
            marker='o', markersize=5, linewidth=2.5,
            label='Train Accuracy', color='#10b981', alpha=0.9)

    # Plot validation accuracy
    ax.plot(val_epochs, val_acc_values,
            marker='s', markersize=5, linewidth=2.5,
            label='Validation Accuracy', color='#6366f1', alpha=0.9)

    # Plot test accuracy if provided
    if test_acc:
        test_epochs, test_acc_values = zip(*test_acc) if test_acc else ([], [])
        if len(test_epochs) > 0:
            ax.plot(test_epochs, test_acc_values,
                    marker='^', markersize=5, linewidth=2.5,
                    label='Test Accuracy', color='#ef4444', alpha=0.9)

    # Labels and title
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Training Comparison: Train vs Validation vs Test Accuracy',
                fontweight='bold', fontsize=15, pad=15)

    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             fontsize=11, edgecolor='black', facecolor='white')

    # Grid and limits
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_ylim(bottom=0, top=100)

    # Add final values annotation
    final_train = train_acc_values[-1] if len(train_acc_values) > 0 else 0
    final_val = val_acc_values[-1] if len(val_acc_values) > 0 else 0
    info_text = f'Train: {final_train:.2f}%\nVal: {final_val:.2f}%'
    if test_acc and len(test_acc_values) > 0:
        final_test = test_acc_values[-1]
        info_text += f'\nTest: {final_test:.2f}%'

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontfamily='monospace')

    # Best validation annotation
    if len(val_acc_values) > 0:
        best_idx = np.argmax(val_acc_values)
        best_epoch = val_epochs[best_idx]
        best_val = val_acc_values[best_idx]
        ax.annotate(f'Best: {best_val:.2f}%\n(Epoch {int(best_epoch)})',
                    xy=(best_epoch, best_val),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#fbbf24', alpha=0.4),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                  color='#fbbf24', lw=2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Plot training metrics from TensorBoard logs')
    parser.add_argument('--log_dir', type=str,
                        default='checkpoints/run_20260125_065908/logs/train',
                        help='Path to TensorBoard log directory')
    parser.add_argument('--save', type=str, default=None,
                        help='Base path to save figures (e.g., training_curves)')
    parser.add_argument('--no_show', action='store_true',
                        help='Don\'t display plot, just save')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number to plot (for multi-fold training)')

    args = parser.parse_args()

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse TensorBoard logs
    print(f"Parsing TensorBoard logs from: {args.log_dir}")
    metrics = parse_tensorboard_logs(args.log_dir)

    # Extract metrics for specific fold
    fold_prefix = f'Fold{args.fold}/'

    # Get train/val/test metrics
    train_acc_key = f'{fold_prefix}train_acc1'
    val_acc_key = f'{fold_prefix}val_acc1'
    test_acc_key = f'{fold_prefix}test_acc1'

    train_loss_key = f'{fold_prefix}train_loss'

    train_acc = metrics.get(train_acc_key, [])
    val_acc = metrics.get(val_acc_key, [])
    test_acc = metrics.get(test_acc_key, [])

    train_loss = metrics.get(train_loss_key, [])

    # Check if we have data
    if not train_acc and not train_loss:
        print("No training metrics found!")
        print(f"Available metrics: {list(metrics.keys())}")
        return

    # Print summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"Train Accuracy: {len(train_acc)} data points")
    print(f"Validation Accuracy: {len(val_acc)} data points")
    print(f"Test Accuracy: {len(test_acc)} data points")
    print(f"Train Loss: {len(train_loss)} data points")
    if train_acc:
        print(f"\nFinal Train Acc: {train_acc[-1][1]:.2f}%")
    if val_acc:
        print(f"Final Val Acc: {val_acc[-1][1]:.2f}%")
        print(f"Best Val Acc: {max(v for _, v in val_acc):.2f}%")
    if train_loss:
        print(f"Final Train Loss: {train_loss[-1][1]:.4f}")
        print(f"Best Train Loss: {min(v for _, v in train_loss):.4f}")

    # Determine save paths with timestamp to avoid overwriting
    loss_save_path = None
    acc_save_path = None
    if args.save:
        base_path = args.save.rsplit('.', 1)[0]  # Remove extension
        loss_save_path = f"{base_path}_{timestamp}_loss.png"
        acc_save_path = f"{base_path}_{timestamp}_accuracy.png"
    else:
        # Default names without timestamp if --save not provided
        loss_save_path = f"train_loss_{timestamp}.png"
        acc_save_path = f"accuracy_{timestamp}.png"

    # Plot train loss (Biểu đồ 1: Train Loss)
    print("\n" + "="*60)
    print("BIỂU ĐỒ 1: TRAIN LOSS OVER EPOCHS")
    print("="*60)
    if train_loss:
        plot_train_loss(
            train_loss=train_loss,
            save_path=loss_save_path,
            show_plot=not args.no_show
        )
    else:
        print("No train loss data available!")

    # Plot accuracy comparison (Biểu đồ 2: Accuracy comparison)
    print("\n" + "="*60)
    print("BIỂU ĐỒ 2: ACCURACY COMPARISON (Train vs Val vs Test)")
    print("="*60)
    if train_acc:
        plot_accuracy_comparison(
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc if test_acc else None,
            save_path=acc_save_path,
            show_plot=not args.no_show
        )
    else:
        print("No accuracy data available!")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
