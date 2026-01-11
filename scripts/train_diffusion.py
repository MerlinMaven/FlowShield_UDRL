#!/usr/bin/env python3
"""
Train Diffusion Shield - Entraîne le shield Diffusion séparément.

Le modèle est sauvegardé dans: results/<env>/models/diffusion_shield.pt
Les courbes dans: results/<env>/figures/diffusion_training.png

Améliorations:
- Early Stopping avec patience configurable
- Cosine Annealing LR Scheduler avec warmup
- Gradient Clipping
- Seeds pour reproductibilité
- Calibration automatique du seuil OOD

Usage:
    python scripts/train_diffusion.py --data data/lunarlander_xxx.npz --epochs 100
"""

import argparse
import numpy as np
import torch
from torch.optim import AdamW
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import (
    DiffusionShield, load_data, ensure_results_dirs, save_model,
    set_seed, EarlyStopping, CosineScheduler, GradientClipper
)


def train_diffusion_shield(
    shield: DiffusionShield,
    states: np.ndarray,
    commands: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    patience: int = 15,
    use_scheduler: bool = True,
    grad_clip: float = 1.0,
):
    """Train the Diffusion Shield with early stopping and calibration."""
    shield = shield.to(device)
    optimizer = AdamW(shield.parameters(), lr=lr, weight_decay=1e-5)
    
    # Scheduler and early stopping
    scheduler = CosineScheduler(optimizer, warmup_epochs=5, total_epochs=epochs) if use_scheduler else None
    early_stopping = EarlyStopping(patience=patience, mode='min')
    grad_clipper = GradientClipper(max_norm=grad_clip)
    
    n = len(states)
    n_val = int(n * 0.1)
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    commands_t = torch.tensor(commands, dtype=torch.float32, device=device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        shield.train()
        epoch_loss = 0
        n_batches = 0
        
        perm = np.random.permutation(len(train_idx))
        
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = shield.loss(states_t[batch_idx], commands_t[batch_idx])
            loss.backward()
            
            grad_clipper(shield)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)
        
        # Validation
        shield.eval()
        with torch.no_grad():
            val_loss = 0
            n_val_batches = 0
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i+batch_size]
                loss = shield.loss(states_t[batch_idx], commands_t[batch_idx])
                val_loss += loss.item()
                n_val_batches += 1
            val_loss /= n_val_batches
        val_losses.append(val_loss)
        
        # Scheduler step
        current_lr = None
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_lr()
        
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            lr_str = f" | LR: {current_lr:.2e}" if current_lr else ""
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{lr_str}")
        
        # Early stopping
        if early_stopping(val_loss, shield):
            if verbose:
                print(f"Early stopping at epoch {epoch} (best: {epoch - early_stopping.counter})")
            break
    
    # Restore best model
    early_stopping.restore_best(shield)
    
    # Calibrate OOD threshold
    if verbose:
        print("Calibrating OOD threshold...")
    shield.eval()
    shield.calibrate(states_t[:500], commands_t[:500])
    if verbose:
        print(f"  Threshold set to: {shield.ood_threshold.item():.3f}")
    
    return train_losses, val_losses


def plot_training_curve(train_losses, val_losses, save_path):
    """Plot and save training curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Val Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('DDPM Loss')
    plt.title('Diffusion Shield Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curve to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Shield")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data file (.npz)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--n-timesteps", type=int, default=100,
                        help="Number of diffusion timesteps")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable learning rate scheduler")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("TRAINING DIFFUSION SHIELD")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Timesteps: {args.n_timesteps}")
    print(f"Patience: {args.patience}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    data = load_data(args.data)
    state_dim = data['state_dim']
    env_name = str(data.get('env_name', 'unknown'))
    
    print(f"State dim: {state_dim}")
    print(f"Transitions: {len(data['states'])}")
    
    # Create shield
    shield = DiffusionShield(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        n_timesteps=args.n_timesteps,
    )
    
    # Train
    train_losses, val_losses = train_diffusion_shield(
        shield,
        data['states'],
        data['commands'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        patience=args.patience,
        use_scheduler=not args.no_scheduler,
    )
    
    # Save
    results_dir = ensure_results_dirs(env_name)
    save_model(shield, env_name, "diffusion_shield")
    
    # Plot
    plot_path = results_dir / "figures" / "diffusion_training.png"
    plot_training_curve(train_losses, val_losses, plot_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: results/{env_name}/models/diffusion_shield.pt")
    print(f"Figure saved: results/{env_name}/figures/diffusion_training.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
