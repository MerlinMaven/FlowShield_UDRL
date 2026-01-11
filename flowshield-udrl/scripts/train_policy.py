#!/usr/bin/env python3
"""
Train UDRL Policy - Entraîne la politique UDRL séparément.

Le modèle est sauvegardé dans: results/<env>/models/policy.pt
Les courbes dans: results/<env>/figures/policy_training.png
Les logs TensorBoard dans: logs/tensorboard/<env>/policy_<timestamp>

Améliorations:
- Early Stopping avec patience configurable
- Cosine Annealing LR Scheduler avec warmup
- Gradient Clipping
- Seeds pour reproductibilité

Usage:
    python scripts/train_policy.py --data data/lunarlander_xxx.npz --epochs 100
    python scripts/train_policy.py --data data/lunarlander_xxx.npz --epochs 50 --lr 0.001
    python scripts/train_policy.py --data data/lunarlander_xxx.npz --epochs 100 --tensorboard
    python scripts/train_policy.py --data data/lunarlander_xxx.npz --epochs 200 --patience 20
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import shared models
from models import (
    UDRLPolicy, load_data, create_dataloaders, ensure_results_dirs, save_model,
    set_seed, EarlyStopping, CosineScheduler, GradientClipper
)
from logger import get_logger


def train_policy(
    policy: UDRLPolicy,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    logger=None,
    patience: int = 15,
    use_scheduler: bool = True,
    grad_clip: float = 1.0,
):
    """Train the UDRL policy with early stopping and LR scheduling."""
    policy = policy.to(device)
    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    # Scheduler and early stopping
    scheduler = CosineScheduler(optimizer, warmup_epochs=5, total_epochs=epochs) if use_scheduler else None
    early_stopping = EarlyStopping(patience=patience, mode='min')
    grad_clipper = GradientClipper(max_norm=grad_clip)
    
    train_losses = []
    val_losses = []
    best_epoch = epochs
    
    for epoch in range(1, epochs + 1):
        # Training
        policy.train()
        epoch_loss = 0
        for states, actions, commands in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            commands = commands.to(device)
            
            optimizer.zero_grad()
            log_prob = policy.log_prob(states, commands, actions)
            loss = -log_prob.mean()
            loss.backward()
            
            # Gradient clipping
            grad_clipper(policy)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions, commands in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                commands = commands.to(device)
                
                log_prob = policy.log_prob(states, commands, actions)
                val_loss += -log_prob.mean().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Scheduler step
        current_lr = None
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_lr()
        
        # TensorBoard logging
        if logger:
            logger.log_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
            if current_lr is not None:
                logger.log_scalar('lr', current_lr, epoch)
        
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            lr_str = f" | LR: {current_lr:.2e}" if current_lr else ""
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{lr_str}")
        
        # Early stopping
        if early_stopping(val_loss, policy):
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch: {epoch - early_stopping.counter})")
            best_epoch = epoch - early_stopping.counter
            break
    
    # Restore best model
    early_stopping.restore_best(policy)
    
    if logger:
        logger.flush()
    
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('UDRL Policy Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train UDRL Policy")
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
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of layers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable learning rate scheduler")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("TRAINING UDRL POLICY")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    data = load_data(args.data)
    state_dim = data['state_dim']
    action_dim = data['action_dim']
    env_name = str(data.get('env_name', 'unknown'))
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Transitions: {len(data['states'])}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data['states'], data['actions'], data['commands'],
        batch_size=args.batch_size
    )
    
    # Create policy
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    
    # Setup logger
    logger = get_logger(env_name, "policy", use_tensorboard=args.tensorboard)
    
    # Train
    train_losses, val_losses = train_policy(
        policy, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        logger=logger,
        patience=args.patience,
        use_scheduler=not args.no_scheduler,
    )
    
    # Log hyperparameters
    if args.tensorboard:
        logger.log_hparams(
            {'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
             'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers,
             'patience': args.patience, 'seed': args.seed},
            {'final_train_loss': train_losses[-1], 'final_val_loss': val_losses[-1]}
        )
        logger.close()
    
    # Save
    results_dir = ensure_results_dirs(env_name)
    save_model(policy, env_name, "policy")
    
    # Plot
    plot_path = results_dir / "figures" / "policy_training.png"
    plot_training_curves(train_losses, val_losses, plot_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: results/{env_name}/models/policy.pt")
    print(f"Figure saved: results/{env_name}/figures/policy_training.png")
    if args.tensorboard:
        print(f"TensorBoard: logs/tensorboard/{env_name}/policy_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
