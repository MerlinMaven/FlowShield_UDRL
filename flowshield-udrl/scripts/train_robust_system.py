#!/usr/bin/env python3
"""
🎯 TRAINING SCRIPT FOR ROBUST UDRL + SHIELD SYSTEM

This script trains:
1. A command-sensitive UDRL policy (UDRLPolicyRobust)
2. A SafetyShield with OOD detection and command projection

The key improvements:
- Policy uses FiLM conditioning for strong command sensitivity
- Shield is trained with contrastive ID/OOD samples
- Shield can project OOD commands to safe region

Usage:
    python train_robust_system.py --data data/lunarlander_expert.npz --epochs 200
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from robust_shield_system import (
    UDRLPolicyRobust, 
    SafetyShield, 
    ShieldedAgent,
    create_ood_samples,
)


def load_data(data_path: str, val_split: float = 0.1):
    """Load and split data."""
    data = np.load(data_path)
    
    states = torch.FloatTensor(data['states'])
    actions = torch.FloatTensor(data['actions'])
    commands = torch.FloatTensor(data['commands'])
    
    # Shuffle
    n = len(states)
    perm = torch.randperm(n)
    states, actions, commands = states[perm], actions[perm], commands[perm]
    
    # Split
    val_size = int(n * val_split)
    train_data = {
        'states': states[val_size:],
        'actions': actions[val_size:],
        'commands': commands[val_size:],
    }
    val_data = {
        'states': states[:val_size],
        'actions': actions[:val_size],
        'commands': commands[:val_size],
    }
    
    print(f"Data loaded: {n} transitions")
    print(f"  Train: {len(train_data['states'])}")
    print(f"  Val: {len(val_data['states'])}")
    print(f"  Command range: H=[{commands[:,0].min():.1f}, {commands[:,0].max():.1f}], "
          f"R=[{commands[:,1].min():.1f}, {commands[:,1].max():.1f}]")
    
    return train_data, val_data


def train_policy(
    policy: UDRLPolicyRobust,
    train_data: dict,
    val_data: dict,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cpu',
):
    """Train the robust UDRL policy."""
    print("\n" + "="*60)
    print("TRAINING ROBUST UDRL POLICY")
    print("="*60)
    
    policy = policy.to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        train_data['states'], 
        train_data['actions'], 
        train_data['commands']
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        val_data['states'], 
        val_data['actions'], 
        val_data['commands']
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    
    # Calibrate command normalizer
    policy.command_norm.update(train_data['commands'])
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Training
        policy.train()
        train_losses = []
        
        for states, actions, commands in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            commands = commands.to(device)
            
            optimizer.zero_grad()
            
            # Compute log probability
            log_prob = policy.log_prob(states, commands, actions)
            
            # Negative log likelihood + regularization for command sensitivity
            nll_loss = -log_prob.mean()
            
            # Command sensitivity loss: different commands should give different outputs
            # Shuffle commands and compute MSE between outputs
            perm = torch.randperm(len(commands))
            shuffled_commands = commands[perm]
            
            mean1, _ = policy(states, commands)
            mean2, _ = policy(states, shuffled_commands)
            
            # Commands that are different should give different actions
            command_diff = (commands - shuffled_commands).abs().sum(dim=1, keepdim=True)
            action_diff = (mean1 - mean2).abs().sum(dim=1, keepdim=True)
            
            # Encourage action difference proportional to command difference
            sensitivity_loss = F.mse_loss(
                action_diff / (action_diff.max() + 1e-8),
                command_diff / (command_diff.max() + 1e-8)
            )
            
            loss = nll_loss + 0.1 * sensitivity_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(nll_loss.item())
        
        # Validation
        policy.eval()
        val_losses = []
        
        with torch.no_grad():
            for states, actions, commands in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                commands = commands.to(device)
                
                log_prob = policy.log_prob(states, commands, actions)
                val_losses.append(-log_prob.mean().item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best weights
    policy.load_state_dict(best_state)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return policy


def train_shield(
    shield: SafetyShield,
    train_data: dict,
    val_data: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    """Train the safety shield with OOD detection."""
    print("\n" + "="*60)
    print("TRAINING SAFETY SHIELD")
    print("="*60)
    
    shield = shield.to(device)
    
    # Calibrate on training data
    shield.calibrate(train_data['states'], train_data['commands'])
    
    # Create OOD samples
    n_ood = len(train_data['commands'])
    ood_commands = create_ood_samples(train_data['commands'], n_ood)
    
    # Use random states for OOD (since we don't have real OOD states)
    ood_states = train_data['states'][torch.randperm(len(train_data['states']))[:n_ood]]
    
    print(f"Created {n_ood} synthetic OOD samples")
    print(f"  OOD H range: [{ood_commands[:,0].min():.1f}, {ood_commands[:,0].max():.1f}]")
    print(f"  OOD R range: [{ood_commands[:,1].min():.1f}, {ood_commands[:,1].max():.1f}]")
    
    # Optimizer
    optimizer = AdamW(shield.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        shield.train()
        
        # Shuffle data
        perm_id = torch.randperm(len(train_data['states']))
        perm_ood = torch.randperm(len(ood_states))
        
        id_states = train_data['states'][perm_id][:batch_size * 10].to(device)
        id_commands = train_data['commands'][perm_id][:batch_size * 10].to(device)
        ood_states_batch = ood_states[perm_ood][:batch_size * 10].to(device)
        ood_commands_batch = ood_commands[perm_ood][:batch_size * 10].to(device)
        
        # Train detector
        for i in range(0, len(id_states), batch_size):
            optimizer.zero_grad()
            
            batch_id_states = id_states[i:i+batch_size]
            batch_id_commands = id_commands[i:i+batch_size]
            batch_ood_states = ood_states_batch[i:i+batch_size]
            batch_ood_commands = ood_commands_batch[i:i+batch_size]
            
            # Detector loss
            detector_loss = shield.detector_loss(
                batch_id_states, batch_id_commands,
                batch_ood_states, batch_ood_commands,
            )
            
            # Flow loss (for density estimation)
            flow_loss = shield.flow_loss(batch_id_states, batch_id_commands)
            
            loss = detector_loss + 0.5 * flow_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shield.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation accuracy
        shield.eval()
        with torch.no_grad():
            val_states = val_data['states'][:1000].to(device)
            val_commands = val_data['commands'][:1000].to(device)
            
            # ID accuracy (should predict NOT OOD)
            id_preds = shield.is_ood(val_states, val_commands)
            id_acc = (~id_preds).float().mean().item()
            
            # OOD accuracy (should predict OOD)
            ood_val_commands = create_ood_samples(val_commands.cpu(), 1000).to(device)
            ood_preds = shield.is_ood(val_states, ood_val_commands)
            ood_acc = ood_preds.float().mean().item()
            
            val_acc = (id_acc + ood_acc) / 2
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in shield.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"ID Acc: {id_acc:.1%} | OOD Acc: {ood_acc:.1%}")
    
    # Load best weights
    shield.load_state_dict(best_state)
    
    # Re-calibrate threshold
    shield.calibrate(train_data['states'], train_data['commands'])
    
    print(f"Best validation accuracy: {best_val_acc:.1%}")
    print(f"OOD threshold: {shield.ood_threshold.item():.4f}")
    
    return shield


def test_system(
    policy: UDRLPolicyRobust,
    shield: SafetyShield,
    n_episodes: int = 10,
    device: str = 'cpu',
):
    """Test the complete system on LunarLander."""
    print("\n" + "="*60)
    print("TESTING SHIELDED AGENT")
    print("="*60)
    
    import gymnasium as gym
    
    # Create agents
    agent_no_shield = ShieldedAgent(policy, shield=None, device=device)
    agent_with_shield = ShieldedAgent(policy, shield=shield, device=device)
    
    # Test commands
    commands = {
        'ID (H=200, R=220)': np.array([200.0, 220.0], dtype=np.float32),
        'OOD (H=50, R=350)': np.array([50.0, 350.0], dtype=np.float32),
        'OOD (H=5, R=500)': np.array([5.0, 500.0], dtype=np.float32),
    }
    
    results = {}
    
    for cmd_name, command in commands.items():
        print(f"\n--- {cmd_name} ---")
        
        for agent_name, agent in [("No Shield", agent_no_shield), ("With Shield", agent_with_shield)]:
            agent.reset_stats()
            returns = []
            
            for ep in range(n_episodes):
                env = gym.make('LunarLander-v3', continuous=True)
                state, _ = env.reset(seed=ep)
                total_return = 0
                current_command = command.copy()
                
                for step in range(1000):
                    # Decay horizon
                    current_command[0] = max(1.0, current_command[0] - 1)
                    
                    action, info = agent.act(state, current_command)
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_return += reward
                    
                    if terminated or truncated:
                        break
                
                returns.append(total_return)
                env.close()
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            stats = agent.get_stats()
            
            print(f"  {agent_name}: Return={mean_return:.1f}±{std_return:.1f}, "
                  f"OOD detected={stats['ood_rate']:.1%}")
            
            results[f"{cmd_name}_{agent_name}"] = {
                'mean': mean_return,
                'std': std_return,
                'ood_rate': stats['ood_rate'],
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Robust UDRL + Shield")
    parser.add_argument('--data', type=str, default='data/lunarlander_expert.npz')
    parser.add_argument('--epochs-policy', type=int, default=200)
    parser.add_argument('--epochs-shield', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/robust')
    args = parser.parse_args()
    
    print("="*60)
    print("ROBUST UDRL + SHIELD TRAINING")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, val_data = load_data(args.data)
    
    # Get dimensions
    state_dim = train_data['states'].shape[1]
    action_dim = train_data['actions'].shape[1]
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create models
    policy = UDRLPolicyRobust(state_dim=state_dim, action_dim=action_dim)
    shield = SafetyShield(state_dim=state_dim)
    
    # Train policy
    policy = train_policy(
        policy, train_data, val_data,
        epochs=args.epochs_policy,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    
    # Save policy
    policy_path = output_dir / "policy_robust.pt"
    torch.save(policy.state_dict(), policy_path)
    print(f"Saved policy to {policy_path}")
    
    # Train shield
    shield = train_shield(
        shield, train_data, val_data,
        epochs=args.epochs_shield,
        batch_size=args.batch_size,
        lr=args.lr * 3,
        device=args.device,
    )
    
    # Save shield
    shield_path = output_dir / "shield_robust.pt"
    torch.save({
        'state_dict': shield.state_dict(),
        'command_mean': shield.command_mean,
        'command_std': shield.command_std,
        'command_min': shield.command_min,
        'command_max': shield.command_max,
        'ood_threshold': shield.ood_threshold,
    }, shield_path)
    print(f"Saved shield to {shield_path}")
    
    # Test if requested
    if args.test:
        results = test_system(policy, shield, n_episodes=5, device=args.device)
        
        # Save results
        import json
        results_path = output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()} 
                      for k, v in results.items()}, f, indent=2)
        print(f"Saved results to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
