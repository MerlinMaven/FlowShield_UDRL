"""
Script pour déboguer le shield et voir comment il projette les commandes OOD.
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_shield(checkpoint_path):
    """Load Flow Matching Shield from checkpoint."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.safety.flow_matching import FlowMatchingSafetyModule
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Create shield
    shield = FlowMatchingSafetyModule(
        state_dim=8,
        command_dim=2,
        hidden_dim=256,
        n_layers=4,
        ood_threshold=ckpt.get("ood_threshold", -5.0),
        projection_method="gradient",
        projection_steps=50,
        projection_lr=0.1,
    )
    
    # Load weights
    if "model_state_dict" in ckpt:
        shield.load_state_dict(ckpt["model_state_dict"])
    else:
        shield.load_state_dict(ckpt)
    
    shield.eval()
    print(f"Shield loaded with threshold: {shield.ood_threshold:.4f}")
    
    return shield


def analyze_projection(shield, state, original_command):
    """Analyze step-by-step projection."""
    print("\n" + "=" * 60)
    print("PROJECTION ANALYSIS")
    print("=" * 60)
    
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    g = original_command.clone().requires_grad_(True)
    
    # Initial log prob
    with torch.no_grad():
        initial_log_prob = shield.flow_model.log_prob(state_t, original_command)
    print(f"\nOriginal command: H={original_command[0, 0]:.1f}, R={original_command[0, 1]:.1f}")
    print(f"Initial log_prob: {initial_log_prob.item():.4f}")
    print(f"Threshold: {shield.ood_threshold:.4f}")
    print(f"Is OOD: {initial_log_prob.item() < shield.ood_threshold}")
    
    # Projection steps
    print(f"\nGradient ascent (lr={shield.projection_lr}, max_steps={shield.projection_steps}):")
    print(f"{'Step':<6} {'H':<8} {'R':<10} {'log_prob':<12} {'grad_norm':<12} {'Safe?'}")
    print("-" * 65)
    
    for step in range(shield.projection_steps):
        # Compute log prob
        log_prob = shield.flow_model.log_prob(state_t, g)
        
        # Check if safe
        is_safe = log_prob.item() > shield.ood_threshold
        
        # Print every 5 steps
        if step % 5 == 0 or is_safe:
            grad = torch.autograd.grad(log_prob.sum(), g, retain_graph=True)[0]
            grad_norm = grad.norm().item()
            print(f"{step:<6} {g[0, 0].item():<8.1f} {g[0, 1].item():<10.1f} "
                  f"{log_prob.item():<12.4f} {grad_norm:<12.4f} {is_safe}")
        
        if is_safe:
            print(f"\n✓ Converged to safe region at step {step}")
            break
        
        # Gradient ascent
        grad = torch.autograd.grad(log_prob.sum(), g)[0]
        with torch.no_grad():
            g = g + shield.projection_lr * grad
            g = g.requires_grad_(True)
    else:
        print(f"\n✗ Did NOT converge after {shield.projection_steps} steps")
    
    with torch.no_grad():
        final_log_prob = shield.flow_model.log_prob(state_t, g)
    
    print(f"\nFinal command: H={g[0, 0].item():.1f}, R={g[0, 1].item():.1f}")
    print(f"Final log_prob: {final_log_prob.item():.4f}")
    print(f"Final is safe: {final_log_prob.item() > shield.ood_threshold}")
    print(f"Projection distance: {torch.norm(g - original_command).item():.2f}")
    
    return g.detach()


def compare_projection_methods(shield, state, original_command):
    """Compare different projection hyperparameters."""
    print("\n" + "=" * 60)
    print("PROJECTION METHOD COMPARISON")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.safety.flow_matching import FlowMatchingSafetyModule
    
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Try different configurations
    configs = [
        {"projection_steps": 50, "projection_lr": 0.1},
        {"projection_steps": 100, "projection_lr": 0.1},
        {"projection_steps": 50, "projection_lr": 0.5},
        {"projection_steps": 200, "projection_lr": 0.2},
    ]
    
    print(f"\n{'Steps':<8} {'LR':<8} {'Final H':<10} {'Final R':<12} {'log_prob':<12} {'Safe?':<8} {'Distance'}")
    print("-" * 75)
    
    for config in configs:
        # Temporary shield with different params
        temp_shield = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=256,
            n_layers=4,
            ood_threshold=shield.ood_threshold,
            projection_method="gradient",
            **config,
        )
        temp_shield.flow_model = shield.flow_model  # Use same trained model
        
        # Project
        projected = temp_shield.project(state_t, original_command)
        log_prob = temp_shield.flow_model.log_prob(state_t, projected)
        is_safe = log_prob.item() > temp_shield.ood_threshold
        distance = torch.norm(projected - original_command).item()
        
        print(f"{config['projection_steps']:<8} {config['projection_lr']:<8.1f} "
              f"{projected[0, 0].item():<10.1f} {projected[0, 1].item():<12.1f} "
              f"{log_prob.item():<12.4f} {is_safe:<8} {distance:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shield-path", default="results/unknown/models/flow_shield.pt")
    parser.add_argument("--horizon", type=float, default=5)
    parser.add_argument("--target-return", type=float, default=500)
    args = parser.parse_args()
    
    print("=" * 60)
    print("SHIELD DEBUG ANALYSIS")
    print("=" * 60)
    
    # Load shield
    print(f"\nLoading shield: {args.shield_path}")
    shield = load_shield(args.shield_path)
    
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    
    # Get initial state
    state, _ = env.reset(seed=42)
    
    # Test command
    command = torch.tensor([[args.horizon, args.target_return]], dtype=torch.float32)
    
    # Analyze projection
    projected = analyze_projection(shield, state, command)
    
    # Compare methods
    compare_projection_methods(shield, state, command)
    
    # Sample achievable commands from this state
    print("\n" + "=" * 60)
    print("ACHIEVABLE COMMANDS (sampled from learned distribution)")
    print("=" * 60)
    
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    samples = shield.sample_commands(state_t, n_samples=20)
    
    print(f"\n{'Sample':<8} {'Horizon':<10} {'Return':<10} {'log_prob'}")
    print("-" * 40)
    
    for i in range(20):
        sample = samples[0, i:i+1]
        log_prob = shield.flow_model.log_prob(state_t, sample)
        print(f"{i+1:<8} {sample[0, 0].item():<10.1f} {sample[0, 1].item():<10.1f} "
              f"{log_prob.item():.4f}")
    
    mean_h = samples[0, :, 0].mean().item()
    mean_r = samples[0, :, 1].mean().item()
    std_h = samples[0, :, 0].std().item()
    std_r = samples[0, :, 1].std().item()
    
    print(f"\nMean achievable: H={mean_h:.1f}±{std_h:.1f}, R={mean_r:.1f}±{std_r:.1f}")
    
    env.close()


if __name__ == "__main__":
    main()
