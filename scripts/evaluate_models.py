#!/usr/bin/env python3
"""
Evaluate and Compare - Évalue et compare tous les modèles.

Charge les modèles depuis: results/<env>/models/
Sauvegarde les résultats dans: results/<env>/figures/ et results/<env>/metrics/

Usage:
    python scripts/evaluate.py --env lunarlander --data data/lunarlander_xxx.npz
    python scripts/evaluate.py --env lunarlander --data data/lunarlander_xxx.npz --episodes 20
    python scripts/evaluate.py --env lunarlander --data data/lunarlander_xxx.npz --offline
"""

import argparse
import json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt

from models import (
    UDRLPolicy, QuantileShield, FlowMatchingShield, DiffusionShield,
    load_data, ensure_results_dirs, load_model, get_results_dir, load_diffusion_shield
)


def create_environment(env_name: str):
    """Create the appropriate environment."""
    if env_name == "lunarlander":
        env = gym.make("LunarLander-v3", continuous=True)
    elif env_name == "cartpole":
        env = gym.make("CartPole-v1")
    elif env_name == "pendulum":
        env = gym.make("Pendulum-v1")
    else:
        env = gym.make(env_name)
    return env


def evaluate_policy(
    env,
    policy: UDRLPolicy,
    commands: list,
    shield=None,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Evaluate policy with optional shield on a list of commands.
    """
    returns = []
    lengths = []
    successes = []
    crashes = []
    ood_detections = 0
    projections = 0
    
    for cmd in commands:
        state, _ = env.reset()
        episode_return = 0
        steps = 0
        done = False
        
        target_horizon, target_return = cmd
        
        while not done and steps < 500:
            state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
            cmd_t = torch.tensor([[target_horizon - steps, target_return - episode_return]], 
                                 dtype=torch.float32, device=device)
            
            # Check OOD and project if needed
            if shield is not None:
                is_ood = shield.is_ood(state_t, cmd_t)
                if is_ood.any():
                    ood_detections += 1
                    cmd_t = shield.project(state_t, cmd_t)
                    projections += 1
            
            # Sample action
            with torch.no_grad():
                action = policy.sample(state_t, cmd_t, deterministic=True)
            action = action.cpu().numpy()[0]
            action = np.clip(action, -1, 1)
            
            # Step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            steps += 1
        
        returns.append(episode_return)
        lengths.append(steps)
        
        # Success/crash detection for LunarLander
        if hasattr(env.unwrapped, 'lander'):
            successes.append(episode_return > 200)
            crashes.append(episode_return < -100)
        else:
            successes.append(episode_return > 0)
            crashes.append(episode_return < -100)
    
    results = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'success_rate': np.mean(successes) if successes else 0,
        'crash_rate': np.mean(crashes) if crashes else 0,
        'ood_detections': ood_detections,
        'projections': projections,
        'returns': returns,
    }
    
    return results


def plot_comparison(results: dict, save_path: Path):
    """Plot comparison bar chart."""
    methods = list(results.keys())
    returns = [results[m]['mean_return'] for m in methods]
    stds = [results[m]['std_return'] for m in methods]
    
    colors = ['gray', 'green', 'purple', 'red'][:len(methods)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Returns
    axes[0].bar(methods, returns, yerr=stds, color=colors, capsize=5)
    axes[0].set_ylabel('Mean Return')
    axes[0].set_title('Return Comparison')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Success/Crash rate
    success_rates = [results[m]['success_rate'] * 100 for m in methods]
    crash_rates = [results[m]['crash_rate'] * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[1].bar(x - width/2, success_rates, width, label='Success %', color='green', alpha=0.7)
    axes[1].bar(x + width/2, crash_rates, width, label='Crash %', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15)
    axes[1].set_ylabel('Rate (%)')
    axes[1].set_title('Success/Crash Rates')
    axes[1].legend()
    
    # OOD detections
    ood = [results[m]['ood_detections'] for m in methods]
    proj = [results[m]['projections'] for m in methods]
    axes[2].bar(x - width/2, ood, width, label='OOD Detected', color='orange', alpha=0.7)
    axes[2].bar(x + width/2, proj, width, label='Projections', color='blue', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15)
    axes[2].set_ylabel('Count')
    axes[2].set_title('OOD Detection & Projection')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def plot_ood_detection(
    shield,
    data: dict,
    shield_name: str,
    save_path: Path,
    n_samples: int = 1000,
    device: str = "cpu",
):
    """Visualize OOD detection boundaries."""
    # Skip Flow Matching for now due to gradient requirements in log_prob
    if shield_name == "Flow Matching":
        print(f"Skipping OOD visualization for {shield_name} (gradient computation required)")
        return
    
    states = torch.tensor(data['states'][:n_samples], dtype=torch.float32, device=device)
    commands = data['commands'][:n_samples]
    
    # Get command ranges
    h_min, h_max = commands[:, 0].min(), commands[:, 0].max()
    r_min, r_max = commands[:, 1].min(), commands[:, 1].max()
    
    # Extend range for OOD visualization
    h_range = np.linspace(h_min - 50, h_max + 50, 50)
    r_range = np.linspace(r_min - 100, r_max + 100, 50)
    
    # Use mean state for visualization
    mean_state = states.mean(dim=0, keepdim=True).repeat(len(h_range) * len(r_range), 1)
    
    # Create grid
    H, R = np.meshgrid(h_range, r_range)
    grid_commands = torch.tensor(
        np.stack([H.flatten(), R.flatten()], axis=1),
        dtype=torch.float32, device=device
    )
    
    # Compute OOD scores
    with torch.no_grad():
        ood = shield.is_ood(mean_state, grid_commands)
    
    ood_grid = ood.cpu().numpy().reshape(H.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(H, R, ood_grid, levels=1, colors=['lightgreen', 'lightcoral'], alpha=0.5)
    plt.contour(H, R, ood_grid, levels=[0.5], colors='red', linewidths=2)
    plt.scatter(commands[:500, 0], commands[:500, 1], c='blue', s=5, alpha=0.3, label='Training data')
    plt.xlabel('Horizon (H)')
    plt.ylabel('Return-to-go (R)')
    plt.title(f'{shield_name} - OOD Detection Boundary')
    plt.legend()
    plt.colorbar(label='OOD (1) / In-distribution (0)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved OOD visualization to {save_path}")


def offline_analysis(data: dict, shields: dict, results_dir: Path, device: str = "cpu"):
    """
    Analyse offline sans environnement.
    Génère les visualisations OOD et les statistiques.
    """
    print("\n" + "=" * 60)
    print("OFFLINE ANALYSIS (no environment needed)")
    print("=" * 60)
    
    commands_data = data['commands']
    states_data = data['states']
    
    h_mean, r_mean = commands_data.mean(axis=0)
    h_std, r_std = commands_data.std(axis=0)
    
    print(f"\nTraining data statistics:")
    print(f"  Horizon: {h_mean:.1f} ± {h_std:.1f}")
    print(f"  Return: {r_mean:.1f} ± {r_std:.1f}")
    
    # Test various commands
    test_cmds = [
        # In-distribution
        ("In-dist: mean", h_mean, r_mean),
        ("In-dist: H-std", h_mean - h_std, r_mean),
        ("In-dist: R+std", h_mean, r_mean + r_std),
        # Border cases
        ("Border: H-2std", h_mean - 2*h_std, r_mean),
        ("Border: R+2std", h_mean, r_mean + 2*r_std),
        # OOD
        ("OOD: ambitious", h_mean - 2*h_std, r_mean + 2*r_std),
        ("OOD: very high R", 50, 300),
        ("OOD: extreme", 20, 500),
    ]
    
    # Analyze OOD detection for each shield
    mean_state = torch.tensor(
        states_data[:1000].mean(axis=0, keepdims=True),
        dtype=torch.float32, device=device
    )
    
    print("\n" + "-" * 80)
    print(f"{'Command':<25} {'H':>8} {'R':>10}", end="")
    for name in shields.keys():
        print(f" | {name:>15}", end="")
    print()
    print("-" * 80)
    
    for label, h, r in test_cmds:
        cmd_t = torch.tensor([[h, r]], dtype=torch.float32, device=device)
        print(f"{label:<25} {h:>8.1f} {r:>10.1f}", end="")
        
        for name, shield in shields.items():
            with torch.no_grad():
                is_ood = shield.is_ood(mean_state, cmd_t).item()
            status = "OOD" if is_ood else "OK"
            print(f" | {status:>15}", end="")
        print()
    
    print("-" * 80)
    
    # Generate OOD visualizations
    for name, shield in shields.items():
        ood_path = results_dir / "figures" / f"ood_{name.lower().replace(' ', '_')}.png"
        plot_ood_detection(shield, data, name, ood_path, device=device)
    
    print(f"\nOOD visualizations saved in: {results_dir / 'figures'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Models")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name (must match trained models)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data file (.npz)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per command")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--offline", action="store_true",
                        help="Offline analysis only (no environment, just OOD visualization)")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("EVALUATION & COMPARISON" + (" (OFFLINE MODE)" if args.offline else ""))
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    data = load_data(args.data)
    state_dim = data['state_dim']
    action_dim = data['action_dim']
    
    # Ensure results directories
    results_dir = ensure_results_dirs(args.env)
    
    # Load shields
    shields = {}
    
    # Try loading each shield
    try:
        quantile = QuantileShield(state_dim)
        load_model(quantile, args.env, "quantile_shield")
        quantile = quantile.to(device)
        quantile.eval()
        shields['Quantile'] = quantile
        print("Loaded Quantile Shield")
    except FileNotFoundError:
        print("Quantile Shield not found, skipping...")
    
    try:
        flow = FlowMatchingShield(state_dim)
        load_model(flow, args.env, "flow_shield")
        flow = flow.to(device)
        flow.eval()
        shields['Flow Matching'] = flow
        print("Loaded Flow Matching Shield")
    except FileNotFoundError:
        print("Flow Shield not found, skipping...")
    
    try:
        diffusion = load_diffusion_shield(args.env, state_dim)
        diffusion = diffusion.to(device)
        diffusion.eval()
        shields['Diffusion'] = diffusion
    except FileNotFoundError:
        print("Diffusion Shield not found, skipping...")
    
    if not shields:
        print("ERROR: No shields found. Train at least one shield first.")
        return
    
    # If offline mode, just do analysis without environment
    if args.offline:
        offline_analysis(data, shields, results_dir, device=device)
        print("\n" + "=" * 60)
        print("OFFLINE ANALYSIS COMPLETE")
        print("=" * 60)
        return
    
    # Create environment (only if not offline)
    env = create_environment(args.env)
    
    # Load policy (only needed for online evaluation)
    print("\nLoading policy...")
    
    policy = UDRLPolicy(state_dim, action_dim)
    try:
        load_model(policy, args.env, "policy")
        policy = policy.to(device)
        policy.eval()
    except FileNotFoundError:
        print("ERROR: Policy not found. Train it first with train_policy.py")
        return
    
    # Generate test commands (mix of ID and OOD)
    commands_data = data['commands']
    h_mean, r_mean = commands_data.mean(axis=0)
    h_std, r_std = commands_data.std(axis=0)
    
    test_commands = [
        # In-distribution
        (h_mean, r_mean),
        (h_mean - h_std, r_mean),
        (h_mean + h_std, r_mean - r_std),
        # Out-of-distribution
        (h_mean - 2*h_std, r_mean + 2*r_std),  # Ambitious
        (50, r_mean + 3*r_std),  # Very ambitious
        (20, 200),  # Unrealistic for random data
        (30, 250),  # Even more unrealistic
        (h_mean, r_mean + r_std),
    ]
    
    print(f"\nTest commands ({len(test_commands)} total):")
    for i, (h, r) in enumerate(test_commands):
        print(f"  {i+1}. H={h:.0f}, R={r:.1f}")
    
    # Evaluate
    results = {}
    
    print("\nEvaluating No Shield...")
    results['No Shield'] = evaluate_policy(env, policy, test_commands, shield=None, device=device)
    
    for name, shield in shields.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_policy(env, policy, test_commands, shield=shield, device=device)
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Method':<20} {'Return':>12} {'Success %':>12} {'Crash %':>12} {'OOD Det.':>10} {'Proj.':>10}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<20} {res['mean_return']:>7.1f} ± {res['std_return']:<4.1f} "
              f"{res['success_rate']*100:>10.1f}% {res['crash_rate']*100:>10.1f}% "
              f"{res['ood_detections']:>10} {res['projections']:>10}")
    print("=" * 80)
    
    # Save results
    results_json = {k: {kk: vv for kk, vv in v.items() if kk != 'returns'} for k, v in results.items()}
    json_path = results_dir / "metrics" / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved metrics to {json_path}")
    
    # Plot comparison
    plot_path = results_dir / "figures" / "comparison.png"
    plot_comparison(results, plot_path)
    
    # Plot OOD detection for each shield
    for name, shield in shields.items():
        ood_path = results_dir / "figures" / f"ood_{name.lower().replace(' ', '_')}.png"
        plot_ood_detection(shield, data, name, ood_path, device=device)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved in: results/{args.env}/")
    print(f"  - figures/comparison.png")
    print(f"  - figures/ood_*.png")
    print(f"  - metrics/comparison_results.json")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
