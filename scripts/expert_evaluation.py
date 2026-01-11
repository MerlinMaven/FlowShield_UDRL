#!/usr/bin/env python3
"""
Expert Evaluation Pipeline - Génère toutes les visualisations et métriques.

Ce script effectue:
1. Évaluation des 4 configurations (sans shield + 3 shields)
2. Tests ID et OOD
3. GIFs comparatifs
4. Tableaux de métriques
5. Courbes de comparaison
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from models import UDRLPolicy, FlowMatchingShield, QuantileShield, DiffusionShield


def load_policy(checkpoint_path, device='cpu'):
    """Load UDRL policy."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # Detect architecture from checkpoint
    hidden_dim = state_dict["net.input_proj.weight"].shape[0]
    
    # Detect command_embed_dim from Fourier features
    if "command_embed.B" in state_dict:
        # Fourier features: B has shape [command_dim, embed_dim//2]
        command_embed_dim = state_dict["command_embed.B"].shape[1] * 2
    elif "command_net.0.weight" in state_dict:
        command_embed_dim = state_dict["command_net.0.weight"].shape[0]
    else:
        command_embed_dim = 64
    
    # Detect dimensions
    input_proj_in = state_dict["net.input_proj.weight"].shape[1]
    state_dim = input_proj_in - command_embed_dim
    
    # Output proj gives mean+log_std, so action_dim = output_dim / 2
    output_dim = state_dict["net.output_proj.weight"].shape[0]
    action_dim = output_dim // 2
    
    print(f"  Architecture: state_dim={state_dim}, action_dim={action_dim}, "
          f"hidden={hidden_dim}, cmd_embed={command_embed_dim}")
    
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        command_embed_dim=command_embed_dim,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_shield(shield_type, checkpoint_path, device='cpu'):
    """Load a shield model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if shield_type == 'flow':
        hidden_dim = ckpt["net.input_proj.weight"].shape[0]
        shield = FlowMatchingShield(state_dim=8, hidden_dim=hidden_dim)
        
        # Load only the model weights, not auxiliary buffers
        model_keys = [k for k in ckpt.keys() if k.startswith(('net.', 'state_norm.', 'command_norm.', 'time_embed.'))]
        state_dict = {k: ckpt[k] for k in model_keys}
        
        # Load with strict=False to allow missing buffers
        shield.load_state_dict(state_dict, strict=False)
        
        # Set threshold from checkpoint or use default
        if 'ood_threshold' in ckpt:
            if isinstance(ckpt['ood_threshold'], torch.Tensor):
                shield.ood_threshold = ckpt['ood_threshold']
            else:
                shield.ood_threshold = torch.tensor(ckpt['ood_threshold'])
        
        if 'log_prob_mean' in ckpt:
            shield.log_prob_mean = ckpt['log_prob_mean'] if isinstance(ckpt['log_prob_mean'], torch.Tensor) else torch.tensor(ckpt['log_prob_mean'])
        if 'log_prob_std' in ckpt:
            shield.log_prob_std = ckpt['log_prob_std'] if isinstance(ckpt['log_prob_std'], torch.Tensor) else torch.tensor(ckpt['log_prob_std'])
            
    elif shield_type == 'quantile':
        hidden_dim = ckpt.get("hidden_dim", 256)
        if isinstance(hidden_dim, torch.Tensor):
            hidden_dim = int(hidden_dim.item())
        shield = QuantileShield(state_dim=8, hidden_dim=int(hidden_dim))
        
        model_keys = [k for k in ckpt.keys() if k.startswith(('net.', 'state_norm.', 'command_norm.'))]
        state_dict = {k: ckpt[k] for k in model_keys}
        shield.load_state_dict(state_dict, strict=False)
        
        if 'ood_threshold' in ckpt:
            if isinstance(ckpt['ood_threshold'], torch.Tensor):
                shield.ood_threshold = ckpt['ood_threshold']
            else:
                shield.ood_threshold = torch.tensor(ckpt['ood_threshold'])
            
    elif shield_type == 'diffusion':
        hidden_dim = ckpt["net.input_proj.weight"].shape[0]
        shield = DiffusionShield(state_dim=8, hidden_dim=hidden_dim)
        
        model_keys = [k for k in ckpt.keys() if k.startswith(('net.', 'state_norm.', 'command_norm.', 'betas', 'alphas'))]
        state_dict = {k: ckpt[k] for k in model_keys}
        shield.load_state_dict(state_dict, strict=False)
        
        if 'ood_threshold' in ckpt:
            if isinstance(ckpt['ood_threshold'], torch.Tensor):
                shield.ood_threshold = ckpt['ood_threshold']
            else:
                shield.ood_threshold = torch.tensor(ckpt['ood_threshold'])
    else:
        raise ValueError(f"Unknown shield type: {shield_type}")
    
    shield.eval()
    return shield


def run_episode(env, policy, shield, command, device='cpu', max_steps=1000, record=False):
    """Run single episode with optional shield."""
    state, _ = env.reset()
    frames = [env.render()] if record else []
    
    total_return = 0
    shield_activations = 0
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    command_t = torch.FloatTensor(command).unsqueeze(0).to(device)
    current_command = command_t.clone()
    
    for step in range(max_steps):
        with torch.no_grad():
            # Apply shield if present
            if shield is not None:
                is_ood = shield.is_ood(state_t, current_command)
                if is_ood.any():
                    current_command = shield.project(state_t, current_command)
                    shield_activations += 1
            
            # Get action
            action = policy.sample(state_t, current_command, deterministic=True)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        
        if record:
            frames.append(env.render())
        
        if terminated or truncated:
            break
        
        state = next_state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Update command
        current_command[0, 0] = max(1, current_command[0, 0] - 1)
        current_command[0, 1] = current_command[0, 1] - reward
    
    # Determine success (landed = terminated without crash)
    success = terminated and total_return > 100
    
    return {
        'return': total_return,
        'length': step + 1,
        'success': success,
        'shield_activations': shield_activations,
        'frames': frames if record else None
    }


def save_gif(frames, output_path, fps=30):
    """Save frames as GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,
        loop=0
    )


def evaluate_configuration(env, policy, shield, command, n_episodes=20, device='cpu', 
                          record_episodes=3, name="config"):
    """Evaluate a configuration over multiple episodes."""
    results = []
    frames_list = []
    
    for i in tqdm(range(n_episodes), desc=f"Evaluating {name}"):
        record = i < record_episodes
        result = run_episode(env, policy, shield, command, device, record=record)
        results.append(result)
        
        if record and result['frames']:
            frames_list.append(result['frames'])
    
    # Compute statistics
    returns = [r['return'] for r in results]
    lengths = [r['length'] for r in results]
    successes = [r['success'] for r in results]
    activations = [r['shield_activations'] for r in results]
    
    stats = {
        'name': name,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_length': np.mean(lengths),
        'success_rate': np.mean(successes) * 100,
        'crash_rate': (1 - np.mean(successes)) * 100,
        'mean_activations': np.mean(activations),
        'returns': returns,
        'frames_list': frames_list
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Expert Evaluation Pipeline")
    parser.add_argument("--models-dir", default="results/final/models")
    parser.add_argument("--output-dir", default="results/final")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--record-episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERT EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {args.models_dir}")
    print(f"Episodes per config: {args.n_episodes}")
    print("=" * 70)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    gifs_dir = output_dir / "gifs"
    tables_dir = output_dir / "tables"
    
    for d in [figures_dir / "comparison", figures_dir / "evaluation", 
              gifs_dir / "id_commands", gifs_dir / "ood_commands", 
              gifs_dir / "shield_comparison", tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n📦 Loading models...")
    models_dir = Path(args.models_dir)
    
    policy = load_policy(models_dir / "policy.pt", args.device)
    print(f"  ✓ Policy loaded")
    
    shields = {}
    shield_files = {
        'flow': 'flow_shield.pt',
        'quantile': 'quantile_shield.pt',
        'diffusion': 'diffusion_shield.pt'
    }
    
    for shield_type, filename in shield_files.items():
        path = models_dir / filename
        if path.exists():
            try:
                shields[shield_type] = load_shield(shield_type, path, args.device)
                print(f"  ✓ {shield_type.capitalize()} Shield loaded")
            except Exception as e:
                print(f"  ✗ {shield_type.capitalize()} Shield failed: {e}")
                shields[shield_type] = None
        else:
            print(f"  ✗ {shield_type.capitalize()} Shield not found")
            shields[shield_type] = None
    
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    
    # Define test commands
    commands = {
        'ID': [200.0, 220.0],      # In-distribution
        'OOD_moderate': [50.0, 350.0],   # Moderately OOD
        'OOD_extreme': [5.0, 500.0],     # Extremely OOD
    }
    
    # Configurations to test
    configs = {
        'No Shield': None,
        'Flow Shield': shields.get('flow'),
        'Quantile Shield': shields.get('quantile'),
        'Diffusion Shield': shields.get('diffusion'),
    }
    
    # Run evaluations
    all_results = []
    
    for cmd_name, command in commands.items():
        print(f"\n{'='*70}")
        print(f"COMMAND: {cmd_name} (H={command[0]}, R={command[1]})")
        print("=" * 70)
        
        cmd_results = {}
        
        for config_name, shield in configs.items():
            if shield is None and config_name != 'No Shield':
                continue
                
            stats = evaluate_configuration(
                env, policy, shield, command,
                n_episodes=args.n_episodes,
                record_episodes=args.record_episodes,
                device=args.device,
                name=f"{cmd_name}_{config_name}"
            )
            
            cmd_results[config_name] = stats
            
            # Print stats
            print(f"\n{config_name}:")
            print(f"  Return: {stats['mean_return']:.1f} ± {stats['std_return']:.1f}")
            print(f"  Success: {stats['success_rate']:.0f}%")
            if shield:
                print(f"  Activations: {stats['mean_activations']:.1f}/ep")
            
            # Save GIFs
            if stats['frames_list']:
                gif_subdir = gifs_dir / ("id_commands" if cmd_name == "ID" else "ood_commands")
                for i, frames in enumerate(stats['frames_list']):
                    gif_path = gif_subdir / f"{cmd_name}_{config_name.replace(' ', '_')}_ep{i+1}.gif"
                    save_gif(frames, gif_path)
            
            # Add to all results
            all_results.append({
                'Command': cmd_name,
                'Configuration': config_name,
                'Mean Return': stats['mean_return'],
                'Std Return': stats['std_return'],
                'Min Return': stats['min_return'],
                'Max Return': stats['max_return'],
                'Success Rate (%)': stats['success_rate'],
                'Crash Rate (%)': stats['crash_rate'],
                'Shield Activations': stats['mean_activations']
            })
        
        # Create comparison bar chart for this command
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        config_names = [k for k in cmd_results.keys()]
        returns = [cmd_results[k]['mean_return'] for k in config_names]
        stds = [cmd_results[k]['std_return'] for k in config_names]
        success_rates = [cmd_results[k]['success_rate'] for k in config_names]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Returns
        axes[0].bar(config_names, returns, yerr=stds, capsize=5, color=colors[:len(config_names)])
        axes[0].set_ylabel('Mean Return')
        axes[0].set_title(f'{cmd_name}: Return Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Variance
        axes[1].bar(config_names, stds, color=colors[:len(config_names)])
        axes[1].set_ylabel('Std Dev')
        axes[1].set_title(f'{cmd_name}: Variance Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Success Rate
        axes[2].bar(config_names, success_rates, color=colors[:len(config_names)])
        axes[2].set_ylabel('Success Rate (%)')
        axes[2].set_title(f'{cmd_name}: Success Rate')
        axes[2].set_ylim(0, 100)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "comparison" / f"comparison_{cmd_name}.png", dpi=150)
        plt.close()
    
    env.close()
    
    # Save results table
    df = pd.DataFrame(all_results)
    df.to_csv(tables_dir / "evaluation_results.csv", index=False)
    
    # Create summary table as markdown
    summary = df.pivot_table(
        index='Configuration',
        columns='Command',
        values=['Mean Return', 'Success Rate (%)'],
        aggfunc='first'
    )
    
    with open(tables_dir / "summary_table.md", 'w') as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n## Full Results\n\n")
        f.write(df.to_markdown(index=False))
    
    # Create final comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Return by Command Type
    ax = axes[0, 0]
    x = np.arange(len(commands))
    width = 0.2
    for i, (config_name, _) in enumerate(configs.items()):
        config_data = df[df['Configuration'] == config_name]
        if len(config_data) > 0:
            ax.bar(x + i*width, config_data['Mean Return'].values, width, 
                   label=config_name, color=colors[i])
    ax.set_xlabel('Command Type')
    ax.set_ylabel('Mean Return')
    ax.set_title('Return by Command Type and Shield')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(list(commands.keys()))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    for i, (config_name, _) in enumerate(configs.items()):
        config_data = df[df['Configuration'] == config_name]
        if len(config_data) > 0:
            ax.bar(x + i*width, config_data['Success Rate (%)'].values, width, 
                   label=config_name, color=colors[i])
    ax.set_xlabel('Command Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Command Type and Shield')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(list(commands.keys()))
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Variance Reduction
    ax = axes[1, 0]
    for i, (config_name, _) in enumerate(configs.items()):
        config_data = df[df['Configuration'] == config_name]
        if len(config_data) > 0:
            ax.bar(x + i*width, config_data['Std Return'].values, width, 
                   label=config_name, color=colors[i])
    ax.set_xlabel('Command Type')
    ax.set_ylabel('Std Dev')
    ax.set_title('Variance by Command Type and Shield')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(list(commands.keys()))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Shield Improvement Summary (OOD only)
    ax = axes[1, 1]
    ood_data = df[df['Command'] == 'OOD_extreme']
    if len(ood_data) > 0:
        baseline = ood_data[ood_data['Configuration'] == 'No Shield']['Mean Return'].values
        if len(baseline) > 0:
            baseline = baseline[0]
            improvements = []
            labels = []
            for config_name in ['Flow Shield', 'Quantile Shield', 'Diffusion Shield']:
                config_val = ood_data[ood_data['Configuration'] == config_name]['Mean Return'].values
                if len(config_val) > 0:
                    improvement = ((config_val[0] - baseline) / abs(baseline)) * 100
                    improvements.append(improvement)
                    labels.append(config_name.replace(' Shield', ''))
            
            colors_imp = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
            ax.bar(labels, improvements, color=colors_imp)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Shield Improvement over Baseline (OOD Extreme)')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "evaluation" / "final_comparison.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results saved to:")
    print(f"  - Tables: {tables_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - GIFs: {gifs_dir}")
    
    # Print summary
    print("\n📈 SUMMARY:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
