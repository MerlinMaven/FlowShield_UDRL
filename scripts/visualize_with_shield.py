"""
Visualise des episodes avec UDRL + Flow Shield.

Compare le comportement avec et sans shield sur des commandes OOD.
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from models import UDRLPolicy, FlowMatchingShield


def load_policy(model_path, state_dim=8, action_dim=2, device='cpu'):
    """Charge la politique UDRL."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    b_shape = state_dict['command_embed.B'].shape
    command_dim = b_shape[0]
    command_embed_dim = b_shape[1] * 2
    hidden_dim = state_dict['net.input_proj.weight'].shape[0]
    
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        command_dim=command_dim,
        hidden_dim=hidden_dim,
        command_embed_dim=command_embed_dim,
    )
    
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def load_shield(model_path, state_dim=8, device='cpu'):
    """Charge le Flow Shield."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer hidden_dim from net structure
    hidden_dim = state_dict['net.input_proj.weight'].shape[0]
    
    # FlowMatchingShield doesn't take projection params as constructor args
    # They're passed to project() method directly
    shield = FlowMatchingShield(
        state_dim=state_dim,
        command_dim=2,
        hidden_dim=hidden_dim,
    )
    
    shield.load_state_dict(state_dict)
    
    # Load threshold
    if 'threshold' in checkpoint:
        shield.ood_threshold = checkpoint['threshold']
    elif 'ood_threshold' in checkpoint:
        shield.ood_threshold = checkpoint['ood_threshold']
    
    print(f'  Threshold: {shield.ood_threshold:.3f}')
    print(f'  Projection: gradient ascent (200 steps, LR=0.5)')
    
    shield.to(device)
    shield.eval()
    return shield


def run_episode_with_shield(env, policy, shield, command, device='cpu', max_steps=1000):
    """Execute un episode avec shield actif."""
    state, info = env.reset()
    
    frames = [env.render()]
    total_return = 0
    shield_activations = 0
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    command_t = torch.FloatTensor(command).unsqueeze(0).to(device)
    current_command = command_t.clone()
    
    for step in range(max_steps):
        with torch.no_grad():
            # Check if OOD and project if needed
            is_ood = shield.is_ood(state_t, current_command, fast=True)
            
            if is_ood.any():
                # OOD detected - project to safe command
                safe_command = shield.project(state_t, current_command)
                current_command = safe_command
                shield_activations += 1
            
            action = policy.sample(state_t, current_command, deterministic=True)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_return += reward
        frames.append(env.render())
        
        if terminated or truncated:
            break
        
        state = next_state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        current_command[0, 0] = max(1, current_command[0, 0] - 1)
        current_command[0, 1] = current_command[0, 1] - reward
    
    return frames, total_return, step + 1, shield_activations


def save_gif(frames, output_path, fps=30):
    """Sauvegarde les frames en GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,
        loop=0
    )


def main():
    parser = argparse.ArgumentParser(description='Visualise UDRL avec Shield')
    parser.add_argument('--policy', type=str, 
                       default='results/unknown/models/policy.pt')
    parser.add_argument('--shield', type=str,
                       default='results/unknown/models/flow_shield.pt')
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--horizon', type=float, default=5)
    parser.add_argument('--target-return', type=float, default=500)
    parser.add_argument('--output-dir', type=str, 
                       default='results/lunarlander/figures/with_shield')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    print('='*60)
    print('VISUALISATION UDRL + FLOW SHIELD')
    print('='*60)
    
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f'\nChargement policy: {args.policy}')
    policy = load_policy(args.policy, device=device)
    
    print(f'Chargement shield: {args.shield}')
    shield = load_shield(args.shield, device=device)
    
    # Create environment
    env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    
    command = [args.horizon, args.target_return]
    print(f'\nCommand OOD: H={command[0]}, R={command[1]}')
    
    # Run episodes
    all_returns = []
    all_lengths = []
    all_activations = []
    
    for ep in range(args.n_episodes):
        print(f'\nEpisode {ep+1}/{args.n_episodes}...')
        
        frames, total_return, length, activations = run_episode_with_shield(
            env, policy, shield, command, device=device
        )
        
        all_returns.append(total_return)
        all_lengths.append(length)
        all_activations.append(activations)
        
        print(f'  Return: {total_return:.1f}')
        print(f'  Length: {length} steps')
        print(f'  Shield activations: {activations}')
        
        # Save GIF
        gif_path = output_dir / f'shielded_episode_{ep+1}.gif'
        save_gif(frames, gif_path, fps=30)
        print(f'  Saved: {gif_path}')
    
    env.close()
    
    # Summary
    print('\n' + '='*60)
    print('RESUME AVEC SHIELD')
    print('='*60)
    print(f'Episodes: {args.n_episodes}')
    print(f'Mean Return: {np.mean(all_returns):.1f} +/- {np.std(all_returns):.1f}')
    print(f'Mean Length: {np.mean(all_lengths):.0f}')
    print(f'Min Return: {np.min(all_returns):.1f}')
    print(f'Max Return: {np.max(all_returns):.1f}')
    print(f'Shield Activation Rate: {100*np.mean([a>0 for a in all_activations]):.0f}%')
    print(f'Mean Activations/Episode: {np.mean(all_activations):.1f}')
    
    # Success rate
    successes = sum(1 for r in all_returns if r > 100)
    print(f'Success Rate: {100*successes/len(all_returns):.0f}%')
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.bar(range(1, len(all_returns)+1), all_returns, color='#2ecc71')
    ax.axhline(np.mean(all_returns), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_returns):.1f}')
    ax.axhline(args.target_return, color='gray', linestyle=':', alpha=0.5,
               label=f'Target: {args.target_return}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Returns avec Shield')
    ax.legend()
    
    ax = axes[1]
    ax.bar(range(1, len(all_lengths)+1), all_lengths, color='#3498db')
    ax.axhline(np.mean(all_lengths), color='red', linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Longueur des Episodes')
    
    ax = axes[2]
    ax.bar(range(1, len(all_activations)+1), all_activations, color='#e74c3c')
    ax.axhline(np.mean(all_activations), color='black', linestyle='--',
               label=f'Mean: {np.mean(all_activations):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Activations')
    ax.set_title('Shield Activations')
    ax.legend()
    
    plt.suptitle(f'UDRL + Flow Shield (Command: H={command[0]}, R={command[1]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / 'shielded_evaluation_summary.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'\nFigure saved: {fig_path}')


if __name__ == '__main__':
    main()
