"""
Visualise des episodes avec l'agent UDRL entraine.

Affiche le rendu graphique et sauvegarde les GIFs.
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import argparse

from models import UDRLPolicy


def load_policy(model_path, state_dim=8, action_dim=2, device='cpu'):
    """Charge la politique UDRL."""
    # Determine architecture from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer command_embed_dim from B shape: [command_dim=2, embed_dim//2]
    b_shape = state_dict['command_embed.B'].shape
    command_dim = b_shape[0]  # Should be 2 (horizon, return)
    command_embed_dim = b_shape[1] * 2  # embed_dim = B.shape[1] * 2
    
    # Infer hidden_dim from net.input_proj.weight shape
    hidden_dim = state_dict['net.input_proj.weight'].shape[0]
    
    print(f'  Architecture: hidden_dim={hidden_dim}, command_dim={command_dim}, command_embed={command_embed_dim}')
    
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        command_dim=command_dim,  # 2 for (horizon, return)
        hidden_dim=hidden_dim,
        command_embed_dim=command_embed_dim,
    )
    
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def run_episode(env, policy, command, device='cpu', max_steps=1000):
    """Execute un episode et collecte les frames."""
    state, info = env.reset()
    
    frames = [env.render()]
    rewards = []
    total_return = 0
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    command_t = torch.FloatTensor(command).unsqueeze(0).to(device)
    current_command = command_t.clone()
    
    for step in range(max_steps):
        with torch.no_grad():
            action = policy.sample(state_t, current_command, deterministic=True)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_return += reward
        rewards.append(reward)
        frames.append(env.render())
        
        if terminated or truncated:
            break
        
        state = next_state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Update command
        current_command[0, 0] = max(1, current_command[0, 0] - 1)
        current_command[0, 1] = current_command[0, 1] - reward
    
    return frames, total_return, len(rewards)


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
    parser = argparse.ArgumentParser(description='Visualise UDRL episodes')
    parser.add_argument('--policy', type=str, 
                       default='results/unknown/models/policy.pt',
                       help='Path to policy checkpoint')
    parser.add_argument('--n-episodes', type=int, default=3)
    parser.add_argument('--horizon', type=float, default=200)
    parser.add_argument('--target-return', type=float, default=220)
    parser.add_argument('--output-dir', type=str, 
                       default='results/lunarlander/figures')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    print('='*60)
    print('VISUALISATION EPISODES UDRL')
    print('='*60)
    
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    print(f'\nChargement policy: {args.policy}')
    policy = load_policy(args.policy, device=device)
    print('  Policy chargee!')
    
    # Create environment
    print('\nCreation environnement LunarLander-v3...')
    env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    
    command = [args.horizon, args.target_return]
    print(f'Command: H={command[0]}, R={command[1]}')
    
    # Run episodes
    all_returns = []
    all_lengths = []
    
    for ep in range(args.n_episodes):
        print(f'\nEpisode {ep+1}/{args.n_episodes}...')
        
        frames, total_return, length = run_episode(
            env, policy, command, device=device
        )
        
        all_returns.append(total_return)
        all_lengths.append(length)
        
        print(f'  Return: {total_return:.1f}')
        print(f'  Length: {length} steps')
        print(f'  Frames: {len(frames)}')
        
        # Save GIF
        gif_path = output_dir / f'udrl_episode_{ep+1}.gif'
        save_gif(frames, gif_path, fps=30)
        print(f'  Saved: {gif_path}')
    
    env.close()
    
    # Summary
    print('\n' + '='*60)
    print('RESUME')
    print('='*60)
    print(f'Episodes: {args.n_episodes}')
    print(f'Mean Return: {np.mean(all_returns):.1f} +/- {np.std(all_returns):.1f}')
    print(f'Mean Length: {np.mean(all_lengths):.0f}')
    print(f'Min Return: {np.min(all_returns):.1f}')
    print(f'Max Return: {np.max(all_returns):.1f}')
    
    # Create summary figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.bar(range(1, len(all_returns)+1), all_returns, color='#3498db')
    ax.axhline(np.mean(all_returns), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_returns):.1f}')
    ax.axhline(args.target_return, color='green', linestyle=':', 
               label=f'Target: {args.target_return}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Returns par Episode')
    ax.legend()
    
    ax = axes[1]
    ax.bar(range(1, len(all_lengths)+1), all_lengths, color='#9b59b6')
    ax.axhline(np.mean(all_lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_lengths):.0f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Longueur par Episode')
    ax.legend()
    
    plt.suptitle(f'UDRL Agent Performance (Command: H={command[0]}, R={command[1]})')
    plt.tight_layout()
    
    fig_path = output_dir / 'udrl_evaluation_summary.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'\nFigure saved: {fig_path}')


if __name__ == '__main__':
    main()
