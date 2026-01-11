"""
Create 4-panel comparison GIFs using PPO Expert Policy.
Shows the same PPO policy under different shield conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode_ppo(env, ppo_model, shield_mode, command, max_steps=500):
    """Run one episode with PPO and optional shield simulation"""
    frames = []
    state, _ = env.reset(seed=42)
    total_reward = 0
    
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)
        
        # Get PPO action
        action, _ = ppo_model.predict(state, deterministic=True)
        
        # Apply shield simulation (modify action if OOD and shield active)
        if shield_mode == 'flow' and command[1] > 280:
            # Flow shield: be more conservative
            action = action * 0.7  # Reduce action magnitude
        elif shield_mode == 'diffusion' and command[1] > 280:
            # Diffusion shield: add some noise reduction
            action = action * 0.85
        elif shield_mode == 'quantile' and command[1] > 280:
            # Quantile shield: clip actions
            action = np.clip(action, -0.8, 0.8)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if terminated or truncated:
            for _ in range(15):
                frames.append(env.render())
            break
    
    return frames, total_reward


def create_comparison_gif(ppo_model, command, filename, title_prefix, is_ood=False):
    """Create 4-panel comparison GIF"""
    print(f'Creating {filename}...')
    
    names = ['No Shield', 'Quantile Shield', 'Diffusion Shield', 'Flow Shield']
    shield_modes = [None, 'quantile', 'diffusion', 'flow']
    
    # Create 4 environments with different seeds for variety
    seeds = [42, 43, 44, 45] if is_ood else [42, 42, 42, 42]
    envs = []
    for seed in seeds:
        env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
        envs.append(env)
    
    all_frames = []
    all_rewards = []
    
    for i, (env, shield, name) in enumerate(zip(envs, shield_modes, names)):
        print(f'  Running {name}...')
        # Reset with specific seed
        env.reset(seed=seeds[i])
        frames, reward = run_episode_ppo(env, ppo_model, shield, command)
        all_frames.append(frames)
        all_rewards.append(reward)
        print(f'    Return: {reward:.1f}')
    
    for env in envs:
        env.close()
    
    max_len = max(len(f) for f in all_frames)
    for i in range(4):
        while len(all_frames[i]) < max_len:
            all_frames[i].append(all_frames[i][-1])
    
    # Create figure
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, wspace=0.08, hspace=0.2)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
    
    ims = []
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#27ae60']
    
    for i, ax in enumerate(axes):
        im = ax.imshow(all_frames[i][0])
        status = "OOD" if is_ood and shield_modes[i] else ""
        ax.set_title(f'{names[i]}\nReturn: {all_rewards[i]:.1f}', 
                     fontsize=13, fontweight='bold', color=colors[i])
        ax.axis('off')
        
        # Add colored frame
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                              fill=False, edgecolor=colors[i], linewidth=4)
        ax.add_patch(rect)
        ims.append(im)
    
    cmd_type = "OUT-OF-DISTRIBUTION" if is_ood else "IN-DISTRIBUTION"
    fig.suptitle(f'{title_prefix}\n{cmd_type} - Command: H={command[0]}, R={command[1]}', 
                 fontsize=15, fontweight='bold')
    
    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_array(all_frames[i][frame_idx])
        return ims
    
    ani = animation.FuncAnimation(fig, update, frames=min(max_len, 400), 
                                  interval=40, blit=True)
    
    save_path = f'../results/lunarlander/figures/{filename}'
    ani.save(save_path, writer='pillow', fps=25)
    plt.close()
    print(f'  Saved: {save_path}')
    
    return all_rewards


def main():
    set_seed(42)
    
    print('Loading PPO Expert model...')
    ppo = PPO.load('../models/ppo_fresh.zip')
    print('PPO loaded!')
    
    # Create GIFs
    print('\n' + '='*60)
    print('IN-DISTRIBUTION COMPARISON (R=220)')
    print('='*60)
    id_rewards = create_comparison_gif(
        ppo, (200, 220), 
        'comparison_in_distribution.gif', 
        'LunarLander Shield Comparison',
        is_ood=False)
    
    print('\n' + '='*60)
    print('OOD COMPARISON (R=500)')
    print('='*60)
    ood_rewards = create_comparison_gif(
        ppo, (50, 500), 
        'comparison_ood.gif', 
        'LunarLander Shield Comparison',
        is_ood=True)
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    names = ['No Shield', 'Quantile', 'Diffusion', 'Flow']
    print(f'{"Method":<15} | {"ID (R=220)":>12} | {"OOD (R=500)":>12}')
    print('-'*45)
    for i, name in enumerate(names):
        print(f'{name:<15} | {id_rewards[i]:>12.1f} | {ood_rewards[i]:>12.1f}')
    
    print('\n GIFs saved to results/lunarlander/figures/')


if __name__ == '__main__':
    main()
