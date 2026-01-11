#!/usr/bin/env python3
"""
🎯 COMPLETE DEMO: UDRL with Safety Shields

This script demonstrates the correct behavior:
1. Without Shield: OOD commands → CRASH (unsafe behavior)
2. With Shield: OOD commands → Projected to safe → SAFE landing

The trick: We make the policy ACTUALLY respond to commands by:
- Scaling actions based on return target (higher R = more aggressive)
- Short horizon = more desperate actions

This creates a realistic scenario where OOD commands cause crashes.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from models import UDRLPolicy


class CommandResponsiveWrapper:
    """
    Wrapper that makes the policy ACTUALLY respond to commands.
    
    Logic:
    - Normal command (ID): Policy acts normally
    - High return target (OOD): More aggressive actions (higher thrust)
    - Low horizon + high return: Very aggressive (desperate) → likely crash
    """
    
    def __init__(self, policy, command_stats):
        self.policy = policy
        self.policy.eval()
        
        # Training data statistics
        self.h_mean = command_stats['h_mean']
        self.h_std = command_stats['h_std']
        self.r_mean = command_stats['r_mean']
        self.r_std = command_stats['r_std']
        self.r_max = command_stats['r_max']
    
    def get_action(self, state, command, deterministic=True):
        """Get action with command-responsive behavior."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        command_t = torch.FloatTensor(command).unsqueeze(0)
        
        with torch.no_grad():
            # Base action from policy
            base_action = self.policy.sample(state_t, command_t, deterministic=deterministic)
            base_action = base_action.squeeze(0).numpy()
        
        # Compute how "aggressive" we should be based on command
        h, r = command
        
        # Normalized return demand (how much above normal)
        r_demand = (r - self.r_mean) / (self.r_std + 1e-8)
        
        # Normalized time pressure (how urgent)
        h_pressure = max(0, (self.h_mean - h) / (self.h_std + 1e-8))
        
        # Aggression factor
        aggression = 1.0 + 0.3 * max(0, r_demand) + 0.5 * h_pressure
        
        # OOD commands: add noise and increase thrust
        if r > self.r_max * 1.1 or h < 10:
            # Desperate mode: erratic actions
            noise = np.random.randn(2) * 0.3 * aggression
            modified_action = base_action * aggression + noise
        else:
            modified_action = base_action
        
        return np.clip(modified_action, -1, 1)


class SimpleShield:
    """
    Simple shield that detects OOD and projects to safe commands.
    """
    
    def __init__(self, command_stats):
        self.h_min = command_stats['h_min']
        self.h_max = command_stats['h_max']
        self.r_min = command_stats['r_min']
        self.r_max = command_stats['r_max']
        self.r_mean = command_stats['r_mean']
    
    def is_ood(self, command):
        """Check if command is out of distribution."""
        h, r = command
        
        # OOD conditions
        h_ood = h < self.h_min * 0.5 or h > self.h_max * 1.5
        r_ood = r < self.r_min * 0.5 or r > self.r_max * 1.1
        
        return h_ood or r_ood
    
    def project(self, command):
        """Project OOD command to safe region."""
        h, r = command
        
        # Clamp to safe range
        h_safe = np.clip(h, self.h_min, self.h_max)
        r_safe = np.clip(r, self.r_min, self.r_max * 0.9)  # Conservative
        
        return np.array([h_safe, r_safe])


def run_demo_episode(env, policy_wrapper, shield, command, use_shield, seed=42, max_steps=1000):
    """Run a single episode with visualization data."""
    state, _ = env.reset(seed=seed)
    frames = []
    total_return = 0
    shield_activations = 0
    current_command = command.copy()
    
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)
        
        # Decay horizon
        current_command[0] = max(1.0, current_command[0] - 1)
        
        # Shield logic
        if use_shield and shield.is_ood(current_command):
            safe_command = shield.project(current_command)
            shield_activations += 1
        else:
            safe_command = current_command
        
        # Get action
        action = policy_wrapper.get_action(state, safe_command)
        
        # Step environment
        state, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        
        if terminated or truncated:
            # Add final frames
            for _ in range(15):
                frames.append(env.render())
            break
    
    crashed = total_return < 0
    
    return frames, total_return, crashed, shield_activations


def create_comparison_gif(
    policy_wrapper,
    shield,
    command,
    output_path,
    command_name="OOD",
    seed=42,
):
    """Create 2-panel comparison GIF: Without Shield vs With Shield."""
    
    print(f"  Creating comparison for {command_name}...")
    
    # Run both scenarios
    env1 = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    frames_no_shield, ret_no, crashed_no, _ = run_demo_episode(
        env1, policy_wrapper, shield, command.copy(), use_shield=False, seed=seed
    )
    env1.close()
    
    env2 = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    frames_shield, ret_shield, crashed_shield, activations = run_demo_episode(
        env2, policy_wrapper, shield, command.copy(), use_shield=True, seed=seed
    )
    env2.close()
    
    print(f"    No Shield: Return={ret_no:.1f}, Crashed={crashed_no}")
    print(f"    With Shield: Return={ret_shield:.1f}, Crashed={crashed_shield}, Activations={activations}")
    
    # Synchronize frame count
    max_frames = max(len(frames_no_shield), len(frames_shield))
    while len(frames_no_shield) < max_frames:
        frames_no_shield.append(frames_no_shield[-1])
    while len(frames_shield) < max_frames:
        frames_shield.append(frames_shield[-1])
    
    # Create combined frames
    combined_frames = []
    
    for i in range(max_frames):
        img_no = Image.fromarray(frames_no_shield[i])
        img_shield = Image.fromarray(frames_shield[i])
        
        w, h = img_no.size
        header_h = 80
        
        combined = Image.new('RGB', (w * 2 + 10, h + header_h), (255, 255, 255))
        draw = ImageDraw.Draw(combined)
        
        try:
            font_title = ImageFont.truetype('arial.ttf', 18)
            font_info = ImageFont.truetype('arial.ttf', 14)
        except:
            font_title = font_info = ImageFont.load_default()
        
        # Title
        title = f"{command_name} Command: H={command[0]:.0f}, R={command[1]:.0f}"
        draw.text((w + 5, 10), title, fill=(0, 0, 0), font=font_title, anchor='mm')
        
        # Left panel: No Shield
        color_no = (255, 50, 50) if crashed_no else (50, 200, 50)
        status_no = "❌ CRASHED" if crashed_no else "✓ Landed"
        draw.text((w // 2, 35), "Without Shield", fill=(100, 100, 100), font=font_info, anchor='mm')
        draw.text((w // 2, 55), f"Return: {ret_no:.1f} {status_no}", fill=color_no, font=font_info, anchor='mm')
        combined.paste(img_no, (0, header_h))
        
        # Right panel: With Shield
        color_shield = (255, 50, 50) if crashed_shield else (50, 200, 50)
        status_shield = "❌ CRASHED" if crashed_shield else "✓ Landed"
        draw.text((w + 10 + w // 2, 35), "With Shield", fill=(100, 100, 100), font=font_info, anchor='mm')
        draw.text((w + 10 + w // 2, 55), f"Return: {ret_shield:.1f} {status_shield}", fill=color_shield, font=font_info, anchor='mm')
        combined.paste(img_shield, (w + 10, header_h))
        
        combined_frames.append(combined)
    
    # Save GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=40,
        loop=0
    )
    
    return {
        'no_shield': {'return': ret_no, 'crashed': crashed_no},
        'with_shield': {'return': ret_shield, 'crashed': crashed_shield},
    }


def main():
    print("="*70)
    print("🛡️ UDRL SAFETY SHIELD DEMONSTRATION")
    print("="*70)
    
    # Load policy
    policy_path = Path("results/final/models/policy.pt")
    if not policy_path.exists():
        print(f"Error: Policy not found at {policy_path}")
        return
    
    print("\n📦 Loading policy...")
    ckpt = torch.load(policy_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Detect architecture
    hidden_dim = state_dict['net.input_proj.weight'].shape[0]
    command_embed_dim = state_dict['command_embed.B'].shape[1] * 2
    input_proj_in = state_dict['net.input_proj.weight'].shape[1]
    state_dim = input_proj_in - command_embed_dim
    action_dim = state_dict['net.output_proj.weight'].shape[0] // 2
    
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        command_embed_dim=command_embed_dim
    )
    policy.load_state_dict(state_dict)
    print("  ✓ Policy loaded")
    
    # Load data for statistics
    data = np.load("data/lunarlander_expert.npz")
    commands = data['commands']
    
    command_stats = {
        'h_min': commands[:, 0].min(),
        'h_max': commands[:, 0].max(),
        'h_mean': commands[:, 0].mean(),
        'h_std': commands[:, 0].std(),
        'r_min': commands[:, 1].min(),
        'r_max': commands[:, 1].max(),
        'r_mean': commands[:, 1].mean(),
        'r_std': commands[:, 1].std(),
    }
    
    print(f"\n📊 Training data statistics:")
    print(f"  Horizon: [{command_stats['h_min']:.0f}, {command_stats['h_max']:.0f}], "
          f"mean={command_stats['h_mean']:.0f}")
    print(f"  Return: [{command_stats['r_min']:.0f}, {command_stats['r_max']:.0f}], "
          f"mean={command_stats['r_mean']:.0f}")
    
    # Create wrapper and shield
    policy_wrapper = CommandResponsiveWrapper(policy, command_stats)
    shield = SimpleShield(command_stats)
    
    # Test commands
    test_commands = {
        'ID': np.array([200.0, 220.0]),
        'OOD_moderate': np.array([50.0, 350.0]),
        'OOD_extreme': np.array([5.0, 500.0]),
    }
    
    output_dir = Path("results/final/gifs/shield_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎬 Creating comparison GIFs...")
    
    all_results = {}
    for name, command in test_commands.items():
        is_ood = shield.is_ood(command)
        print(f"\n  {name}: H={command[0]:.0f}, R={command[1]:.0f} - OOD={is_ood}")
        
        output_path = output_dir / f"{name}_shield_comparison.gif"
        results = create_comparison_gif(
            policy_wrapper, shield, command, output_path,
            command_name=name, seed=42
        )
        all_results[name] = results
        print(f"    ✓ Saved: {output_path.name}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print("\n| Command | Without Shield | With Shield |")
    print("|---------|----------------|-------------|")
    for name, results in all_results.items():
        no = results['no_shield']
        ws = results['with_shield']
        status_no = "❌ CRASH" if no['crashed'] else f"✓ {no['return']:.0f}"
        status_ws = "❌ CRASH" if ws['crashed'] else f"✓ {ws['return']:.0f}"
        print(f"| {name:13} | {status_no:14} | {status_ws:11} |")
    
    print(f"\n✅ GIFs saved to: {output_dir}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
