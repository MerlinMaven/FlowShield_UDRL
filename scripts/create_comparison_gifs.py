"""
Create 4-panel comparison GIFs for FlowShield-UDRL.
Compares: No Shield, Quantile, Diffusion, Flow
For both in-distribution and OOD commands.

Uses legacy model architecture to match saved checkpoints.
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# LEGACY MODEL ARCHITECTURES (matching saved checkpoints)
# ============================================================================

class LegacyUDRLPolicy(nn.Module):
    """UDRL Policy matching the saved model architecture"""
    def __init__(self, state_dim=8, action_dim=2, command_dim=2, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(command_dim, 32))
        
        input_dim = state_dim + 64  # state + sin/cos embeddings
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )
        self.action_dim = action_dim
        
    def forward(self, state, command):
        cmd_proj = command @ self.B
        cmd_embed = torch.cat([torch.sin(cmd_proj), torch.cos(cmd_proj)], dim=-1)
        x = torch.cat([state, cmd_embed], dim=-1)
        out = self.net(x)
        mean = torch.tanh(out[:, :self.action_dim])
        return mean
    
    @classmethod
    def load_from_checkpoint(cls, path, device):
        model = cls().to(device)
        cp = torch.load(path, map_location=device, weights_only=False)
        model.B = cp['command_embed.B'].to(device)
        state_dict = {k.replace('net.net.', ''): v for k, v in cp.items() if 'net.net' in k}
        model.net.load_state_dict(state_dict)
        return model


class LegacyQuantileShield(nn.Module):
    """Quantile Shield matching saved architecture"""
    def __init__(self, state_dim=8, command_dim=2, hidden_dim=256, tau=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + command_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.tau = tau
        self.return_max = 300.0
        
    def forward(self, state, command):
        x = torch.cat([state, command], dim=-1)
        return self.net(x)
    
    def project(self, state, command):
        """Project command to safe region"""
        with torch.no_grad():
            q = self.forward(state, command)
            projected = command.clone()
            projected[:, 1] = torch.clamp(projected[:, 1], max=q.squeeze(-1) + 50)
        return projected
    
    @classmethod
    def load_from_checkpoint(cls, path, device):
        model = cls().to(device)
        cp = torch.load(path, map_location=device, weights_only=False)
        state_dict = {k.replace('net.net.', ''): v for k, v in cp.items() if 'net.net' in k}
        model.net.load_state_dict(state_dict)
        return model


class LegacyFlowShield(nn.Module):
    """Flow Matching Shield matching saved architecture"""
    def __init__(self, state_dim=8, command_dim=2, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(1, 32))
        input_dim = state_dim + command_dim + 64
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, command_dim)
        )
        self.command_dim = command_dim
        self.command_mean = torch.zeros(command_dim)
        self.command_std = torch.ones(command_dim)
        
    def forward(self, state, command, t):
        t_proj = t.view(-1, 1) @ self.B
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        x = torch.cat([state, command, t_embed], dim=-1)
        return self.net(x)
    
    def sample(self, state, n_steps=20):
        """Sample from the flow"""
        batch_size = state.shape[0]
        device = state.device
        
        x = torch.randn(batch_size, self.command_dim, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.ones(batch_size, device=device) * (i / n_steps)
            v = self.forward(state, x, t)
            x = x + v * dt
        
        x = x * self.command_std.to(device) + self.command_mean.to(device)
        return x
    
    def project(self, state, command):
        """Project OOD command to distribution"""
        with torch.no_grad():
            sampled = self.sample(state, n_steps=10)
            mask = command[:, 1] > 280
            projected = command.clone()
            if mask.any():
                projected[mask] = sampled[mask]
        return projected
    
    @classmethod
    def load_from_checkpoint(cls, path, device, data_path='../data/lunarlander_expert.npz'):
        model = cls().to(device)
        cp = torch.load(path, map_location=device, weights_only=False)
        state_dict = {k.replace('net.net.', ''): v for k, v in cp.items() if 'net.net' in k}
        model.net.load_state_dict(state_dict)
        
        try:
            data = np.load(data_path)
            returns = data['episode_returns']
            model.command_mean = torch.tensor([150.0, float(np.mean(returns))], dtype=torch.float32)
            model.command_std = torch.tensor([100.0, float(np.std(returns)) + 1e-6], dtype=torch.float32)
        except:
            model.command_mean = torch.tensor([150.0, 240.0], dtype=torch.float32)
            model.command_std = torch.tensor([100.0, 30.0], dtype=torch.float32)
        
        return model


class LegacyDiffusionShield(nn.Module):
    """Diffusion Shield matching saved architecture"""
    def __init__(self, state_dim=8, command_dim=2, hidden_dim=256, n_timesteps=100):
        super().__init__()
        self.register_buffer('B', torch.randn(1, 32))
        input_dim = state_dim + command_dim + 64
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, command_dim)
        )
        self.command_dim = command_dim
        self.n_timesteps = n_timesteps
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        self.command_mean = torch.zeros(command_dim)
        self.command_std = torch.ones(command_dim)
        
    def forward(self, state, command, t):
        t_proj = t.view(-1, 1).float() @ self.B
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        x = torch.cat([state, command, t_embed], dim=-1)
        return self.net(x)
    
    def sample(self, state, n_steps=20):
        """Sample using DDPM"""
        batch_size = state.shape[0]
        device = state.device
        
        x = torch.randn(batch_size, self.command_dim, device=device)
        
        step_size = max(1, self.n_timesteps // n_steps)
        timesteps = list(range(self.n_timesteps - 1, -1, -step_size))
        
        for t_idx in timesteps:
            t = torch.ones(batch_size, device=device, dtype=torch.long) * t_idx
            
            noise_pred = self.forward(state, x, t)
            
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(self.betas[t_idx]) * noise
        
        x = x * self.command_std.to(device) + self.command_mean.to(device)
        return x
    
    def project(self, state, command):
        """Project OOD command"""
        with torch.no_grad():
            sampled = self.sample(state, n_steps=10)
            mask = command[:, 1] > 280
            projected = command.clone()
            if mask.any():
                projected[mask] = sampled[mask]
        return projected
    
    @classmethod
    def load_from_checkpoint(cls, path, device, data_path='../data/lunarlander_expert.npz'):
        cp = torch.load(path, map_location=device, weights_only=False)
        n_timesteps = len(cp['betas'])
        
        model = cls(n_timesteps=n_timesteps).to(device)
        state_dict = {k.replace('net.net.', ''): v for k, v in cp.items() if 'net.net' in k}
        model.net.load_state_dict(state_dict)
        
        model.betas = cp['betas'].to(device)
        model.alphas = cp['alphas'].to(device)
        model.alphas_cumprod = cp['alphas_cumprod'].to(device)
        
        try:
            data = np.load(data_path)
            returns = data['episode_returns']
            model.command_mean = torch.tensor([150.0, float(np.mean(returns))], dtype=torch.float32)
            model.command_std = torch.tensor([100.0, float(np.std(returns)) + 1e-6], dtype=torch.float32)
        except:
            model.command_mean = torch.tensor([150.0, 240.0], dtype=torch.float32)
            model.command_std = torch.tensor([100.0, 30.0], dtype=torch.float32)
        
        return model


# ============================================================================
# EPISODE RUNNING AND GIF CREATION
# ============================================================================

def run_episode(env, policy, shield, command, device, max_steps=500):
    """Run one episode and collect frames"""
    frames = []
    state, _ = env.reset(seed=42)
    total_reward = 0
    horizon = command[0]
    
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        cmd = [horizon - step, command[1] - total_reward]
        cmd_t = torch.FloatTensor(cmd).unsqueeze(0).to(device)
        
        if shield is not None:
            with torch.no_grad():
                cmd_t = shield.project(state_t, cmd_t)
        
        with torch.no_grad():
            action = policy(state_t, cmd_t).cpu().numpy()[0]
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if terminated or truncated:
            for _ in range(10):
                frames.append(env.render())
            break
    
    return frames, total_reward


def create_comparison_gif(policy, shields, command, filename, title_prefix, device):
    """Create 4-panel comparison GIF"""
    print(f'Creating {filename}...')
    
    names = ['No Shield', 'Quantile Shield', 'Diffusion Shield', 'Flow Shield']
    envs = [gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array') for _ in range(4)]
    
    all_frames = []
    all_rewards = []
    for i, (env, shield, name) in enumerate(zip(envs, shields, names)):
        print(f'  Running {name}...')
        frames, reward = run_episode(env, policy, shield, command, device)
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
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.25)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
    
    ims = []
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']  # Red, Blue, Purple, Green
    for i, ax in enumerate(axes):
        im = ax.imshow(all_frames[i][0])
        ax.set_title(f'{names[i]}\nReturn: {all_rewards[i]:.1f}', 
                     fontsize=12, fontweight='bold', color=colors[i])
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)
        ims.append(im)
    
    fig.suptitle(f'{title_prefix}\nCommand: H={command[0]}, R={command[1]}', 
                 fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_array(all_frames[i][frame_idx])
        return ims
    
    ani = animation.FuncAnimation(fig, update, frames=min(max_len, 300), interval=50, blit=True)
    
    save_path = f'../results/lunarlander/figures/{filename}'
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f'  Saved: {save_path}')
    
    return all_rewards


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    print('Loading models...')
    
    policy = LegacyUDRLPolicy.load_from_checkpoint(
        '../results/lunarlander/models/policy.pt', device)
    policy.eval()
    print('  Policy loaded')
    
    quantile = LegacyQuantileShield.load_from_checkpoint(
        '../results/lunarlander/models/quantile_shield.pt', device)
    quantile.eval()
    print('  Quantile shield loaded')
    
    flow = LegacyFlowShield.load_from_checkpoint(
        '../results/lunarlander/models/flow_shield.pt', device)
    flow.eval()
    print('  Flow shield loaded')
    
    diffusion = LegacyDiffusionShield.load_from_checkpoint(
        '../results/lunarlander/models/diffusion_shield.pt', device)
    diffusion.eval()
    print('  Diffusion shield loaded')
    
    shields = [None, quantile, diffusion, flow]
    print('All models loaded!')
    
    # Create GIFs
    print('\n' + '='*60)
    print('IN-DISTRIBUTION COMPARISON (R=220)')
    print('='*60)
    id_rewards = create_comparison_gif(
        policy, shields, (200, 220), 
        'comparison_in_distribution.gif', 
        'In-Distribution Command', device)
    
    print('\n' + '='*60)
    print('OOD COMPARISON (R=350)')
    print('='*60)
    ood_rewards = create_comparison_gif(
        policy, shields, (50, 350), 
        'comparison_ood.gif', 
        'Out-of-Distribution Command', device)
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    names = ['No Shield', 'Quantile', 'Diffusion', 'Flow']
    print(f'{"Method":<15} | {"ID (R=220)":>12} | {"OOD (R=350)":>12}')
    print('-'*45)
    for i, name in enumerate(names):
        print(f'{name:<15} | {id_rewards[i]:>12.1f} | {ood_rewards[i]:>12.1f}')
    
    print('\nGIFs saved to results/lunarlander/figures/')


if __name__ == '__main__':
    main()
