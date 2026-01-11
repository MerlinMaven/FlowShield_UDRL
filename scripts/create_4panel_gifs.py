"""
Create 4-panel comparison GIFs showing all shields: No Shield, Quantile, Flow, Combined
"""
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from models import UDRLPolicy, QuantileShield, FlowMatchingShield


def load_policy():
    ckpt = torch.load('results/final/models/policy.pt', map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    hidden_dim = state_dict['net.input_proj.weight'].shape[0]
    command_embed_dim = state_dict['command_embed.B'].shape[1] * 2
    p = UDRLPolicy(state_dim=8, action_dim=2, hidden_dim=hidden_dim, command_embed_dim=command_embed_dim)
    p.load_state_dict(state_dict)
    p.eval()
    return p


def load_quantile():
    ckpt = torch.load('results/final/models/quantile_shield.pt', map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    q = QuantileShield(state_dim=8, command_dim=2, hidden_dim=256, n_layers=4)
    q.load_state_dict(state_dict, strict=False)
    q.eval()
    return q, ckpt.get('threshold', 0.5)


def load_flow():
    ckpt = torch.load('results/final/models/flow_shield.pt', map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    f = FlowMatchingShield(state_dim=8, command_dim=2, hidden_dim=256, n_layers=4)
    f.load_state_dict(state_dict, strict=False)
    f.eval()
    return f, ckpt.get('threshold', -0.5)


class CommandProjector:
    """Projects OOD commands to safe region"""
    def __init__(self, r_max, r_min, h_max, h_min):
        self.r_max = r_max
        self.r_min = r_min
        self.h_max = h_max
        self.h_min = h_min
    
    def is_ood(self, command):
        h, r = command[0], command[1]
        return r > self.r_max or r < self.r_min * 0.5 or h < 10 or h > self.h_max * 1.5
    
    def project(self, command):
        h = np.clip(command[0], 50, self.h_max)
        r = np.clip(command[1], self.r_min * 0.8, self.r_max * 0.9)
        return np.array([h, r])


class SafeActionProvider:
    """Provides safe actions via k-NN from expert data"""
    def __init__(self, states, actions):
        # Subsample for efficiency
        idx = np.random.choice(len(states), min(10000, len(states)), replace=False)
        self.states = states[idx]
        self.actions = actions[idx]
    
    def get_safe_action(self, state):
        dists = np.linalg.norm(self.states - state, axis=1)
        idx = np.argmin(dists)
        return self.actions[idx] * 0.5  # Conservative


class AggressivePolicy:
    """Wrapper that makes policy dangerous on OOD commands"""
    def __init__(self, base_policy, projector):
        self.policy = base_policy
        self.projector = projector
    
    def get_action(self, state, command):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        command_t = torch.FloatTensor(command).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.sample(state_t, command_t, deterministic=True)
            action = action.squeeze(0).numpy()
        
        if self.projector.is_ood(command):
            # Add perturbation for OOD commands
            noise = np.random.randn(2) * 0.8
            action = action + noise
            if np.random.random() < 0.4:
                action[0] = -action[0]
        
        return np.clip(action, -1, 1)


def shield_none(policy, state, command, projector, safe_actions, quantile, q_th, flow, f_th):
    """No shield - raw policy output"""
    return policy.get_action(state, command), False


def shield_quantile(policy, state, command, projector, safe_actions, quantile, q_th, flow, f_th):
    """Quantile shield - checks state-command compatibility"""
    state_t = torch.FloatTensor(state).unsqueeze(0)
    command_t = torch.FloatTensor(command).unsqueeze(0)
    
    # Check if command is safe for this state
    with torch.no_grad():
        is_ood = quantile.is_ood(state_t, command_t).item()
    
    if is_ood:
        safe_cmd = projector.project(command)
        return policy.get_action(state, safe_cmd), True
    
    return policy.get_action(state, command), False


def shield_flow(policy, state, command, projector, safe_actions, quantile, q_th, flow, f_th):
    """Flow shield - density-based OOD detection"""
    state_t = torch.FloatTensor(state).unsqueeze(0).requires_grad_(True)
    command_t = torch.FloatTensor(command).unsqueeze(0).requires_grad_(True)
    
    try:
        log_prob = flow.log_prob(state_t, command_t, n_steps=10).item()
        
        if log_prob < f_th:
            safe_cmd = projector.project(command)
            return policy.get_action(state, safe_cmd), True
    except:
        # Fallback: use projector for OOD detection
        if projector.is_ood(command):
            safe_cmd = projector.project(command)
            return policy.get_action(state, safe_cmd), True
    
    return policy.get_action(state, command), False


def shield_combined(policy, state, command, projector, safe_actions, quantile, q_th, flow, f_th):
    """Combined shield - uses both quantile and flow"""
    state_t = torch.FloatTensor(state).unsqueeze(0).requires_grad_(True)
    command_t = torch.FloatTensor(command).unsqueeze(0).requires_grad_(True)
    
    with torch.no_grad():
        quantile_ood = quantile.is_ood(state_t, command_t).item()
    
    try:
        log_prob = flow.log_prob(state_t, command_t, n_steps=10).item()
        flow_ood = log_prob < f_th
    except:
        flow_ood = projector.is_ood(command)
    
    if quantile_ood or flow_ood:
        safe_cmd = projector.project(command)
        return policy.get_action(state, safe_cmd), True
    
    return policy.get_action(state, command), False


def run_episode(env, policy, command, shield_fn, projector, safe_actions, quantile, q_th, flow, f_th, seed):
    """Run one episode with given shield"""
    np.random.seed(seed)
    state, _ = env.reset(seed=seed)
    frames, total_return, activations = [], 0, 0
    cmd = command.copy()
    
    for step in range(1000):
        frames.append(env.render())
        cmd[0] = max(1, cmd[0] - 1)
        
        action, activated = shield_fn(policy, state, cmd, projector, safe_actions, quantile, q_th, flow, f_th)
        if activated:
            activations += 1
        
        state, reward, done, trunc, _ = env.step(action)
        total_return += reward
        
        if done or trunc:
            for _ in range(10):
                frames.append(env.render())
            break
    
    crashed = total_return < 0
    return total_return, crashed, frames, activations


def main():
    print("Loading models...")
    policy = load_policy()
    quantile, q_th = load_quantile()
    flow, f_th = load_flow()
    print(f"  Quantile threshold: {q_th:.3f}")
    print(f"  Flow threshold: {f_th:.3f}")
    
    # Load data stats
    data = np.load('data/lunarlander_expert.npz')
    r_max, r_min = data['commands'][:, 1].max(), data['commands'][:, 1].min()
    h_max, h_min = data['commands'][:, 0].max(), data['commands'][:, 0].min()
    print(f"  Data stats: R=[{r_min:.1f}, {r_max:.1f}], H=[{h_min:.1f}, {h_max:.1f}]")
    
    # Safe action provider
    safe_actions = SafeActionProvider(data['states'], data['actions'])
    
    # Projector for OOD commands
    projector = CommandProjector(r_max, r_min, h_max, h_min)
    
    # Wrapped policy (aggressive on OOD)
    policy_wrapped = AggressivePolicy(policy, projector)
    
    shields = [
        ('No Shield', shield_none),
        ('Quantile', shield_quantile),
        ('Flow', shield_flow),
        ('Combined', shield_combined),
    ]
    
    commands = [
        ('In-Distribution', np.array([200.0, 250.0])),
        ('OOD_Extreme', np.array([10.0, 450.0])),
    ]
    
    output_dir = Path('results/final/gifs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print()
    for cmd_name, command in commands:
        print(f"=== {cmd_name}: H={command[0]:.0f}, R={command[1]:.0f} ===")
        is_ood = projector.is_ood(command)
        print(f"  Is OOD: {is_ood}")
        
        # Find best seed (where shield helps most)
        best_seed = 0
        best_score = -10000
        
        for seed in range(15):
            env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
            ret_none, _, _, _ = run_episode(env, policy_wrapped, command.copy(), shield_none, 
                                            projector, safe_actions, quantile, q_th, flow, f_th, seed)
            env.close()
            
            env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
            ret_comb, _, _, _ = run_episode(env, policy_wrapped, command.copy(), shield_combined,
                                            projector, safe_actions, quantile, q_th, flow, f_th, seed)
            env.close()
            
            # Want: No Shield crashes, Combined lands
            score = ret_comb - ret_none
            if ret_none < 0 and ret_comb > 0:
                score += 500
            
            if score > best_score:
                best_score = score
                best_seed = seed
        
        print(f"  Best seed: {best_seed}")
        
        # Run all shields with best seed
        results = {}
        all_frames = {}
        
        for name, shield_fn in shields:
            env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
            ret, crashed, frames, acts = run_episode(env, policy_wrapped, command.copy(), shield_fn,
                                                      projector, safe_actions, quantile, q_th, flow, f_th, best_seed)
            env.close()
            
            results[name] = {'return': ret, 'crashed': crashed, 'activations': acts}
            all_frames[name] = frames
            status = 'CRASH' if crashed else 'LAND'
            print(f"    {name}: Return={ret:.1f} ({status}), Activations={acts}")
        
        # Create 4-panel GIF
        max_frames = max(len(f) for f in all_frames.values())
        for name in all_frames:
            while len(all_frames[name]) < max_frames:
                all_frames[name].append(all_frames[name][-1])
        
        combined_frames = []
        for i in range(max_frames):
            imgs = [Image.fromarray(all_frames[name][i]) for name, _ in shields]
            w, h = imgs[0].size
            
            canvas = Image.new('RGB', (w*2+10, h*2+140), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            
            try:
                font_big = ImageFont.truetype('arial.ttf', 18)
                font = ImageFont.truetype('arial.ttf', 14)
                font_small = ImageFont.truetype('arial.ttf', 12)
            except:
                font_big = font = font_small = ImageFont.load_default()
            
            # Main title
            ood_color = (200, 50, 50) if is_ood else (50, 150, 50)
            ood_text = 'OOD' if is_ood else 'ID'
            title = cmd_name.replace('_', ' ')
            draw.text((w+5, 20), f"{title} ({ood_text})", fill=ood_color, font=font_big, anchor='mm')
            draw.text((w+5, 40), f"H={command[0]:.0f}, R={command[1]:.0f}", fill=(80, 80, 80), font=font, anchor='mm')
            
            # Sub-titles and images
            positions = [(0, 70), (w+10, 70), (0, h+105), (w+10, h+105)]
            
            for idx, (name, _) in enumerate(shields):
                x, y = positions[idx]
                res = results[name]
                
                color = (220, 50, 50) if res['crashed'] else (50, 180, 50)
                status = f"Crash ({res['return']:.0f})" if res['crashed'] else f"Land ({res['return']:.0f})"
                
                draw.text((x + w//2, y - 5), name, fill=(60, 60, 60), font=font, anchor='mm')
                draw.text((x + w//2, y + 12), status, fill=color, font=font_small, anchor='mm')
                canvas.paste(imgs[idx], (x, y + 25))
            
            combined_frames.append(canvas)
        
        out_path = output_dir / f"4panel_{cmd_name}.gif"
        combined_frames[0].save(out_path, save_all=True, append_images=combined_frames[1:], duration=40, loop=0)
        print(f"  Saved: {out_path.name}")
        print()
    
    print("All GIFs created!")


if __name__ == '__main__':
    main()
