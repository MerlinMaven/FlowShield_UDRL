#!/usr/bin/env python3
"""
Expert Data Collection - Collecte des données de haute qualité.

Méthodes disponibles:
1. Heuristic Expert: Utilise une politique heuristique connue pour LunarLander
2. Filter Best: Collecte beaucoup d'épisodes et garde les meilleurs
3. Mixed: Mélange expert et exploration

Usage:
    python scripts/collect_expert_data.py --env lunarlander --method heuristic --episodes 1000
    python scripts/collect_expert_data.py --env lunarlander --method filter --episodes 5000 --keep-best 500
"""

import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def heuristic_lunar_lander(state):
    """
    Heuristic expert policy for LunarLander.
    Based on the official OpenAI solution.
    """
    # State: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state
    
    # Target position (center pad)
    target_x = 0.0
    target_y = 0.0
    
    # Horizontal control
    x_error = target_x - x
    vx_target = x_error * 0.5
    vx_error = vx_target - vx
    
    # Main engine (vertical thrust)
    # Fire if falling too fast or too high
    main_thrust = 0.0
    if vy < -0.5 or y > 0.5:
        main_thrust = min(1.0, max(0.0, -vy * 0.5 + (y - 0.2) * 0.5))
    
    # Side engines for horizontal and angular control
    # Positive = right thrust, Negative = left thrust
    side_thrust = 0.0
    side_thrust += vx_error * 0.5  # Horizontal velocity control
    side_thrust += angle * 2.0     # Angle correction
    side_thrust += angular_vel * 0.5  # Angular velocity damping
    
    # Clamp to [-1, 1]
    main_thrust = np.clip(main_thrust, 0.0, 1.0)
    side_thrust = np.clip(side_thrust, -1.0, 1.0)
    
    return np.array([main_thrust, side_thrust], dtype=np.float32)


def heuristic_lunar_lander_v2(state):
    """
    Improved heuristic for continuous LunarLander.
    More aggressive control for better landings.
    """
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state
    
    # PID-like control
    # Horizontal position control
    kp_x = 0.4
    kd_x = 0.3
    target_x = 0.0
    x_error = target_x - x
    horizontal_control = kp_x * x_error - kd_x * vx
    
    # Angle control - try to stay upright
    kp_angle = 1.0
    kd_angle = 0.5
    angle_control = -kp_angle * angle - kd_angle * angular_vel
    
    # Combine for side thrust
    side_thrust = horizontal_control + angle_control
    
    # Vertical control - main engine
    # More thrust when high or falling fast
    target_vy = -0.3 if y > 0.3 else -0.1  # Gentle descent
    vy_error = target_vy - vy
    
    kp_y = 0.8
    main_thrust = kp_y * vy_error
    
    # Add some thrust based on height
    if y > 0.5:
        main_thrust += 0.3
    
    # Near landing, be gentler
    if y < 0.2 and vy > -0.5:
        main_thrust = 0.0
    
    # Boost if falling too fast
    if vy < -0.8:
        main_thrust = 1.0
    
    # Clamp
    main_thrust = np.clip(main_thrust, 0.0, 1.0)
    side_thrust = np.clip(side_thrust, -1.0, 1.0)
    
    return np.array([main_thrust, side_thrust], dtype=np.float32)


def collect_expert_trajectories(env, n_episodes: int = 1000, policy_fn=None, verbose: bool = True):
    """
    Collect trajectories using an expert policy.
    """
    trajectories = []
    episode_returns = []
    episode_lengths = []
    
    iterator = tqdm(range(n_episodes), desc="Collecting expert data") if verbose else range(n_episodes)
    
    for _ in iterator:
        obs, _ = env.reset()
        trajectory = []
        episode_return = 0
        done = False
        
        while not done:
            # Use expert policy
            if policy_fn is not None:
                action = policy_fn(obs)
            else:
                action = env.action_space.sample()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory.append({
                'state': obs,
                'action': action,
                'reward': reward,
                'next_state': next_obs,
                'done': done,
            })
            
            episode_return += reward
            obs = next_obs
        
        # Compute hindsight commands
        for i, trans in enumerate(trajectory):
            horizon = len(trajectory) - i
            return_to_go = sum(t['reward'] for t in trajectory[i:])
            trans['command'] = np.array([horizon, return_to_go], dtype=np.float32)
        
        trajectories.append((trajectory, episode_return))
        episode_returns.append(episode_return)
        episode_lengths.append(len(trajectory))
    
    return trajectories, episode_returns, episode_lengths


def filter_best_trajectories(trajectories, keep_n: int = None, min_return: float = None):
    """
    Filter to keep only the best trajectories.
    """
    # Sort by return (highest first)
    sorted_trajs = sorted(trajectories, key=lambda x: x[1], reverse=True)
    
    if min_return is not None:
        sorted_trajs = [(t, r) for t, r in sorted_trajs if r >= min_return]
    
    if keep_n is not None:
        sorted_trajs = sorted_trajs[:keep_n]
    
    return sorted_trajs


def trajectories_to_arrays(trajectories):
    """Convert list of trajectories to numpy arrays."""
    all_transitions = []
    for traj, _ in trajectories:
        all_transitions.extend(traj)
    
    states = np.array([t['state'] for t in all_transitions], dtype=np.float32)
    actions = np.array([t['action'] for t in all_transitions], dtype=np.float32)
    rewards = np.array([t['reward'] for t in all_transitions], dtype=np.float32)
    next_states = np.array([t['next_state'] for t in all_transitions], dtype=np.float32)
    dones = np.array([t['done'] for t in all_transitions], dtype=np.float32)
    commands = np.array([t['command'] for t in all_transitions], dtype=np.float32)
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
        'commands': commands,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Collect expert training data")
    parser.add_argument("--env", type=str, default="lunarlander", 
                        choices=["lunarlander", "cartpole", "pendulum"],
                        help="Environment name")
    parser.add_argument("--method", type=str, default="heuristic",
                        choices=["heuristic", "filter", "mixed"],
                        help="Collection method")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to collect")
    parser.add_argument("--keep-best", type=int, default=None,
                        help="Keep only the best N episodes (for filter method)")
    parser.add_argument("--min-return", type=float, default=None,
                        help="Minimum return to keep episode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create output directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f"{args.env}_expert_{args.method}_{timestamp}.npz"
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("EXPERT DATA COLLECTION")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Method: {args.method}")
    print(f"Episodes: {args.episodes}")
    if args.keep_best:
        print(f"Keep best: {args.keep_best}")
    if args.min_return:
        print(f"Min return: {args.min_return}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Create environment
    env = create_environment(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Select policy based on method
    if args.method == "heuristic":
        if args.env == "lunarlander":
            policy_fn = heuristic_lunar_lander_v2
            print("Using improved heuristic expert policy")
        else:
            policy_fn = None
            print("No expert heuristic available, using random policy")
    elif args.method == "filter":
        policy_fn = None  # Random, then filter
        print("Using random policy + filtering best episodes")
    elif args.method == "mixed":
        # Will alternate between expert and random
        policy_fn = None
        print("Using mixed expert/random collection")
    
    # Collect trajectories
    if args.method == "mixed" and args.env == "lunarlander":
        # Collect half expert, half random
        n_expert = args.episodes // 2
        n_random = args.episodes - n_expert
        
        print(f"\nCollecting {n_expert} expert episodes...")
        expert_trajs, expert_returns, _ = collect_expert_trajectories(
            env, n_expert, heuristic_lunar_lander_v2
        )
        print(f"Expert mean return: {np.mean(expert_returns):.2f} ± {np.std(expert_returns):.2f}")
        
        print(f"\nCollecting {n_random} random episodes...")
        random_trajs, random_returns, _ = collect_expert_trajectories(
            env, n_random, None
        )
        print(f"Random mean return: {np.mean(random_returns):.2f} ± {np.std(random_returns):.2f}")
        
        trajectories = expert_trajs + random_trajs
        episode_returns = expert_returns + random_returns
    else:
        trajectories, episode_returns, episode_lengths = collect_expert_trajectories(
            env, args.episodes, policy_fn
        )
    
    # Filter if requested
    if args.method == "filter" or args.keep_best or args.min_return:
        original_count = len(trajectories)
        trajectories = filter_best_trajectories(
            trajectories, 
            keep_n=args.keep_best,
            min_return=args.min_return
        )
        print(f"\nFiltered: {original_count} -> {len(trajectories)} episodes")
        episode_returns = [r for _, r in trajectories]
    
    # Convert to arrays
    data = trajectories_to_arrays(trajectories)
    
    # Add metadata
    data['env_name'] = args.env
    data['state_dim'] = state_dim
    data['action_dim'] = action_dim
    data['n_episodes'] = len(trajectories)
    data['collection_method'] = args.method
    data['episode_returns'] = np.array(episode_returns)
    data['seed'] = args.seed
    
    # Save
    np.savez(output_path, **data)
    
    print("\n" + "=" * 60)
    print("EXPERT DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Episodes kept: {len(trajectories)}")
    print(f"Total transitions: {len(data['states'])}")
    print(f"Mean episode return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Return range: [{np.min(episode_returns):.1f}, {np.max(episode_returns):.1f}]")
    print(f"\n>>> Saved to: {output_path}")
    print("=" * 60)
    
    env.close()
    return str(output_path)


if __name__ == "__main__":
    main()
