"""
Evaluation complete avec toutes les metriques.

Ce script effectue une evaluation exhaustive de l'agent et des shields.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym

# Ajout du path pour imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.lunarlander_metrics import (
    compute_lunarlander_metrics,
    compute_command_tracking_metrics,
    compute_stability_metrics,
    full_evaluation_report,
)


def evaluate_episode(env, policy, command, shield=None, max_steps=1000, 
                     deterministic=True, device='cuda'):
    """
    Execute un episode et collecte les metriques.
    
    Returns:
        Dict avec toutes les infos de l'episode
    """
    state, info = env.reset()
    
    episode_return = 0.0
    episode_length = 0
    actions_taken = []
    states_visited = []
    rewards_received = []
    shield_activated = False
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    command_tensor = torch.FloatTensor(command).unsqueeze(0).to(device)
    current_command = command_tensor.clone()
    
    for step in range(max_steps):
        states_visited.append(state.copy())
        
        # Appliquer le shield si present
        if shield is not None:
            with torch.no_grad():
                log_prob = shield.log_prob(state_tensor, current_command)
                if log_prob.item() < shield.threshold:
                    shield_activated = True
                    current_command = shield.project(state_tensor, current_command)
        
        # Obtenir l'action
        with torch.no_grad():
            if hasattr(policy, 'get_action'):
                action = policy.get_action(state_tensor, current_command, 
                                          deterministic=deterministic)
            else:
                action = policy(state_tensor, current_command)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()
        
        actions_taken.append(action.copy() if hasattr(action, 'copy') else action)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        rewards_received.append(reward)
        episode_return += reward
        episode_length += 1
        
        done = terminated or truncated
        
        if done:
            break
        
        state = next_state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Update command
        current_command[0, 0] = max(1, current_command[0, 0] - 1)
        current_command[0, 1] = current_command[0, 1] - reward
    
    # Determiner le type de terminaison
    final_reward = rewards_received[-1] if rewards_received else 0
    final_state = states_visited[-1] if states_visited else state
    
    if final_reward >= 100:
        termination = 'success'
    elif final_reward <= -100:
        termination = 'crash'
    elif episode_length >= max_steps - 1:
        termination = 'timeout'
    else:
        termination = 'other'
    
    return {
        'return': episode_return,
        'length': episode_length,
        'final_reward': final_reward,
        'final_state': final_state,
        'termination': termination,
        'shield_activated': shield_activated,
        'states': np.array(states_visited),
        'actions': np.array(actions_taken),
        'rewards': np.array(rewards_received),
    }


def evaluate_command_set(env, policy, commands, shield=None, 
                         n_episodes=20, device='cuda', verbose=True):
    """
    Evalue un ensemble de commandes.
    """
    all_results = []
    
    for cmd_idx, (horizon, target_return) in enumerate(commands):
        if verbose:
            print(f"\n  Command {cmd_idx+1}/{len(commands)}: H={horizon}, R={target_return}")
        
        cmd_results = []
        for ep in range(n_episodes):
            result = evaluate_episode(
                env, policy, 
                command=[horizon, target_return],
                shield=shield,
                device=device,
            )
            result['target_return'] = target_return
            result['target_horizon'] = horizon
            cmd_results.append(result)
            
            if verbose and (ep + 1) % 10 == 0:
                mean_r = np.mean([r['return'] for r in cmd_results])
                print(f"    Episode {ep+1}: Mean R = {mean_r:.1f}")
        
        all_results.extend(cmd_results)
    
    return all_results


def compute_all_metrics(results):
    """
    Calcule toutes les metriques a partir des resultats.
    """
    returns = [r['return'] for r in results]
    lengths = [r['length'] for r in results]
    final_rewards = [r['final_reward'] for r in results]
    terminal_states = [r['final_state'] for r in results]
    target_returns = [r['target_return'] for r in results]
    target_horizons = [r['target_horizon'] for r in results]
    terminations = [r['termination'] for r in results]
    shield_activations = [r['shield_activated'] for r in results]
    
    # Metriques de base
    metrics = {
        'n_episodes': len(results),
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'median_return': float(np.median(returns)),
    }
    
    # Metriques LunarLander
    ll_metrics = compute_lunarlander_metrics(
        final_rewards, terminal_states, lengths
    )
    metrics.update({f'll_{k}': v for k, v in ll_metrics.items()})
    
    # Metriques de tracking
    tracking = compute_command_tracking_metrics(
        target_returns, returns, target_horizons, lengths
    )
    metrics.update({f'track_{k}': v for k, v in tracking.items()})
    
    # Metriques de stabilite
    stability = compute_stability_metrics(returns)
    metrics.update({f'stab_{k}': v for k, v in stability.items()})
    
    # Shield metrics
    metrics['shield_activation_rate'] = float(np.mean(shield_activations))
    metrics['shield_activations'] = int(np.sum(shield_activations))
    
    # Termination breakdown
    metrics['success_count'] = sum(1 for t in terminations if t == 'success')
    metrics['crash_count'] = sum(1 for t in terminations if t == 'crash')
    metrics['timeout_count'] = sum(1 for t in terminations if t == 'timeout')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Complete evaluation')
    parser.add_argument('--policy', type=str, required=True, 
                       help='Path to policy checkpoint')
    parser.add_argument('--shield', type=str, default=None,
                       help='Path to shield checkpoint')
    parser.add_argument('--shield-type', type=str, default='flow',
                       choices=['flow', 'quantile', 'diffusion', 'none'])
    parser.add_argument('--n-episodes', type=int, default=50)
    parser.add_argument('--output', type=str, default='evaluation_report.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print('='*60)
    print('EVALUATION COMPLETE FlowShield-UDRL')
    print('='*60)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'\nDevice: {device}')
    
    # Create environment
    print('\nCreation environnement...')
    env = gym.make('LunarLander-v3', continuous=True)
    
    # Load policy
    print(f'Chargement policy: {args.policy}')
    # Note: Implementation depends on your model class
    # policy = UDRLPolicy.load(args.policy).to(device)
    print('  [Placeholder - implementer le chargement]')
    
    # Load shield if specified
    shield = None
    if args.shield:
        print(f'Chargement shield: {args.shield}')
        # shield = load_shield(args.shield, args.shield_type).to(device)
        print('  [Placeholder - implementer le chargement]')
    
    # Define command sets
    id_commands = [
        (200, 220),  # Typique
        (150, 200),  # Modere
        (250, 240),  # Long horizon
    ]
    
    ood_commands = [
        (50, 350),   # Impossible: return trop haut
        (30, 250),   # Difficile: horizon court
        (100, 400),  # Extreme
    ]
    
    print('\n' + '-'*60)
    print('EVALUATION IN-DISTRIBUTION')
    print('-'*60)
    # id_results = evaluate_command_set(env, policy, id_commands, 
    #                                   shield=None, n_episodes=args.n_episodes)
    # id_metrics = compute_all_metrics(id_results)
    
    print('\n' + '-'*60)
    print('EVALUATION OOD (sans shield)')
    print('-'*60)
    # ood_no_shield = evaluate_command_set(env, policy, ood_commands,
    #                                      shield=None, n_episodes=args.n_episodes)
    # ood_no_shield_metrics = compute_all_metrics(ood_no_shield)
    
    if shield:
        print('\n' + '-'*60)
        print('EVALUATION OOD (avec shield)')
        print('-'*60)
        # ood_with_shield = evaluate_command_set(env, policy, ood_commands,
        #                                        shield=shield, n_episodes=args.n_episodes)
        # ood_with_shield_metrics = compute_all_metrics(ood_with_shield)
    
    env.close()
    
    # Compile report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'policy': args.policy,
            'shield': args.shield,
            'shield_type': args.shield_type,
            'n_episodes': args.n_episodes,
        },
        # 'in_distribution': id_metrics,
        # 'ood_no_shield': ood_no_shield_metrics,
    }
    
    # if shield:
    #     report['ood_with_shield'] = ood_with_shield_metrics
    #     report['shield_improvement'] = {
    #         'return_improvement': ood_with_shield_metrics['mean_return'] - ood_no_shield_metrics['mean_return'],
    #         'crash_reduction': ood_no_shield_metrics['ll_crash_rate'] - ood_with_shield_metrics['ll_crash_rate'],
    #     }
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'\nRapport sauvegarde: {output_path}')
    
    # Print summary
    print('\n' + '='*60)
    print('RESUME')
    print('='*60)
    print('\n[Implementer l\'affichage du resume]')


if __name__ == '__main__':
    main()
