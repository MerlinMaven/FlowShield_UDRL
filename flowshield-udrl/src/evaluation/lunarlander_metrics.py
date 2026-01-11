"""
Metriques avancees pour LunarLander : Success Rate, Crash Rate, etc.

Ajoute des metriques specifiques a l'environnement:
- Success Rate (atterrissage reussi)
- Crash Rate (collision)
- Timeout Rate (temps ecoule)
- Fuel Efficiency
- Landing Quality Score
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_lunarlander_metrics(
    final_rewards: List[float],
    terminal_states: List[np.ndarray],
    episode_lengths: List[int],
    max_length: int = 1000,
) -> Dict[str, float]:
    """
    Calcule les metriques specifiques a LunarLander.
    
    LunarLander rewards:
    - +100 to +140 for landing on pad (depends on precision)
    - -100 for crash
    - ~+10 for each leg contact
    - -0.3 per main engine use
    - -0.03 per side engine use
    
    Args:
        final_rewards: Reward final de chaque episode
        terminal_states: Etat terminal [x, y, vx, vy, angle, angular_vel, leg1, leg2]
        episode_lengths: Longueur de chaque episode
        max_length: Longueur maximale (timeout)
        
    Returns:
        Dictionnaire de metriques
    """
    n_episodes = len(final_rewards)
    final_rewards = np.array(final_rewards)
    terminal_states = np.array(terminal_states)
    episode_lengths = np.array(episode_lengths)
    
    # Classification des issues
    # Landing reussi: reward final >= 100 et les deux jambes au sol
    leg1_contact = terminal_states[:, 6] > 0.5
    leg2_contact = terminal_states[:, 7] > 0.5
    both_legs = leg1_contact & leg2_contact
    high_final_reward = final_rewards >= 100
    
    successful_landings = both_legs & high_final_reward
    
    # Crash: reward final <= -100 ou grande vitesse a y~0
    crash_reward = final_rewards <= -100
    
    # Peut aussi detecter par la velocite au sol
    y_near_ground = terminal_states[:, 1] < 0.1
    high_velocity = np.sqrt(terminal_states[:, 2]**2 + terminal_states[:, 3]**2) > 0.5
    physical_crash = y_near_ground & high_velocity & ~both_legs
    
    crashes = crash_reward | physical_crash
    
    # Timeout: episode atteint la longueur max sans landing ni crash
    timeouts = (episode_lengths >= max_length - 10) & ~successful_landings & ~crashes
    
    # Partial success: un peu de reward positif mais pas de landing propre
    partial_success = ~successful_landings & ~crashes & ~timeouts & (final_rewards > 0)
    
    # Metriques de base
    metrics = {
        'success_rate': float(np.mean(successful_landings)),
        'crash_rate': float(np.mean(crashes)),
        'timeout_rate': float(np.mean(timeouts)),
        'partial_success_rate': float(np.mean(partial_success)),
    }
    
    # Metriques de qualite d'atterrissage (sur les succes seulement)
    if np.sum(successful_landings) > 0:
        success_states = terminal_states[successful_landings]
        
        # Position X (0 = centre du pad)
        x_precision = np.abs(success_states[:, 0])
        metrics['landing_x_error_mean'] = float(np.mean(x_precision))
        metrics['landing_x_error_std'] = float(np.std(x_precision))
        
        # Angle (0 = vertical)
        angle_precision = np.abs(success_states[:, 4])
        metrics['landing_angle_error_mean'] = float(np.mean(angle_precision))
        
        # Velocite a l'atterrissage
        landing_velocity = np.sqrt(success_states[:, 2]**2 + success_states[:, 3]**2)
        metrics['landing_velocity_mean'] = float(np.mean(landing_velocity))
        
        # Score de qualite composite
        # Plus c'est bas, mieux c'est
        quality_score = (
            0.5 * x_precision / 0.5 +  # Normalise par ecart typique
            0.3 * angle_precision / 0.3 +
            0.2 * landing_velocity / 0.5
        )
        metrics['landing_quality_score'] = float(1.0 - np.mean(np.clip(quality_score, 0, 1)))
    else:
        metrics['landing_x_error_mean'] = float('nan')
        metrics['landing_x_error_std'] = float('nan')
        metrics['landing_angle_error_mean'] = float('nan')
        metrics['landing_velocity_mean'] = float('nan')
        metrics['landing_quality_score'] = 0.0
    
    # Efficacite (sur tous les episodes)
    metrics['mean_episode_length'] = float(np.mean(episode_lengths))
    metrics['std_episode_length'] = float(np.std(episode_lengths))
    
    # Temps pour reussir (sur les succes)
    if np.sum(successful_landings) > 0:
        success_lengths = episode_lengths[successful_landings]
        metrics['success_mean_length'] = float(np.mean(success_lengths))
    else:
        metrics['success_mean_length'] = float('nan')
    
    return metrics


def compute_shield_effectiveness(
    with_shield_crashes: List[bool],
    without_shield_crashes: List[bool],
    shield_activations: List[bool],
    ood_commands: List[bool],
) -> Dict[str, float]:
    """
    Mesure l'efficacite du shield.
    
    Args:
        with_shield_crashes: Crashes avec shield actif
        without_shield_crashes: Crashes sans shield
        shield_activations: Quand le shield s'est active
        ood_commands: Quels commands etaient OOD
        
    Returns:
        Metriques d'efficacite
    """
    with_crashes = np.array(with_shield_crashes)
    without_crashes = np.array(without_shield_crashes)
    activations = np.array(shield_activations)
    ood = np.array(ood_commands)
    
    metrics = {}
    
    # Reduction du taux de crash
    crash_rate_with = np.mean(with_crashes)
    crash_rate_without = np.mean(without_crashes)
    
    metrics['crash_rate_with_shield'] = float(crash_rate_with)
    metrics['crash_rate_without_shield'] = float(crash_rate_without)
    metrics['crash_reduction_absolute'] = float(crash_rate_without - crash_rate_with)
    metrics['crash_reduction_relative'] = float(
        (crash_rate_without - crash_rate_with) / (crash_rate_without + 1e-8)
    )
    
    # Precision du shield
    if np.sum(ood) > 0:
        # True Positive: shield active sur OOD
        tp = np.sum(activations & ood)
        # False Negative: OOD non detecte
        fn = np.sum(~activations & ood)
        
        metrics['ood_detection_recall'] = float(tp / (tp + fn + 1e-8))
    else:
        metrics['ood_detection_recall'] = float('nan')
    
    if np.sum(~ood) > 0:
        # False Positive: shield active sur ID
        fp = np.sum(activations & ~ood)
        # True Negative: pas d'activation sur ID
        tn = np.sum(~activations & ~ood)
        
        metrics['id_preservation_rate'] = float(tn / (tn + fp + 1e-8))
    else:
        metrics['id_preservation_rate'] = float('nan')
    
    # Activation rate
    metrics['shield_activation_rate'] = float(np.mean(activations))
    
    return metrics


def compute_command_tracking_metrics(
    target_returns: List[float],
    actual_returns: List[float],
    target_horizons: List[int],
    actual_lengths: List[int],
) -> Dict[str, float]:
    """
    Mesure la capacite a suivre les commandes.
    
    Args:
        target_returns: Returns cibles (commandes)
        actual_returns: Returns realises
        target_horizons: Horizons cibles
        actual_lengths: Longueurs realisees
        
    Returns:
        Metriques de suivi
    """
    target_r = np.array(target_returns)
    actual_r = np.array(actual_returns)
    target_h = np.array(target_horizons)
    actual_l = np.array(actual_lengths)
    
    # Achievement ratio
    achievement = actual_r / (target_r + 1e-8)
    
    metrics = {
        'mean_achievement_ratio': float(np.mean(achievement)),
        'std_achievement_ratio': float(np.std(achievement)),
        'achievement_90pct': float(np.mean(achievement >= 0.9)),
        'achievement_100pct': float(np.mean(achievement >= 1.0)),
        'over_achievement_rate': float(np.mean(achievement > 1.0)),
    }
    
    # Erreur absolue
    return_error = actual_r - target_r
    metrics['mean_return_error'] = float(np.mean(return_error))
    metrics['mean_abs_return_error'] = float(np.mean(np.abs(return_error)))
    
    # Horizon tracking
    horizon_ratio = actual_l / (target_h + 1e-8)
    metrics['mean_horizon_ratio'] = float(np.mean(horizon_ratio))
    
    # Correlation commande-resultat
    metrics['return_correlation'] = float(np.corrcoef(target_r, actual_r)[0, 1])
    
    return metrics


def compute_stability_metrics(
    returns: List[float],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Mesure la stabilite du comportement.
    
    Args:
        returns: Liste des returns
        n_bins: Nombre de bins pour l'analyse temporelle
        
    Returns:
        Metriques de stabilite
    """
    returns = np.array(returns)
    n = len(returns)
    
    if n < n_bins:
        n_bins = max(2, n // 2)
    
    # Diviser en bins temporels
    bin_size = n // n_bins
    bin_means = []
    bin_stds = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        bin_returns = returns[start:end]
        bin_means.append(np.mean(bin_returns))
        bin_stds.append(np.std(bin_returns))
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    metrics = {
        # Variance de la moyenne entre bins (stabilite)
        'performance_stability': float(np.std(bin_means)),
        
        # Trend (amelioration/degradation)
        'performance_trend': float(np.corrcoef(range(n_bins), bin_means)[0, 1]),
        
        # Variance moyenne intra-bin
        'mean_intra_variance': float(np.mean(bin_stds)),
        
        # Coefficient de variation
        'coefficient_of_variation': float(np.std(returns) / (np.mean(returns) + 1e-8)),
        
        # Worst-case analysis
        'worst_5pct_mean': float(np.mean(np.sort(returns)[:max(1, n // 20)])),
        'best_5pct_mean': float(np.mean(np.sort(returns)[-max(1, n // 20):])),
    }
    
    return metrics


def full_evaluation_report(
    returns: List[float],
    lengths: List[int],
    terminal_states: List[np.ndarray],
    final_rewards: List[float],
    target_returns: List[float],
    target_horizons: List[int],
    shield_activations: Optional[List[bool]] = None,
    ood_commands: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """
    Rapport complet d'evaluation.
    
    Combine toutes les metriques en un seul rapport.
    """
    report = {}
    
    # Performance de base
    report['mean_return'] = float(np.mean(returns))
    report['std_return'] = float(np.std(returns))
    report['min_return'] = float(np.min(returns))
    report['max_return'] = float(np.max(returns))
    
    # Metriques LunarLander
    ll_metrics = compute_lunarlander_metrics(
        final_rewards, terminal_states, lengths
    )
    report.update({f'll_{k}': v for k, v in ll_metrics.items()})
    
    # Command tracking
    tracking = compute_command_tracking_metrics(
        target_returns, returns, target_horizons, lengths
    )
    report.update({f'track_{k}': v for k, v in tracking.items()})
    
    # Stabilite
    stability = compute_stability_metrics(returns)
    report.update({f'stab_{k}': v for k, v in stability.items()})
    
    # Shield effectiveness (si disponible)
    if shield_activations is not None:
        report['shield_activation_rate'] = float(np.mean(shield_activations))
    
    return report
