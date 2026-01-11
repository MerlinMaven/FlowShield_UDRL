"""
Performance and safety metrics for evaluation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_performance_metrics(
    returns: List[float],
    lengths: List[int],
    target_returns: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute standard RL performance metrics.
    
    Args:
        returns: List of episode returns
        lengths: List of episode lengths
        target_returns: Optional target returns for gap analysis
        
    Returns:
        Dictionary of metrics
    """
    returns = np.array(returns)
    lengths = np.array(lengths)
    
    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }
    
    # Interquartile range
    metrics["iqr_return"] = float(np.percentile(returns, 75) - np.percentile(returns, 25))
    
    # Target tracking (if target returns provided)
    if target_returns is not None:
        target_returns = np.array(target_returns)
        gaps = target_returns - returns
        
        metrics["mean_gap"] = float(np.mean(gaps))
        metrics["gap_ratio"] = float(np.mean(returns / (target_returns + 1e-8)))
        metrics["target_achieved_rate"] = float(np.mean(returns >= target_returns * 0.9))
    
    return metrics


def compute_safety_metrics(
    is_safe_predictions: List[bool],
    actual_outcomes: List[float],
    safety_threshold: float,
    target_returns: List[float],
) -> Dict[str, float]:
    """
    Compute safety-related metrics.
    
    Measures how well the safety module prevents harmful commands.
    
    Args:
        is_safe_predictions: Safety module predictions
        actual_outcomes: Actual returns achieved
        safety_threshold: Return threshold for "safe" outcome
        target_returns: Target returns from commands
        
    Returns:
        Dictionary of safety metrics
    """
    is_safe_predictions = np.array(is_safe_predictions)
    actual_outcomes = np.array(actual_outcomes)
    target_returns = np.array(target_returns)
    
    # Ground truth: was the command actually achievable?
    actually_achievable = actual_outcomes >= target_returns * 0.8
    
    # Classification metrics
    true_positives = np.sum(is_safe_predictions & actually_achievable)
    true_negatives = np.sum(~is_safe_predictions & ~actually_achievable)
    false_positives = np.sum(is_safe_predictions & ~actually_achievable)
    false_negatives = np.sum(~is_safe_predictions & actually_achievable)
    
    total = len(is_safe_predictions)
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Safety-specific metrics
    # False alarm rate: predicting unsafe when actually safe
    false_alarm_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Miss rate: predicting safe when actually unsafe (dangerous!)
    miss_rate = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,  # Critical for safety
        "true_positive_rate": recall,
        "true_negative_rate": true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
    }


def obedient_suicide_rate(
    commands: List[Tuple[float, float]],
    outcomes: List[Tuple[float, float]],
    termination_types: List[str],
    trap_threshold: float = -10.0,
) -> Dict[str, float]:
    """
    Compute the "Obedient Suicide" rate.
    
    Obedient Suicide occurs when an agent is given an impossible command
    (e.g., very high return in short horizon) and ends up in a terminal
    negative state (trap/death) while trying to achieve it.
    
    This is the core problem that FlowShield-UDRL aims to solve.
    
    Args:
        commands: List of (horizon, target_return) commands
        outcomes: List of (actual_steps, actual_return) outcomes
        termination_types: How each episode ended ("goal", "trap", "timeout", etc.)
        trap_threshold: Return threshold indicating trap/death state
        
    Returns:
        Obedient Suicide metrics
    """
    commands = np.array(commands)
    outcomes = np.array(outcomes)
    
    n_episodes = len(commands)
    
    # Identify potentially dangerous commands (high return, low horizon)
    # Heuristic: command is "ambitious" if return/horizon ratio is high
    command_ambition = commands[:, 1] / (commands[:, 0] + 1)  # return per step
    ambitious_threshold = np.percentile(command_ambition, 75)
    ambitious_commands = command_ambition > ambitious_threshold
    
    # Identify suicide outcomes
    trap_outcomes = np.array([t == "trap" for t in termination_types])
    negative_returns = outcomes[:, 1] < trap_threshold
    
    # Obedient suicide: ambitious command + trap outcome
    obedient_suicides = ambitious_commands & trap_outcomes
    
    # Metrics
    suicide_rate = np.mean(obedient_suicides)
    
    # Among ambitious commands
    suicide_given_ambitious = np.mean(obedient_suicides[ambitious_commands]) if np.sum(ambitious_commands) > 0 else 0
    
    # Among non-ambitious commands (baseline)
    baseline_trap_rate = np.mean(trap_outcomes[~ambitious_commands]) if np.sum(~ambitious_commands) > 0 else 0
    
    # Risk increase from ambitious commands
    risk_increase = suicide_given_ambitious / (baseline_trap_rate + 1e-8)
    
    return {
        "obedient_suicide_rate": float(suicide_rate),
        "suicide_given_ambitious": float(suicide_given_ambitious),
        "baseline_trap_rate": float(baseline_trap_rate),
        "risk_increase": float(risk_increase),
        "n_obedient_suicides": int(np.sum(obedient_suicides)),
        "n_ambitious_commands": int(np.sum(ambitious_commands)),
        "n_trap_outcomes": int(np.sum(trap_outcomes)),
    }


def compute_ood_metrics(
    in_distribution_scores: torch.Tensor,
    out_of_distribution_scores: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute out-of-distribution detection metrics.
    
    Args:
        in_distribution_scores: Safety scores for ID data
        out_of_distribution_scores: Safety scores for OOD data
        
    Returns:
        OOD detection metrics (AUROC, AUPRC, FPR@95TPR, etc.)
    """
    id_scores = in_distribution_scores.cpu().numpy()
    ood_scores = out_of_distribution_scores.cpu().numpy()
    
    # Labels: 1 for ID, 0 for OOD
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    # Sort by score (descending - higher score = more likely ID)
    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]
    sorted_scores = scores[sorted_idx]
    
    # ROC curve
    tpr_list = []
    fpr_list = []
    
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    thresholds = np.unique(sorted_scores)
    
    for thresh in thresholds:
        pred_positive = scores >= thresh
        
        tp = np.sum((labels == 1) & pred_positive)
        fp = np.sum((labels == 0) & pred_positive)
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    
    # AUROC (trapezoidal integration)
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]
    
    auroc = np.trapz(tpr_sorted, fpr_sorted)
    
    # FPR at 95% TPR
    idx_95tpr = np.argmin(np.abs(tpr_arr - 0.95))
    fpr_at_95tpr = fpr_arr[idx_95tpr]
    
    # Detection error (average of FPR and FNR at optimal threshold)
    fnr_arr = 1 - tpr_arr
    detection_error = np.min((fpr_arr + fnr_arr) / 2)
    
    # Separation metrics
    mean_gap = np.mean(id_scores) - np.mean(ood_scores)
    overlap = compute_distribution_overlap(id_scores, ood_scores)
    
    return {
        "auroc": float(auroc),
        "fpr_at_95tpr": float(fpr_at_95tpr),
        "detection_error": float(detection_error),
        "id_score_mean": float(np.mean(id_scores)),
        "ood_score_mean": float(np.mean(ood_scores)),
        "score_gap": float(mean_gap),
        "distribution_overlap": float(overlap),
    }


def compute_distribution_overlap(
    scores1: np.ndarray,
    scores2: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Compute overlap between two score distributions.
    
    Lower overlap indicates better separation.
    """
    # Combined range
    all_scores = np.concatenate([scores1, scores2])
    min_val, max_val = np.min(all_scores), np.max(all_scores)
    
    if min_val == max_val:
        return 1.0
    
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist1, _ = np.histogram(scores1, bins=bins, density=True)
    hist2, _ = np.histogram(scores2, bins=bins, density=True)
    
    # Normalize
    hist1 = hist1 / (np.sum(hist1) + 1e-8)
    hist2 = hist2 / (np.sum(hist2) + 1e-8)
    
    # Overlap (intersection)
    overlap = np.sum(np.minimum(hist1, hist2))
    
    return overlap


def command_achievability_metrics(
    commands: torch.Tensor,
    outcomes: torch.Tensor,
    safety_scores: torch.Tensor,
) -> Dict[str, float]:
    """
    Analyze relationship between commands, safety scores, and outcomes.
    
    Args:
        commands: (N, 2) tensor of [horizon, return] commands
        outcomes: (N, 2) tensor of [steps, return] outcomes
        safety_scores: (N,) tensor of safety scores
        
    Returns:
        Correlation and calibration metrics
    """
    commands = commands.cpu().numpy()
    outcomes = outcomes.cpu().numpy()
    safety_scores = safety_scores.cpu().numpy()
    
    target_returns = commands[:, 1]
    actual_returns = outcomes[:, 1]
    
    # Achievement ratio
    achievement_ratio = actual_returns / (target_returns + 1e-8)
    
    # Correlation between safety score and achievement
    correlation = np.corrcoef(safety_scores, achievement_ratio)[0, 1]
    
    # Calibration: do high-confidence predictions come true?
    sorted_idx = np.argsort(safety_scores)[::-1]
    top_10_pct = sorted_idx[:len(sorted_idx) // 10]
    bottom_10_pct = sorted_idx[-len(sorted_idx) // 10:]
    
    top_achievement = np.mean(achievement_ratio[top_10_pct])
    bottom_achievement = np.mean(achievement_ratio[bottom_10_pct])
    
    return {
        "mean_achievement_ratio": float(np.mean(achievement_ratio)),
        "std_achievement_ratio": float(np.std(achievement_ratio)),
        "score_achievement_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "top_10pct_achievement": float(top_achievement),
        "bottom_10pct_achievement": float(bottom_achievement),
        "calibration_gap": float(top_achievement - bottom_achievement),
    }
