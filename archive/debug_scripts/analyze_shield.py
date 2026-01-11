"""
Simple script to analyze shield projections without complex imports.
"""

import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def analyze_shield_checkpoint(shield_path):
    """Analyze what's in the shield checkpoint."""
    print("=" * 60)
    print("SHIELD CHECKPOINT ANALYSIS")
    print("=" * 60)
    
    ckpt = torch.load(shield_path, map_location="cpu", weights_only=False)
    
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    
    if "ood_threshold" in ckpt:
        print(f"OOD Threshold: {ckpt['ood_threshold']:.4f}")
    
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    print(f"\nModel architecture:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")
    
    # Check log prob distribution
    if "log_probs_train" in ckpt:
        print(f"\nTraining log probs:")
        print(f"  Mean: {ckpt['log_probs_train']['mean']:.4f}")
        print(f"  Std: {ckpt['log_probs_train']['std']:.4f}")
        print(f"  Min: {ckpt['log_probs_train']['min']:.4f}")
        print(f"  Max: {ckpt['log_probs_train']['max']:.4f}")
        print(f"  Percentiles: 5%={ckpt['log_probs_train']['p05']:.4f}, "
              f"95%={ckpt['log_probs_train']['p95']:.4f}")
    
    return ckpt


def test_commands_with_threshold(shield_path):
    """Test different commands to see their log probs."""
    print("\n" + "=" * 60)
    print("COMMAND LOG PROBABILITY TEST")
    print("=" * 60)
    
    # We'll test commands manually
    commands_to_test = [
        ("ID (H=200, R=220)", 200.0, 220.0),
        ("Moderate OOD (H=50, R=350)", 50.0, 350.0),
        ("Strong OOD (H=20, R=400)", 20.0, 400.0),
        ("Extreme OOD (H=5, R=500)", 5.0, 500.0),
    ]
    
    ckpt = torch.load(shield_path, map_location="cpu", weights_only=False)
    threshold = ckpt.get("ood_threshold", -5.0)
    
    print(f"\nOOD Threshold: {threshold:.4f}")
    print(f"Commands with log_prob < {threshold:.4f} will be flagged as OOD")
    
    print(f"\n{'Command':<30} {'Expected log_prob':<20} {'OOD?'}")
    print("-" * 60)
    
    # Based on expert data: mean R=242.7±26.8
    # ID commands should have high log prob
    # OOD commands should have low log prob
    for name, h, r in commands_to_test:
        # Heuristic: log prob decreases as we move away from expert distribution
        expert_mean_r = 242.7
        expert_std_r = 26.8
        expert_mean_h = 200  # Typical
        
        # Distance from expert distribution (simplified)
        r_distance = abs(r - expert_mean_r) / expert_std_r
        h_distance = abs(h - expert_mean_h) / 50
        total_distance = r_distance + h_distance
        
        # Rough estimate: log prob decreases with distance
        estimated_log_prob = -total_distance * 2
        is_ood = estimated_log_prob < threshold
        
        print(f"{name:<30} ~{estimated_log_prob:<19.2f} {is_ood}")
    
    print(f"\nNote: These are ROUGH estimates. Actual log probs require ")
    print(f"running the flow model with ODE integration.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shield-path", default="results/unknown/models/flow_shield.pt")
    args = parser.parse_args()
    
    # Analyze checkpoint
    ckpt = analyze_shield_checkpoint(args.shield_path)
    
    # Test different commands
    test_commands_with_threshold(args.shield_path)
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    threshold = ckpt.get("ood_threshold", -5.0)
    
    print(f"\nCurrent threshold: {threshold:.4f}")
    print(f"\nIssues observed:")
    print(f"1. Shield activates but doesn't prevent crashes (30% failure rate unchanged)")
    print(f"2. Projection may not be converging to safe region")
    print(f"\nPossible causes:")
    print(f"- Threshold too strict (too negative) → legitimate commands flagged as OOD")
    print(f"- Threshold too loose (not negative enough) → dangerous commands pass through")
    print(f"- Projection method (gradient ascent) not effective for very OOD commands")
    print(f"- Flow model quality - may not accurately model p(g|s)")
    
    print(f"\nSuggested fixes:")
    print(f"1. Retrain shield with better threshold calibration")
    print(f"2. Increase projection steps: 50 → 200")
    print(f"3. Increase projection LR: 0.1 → 0.5")
    print(f"4. Use 'sample' projection method instead of 'gradient'")
    print(f"5. Collect more diverse expert data")


if __name__ == "__main__":
    main()
