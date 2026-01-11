"""
Re-train Flow Shield with improved projection hyperparameters.

The problem: gradient ascent projection with 50 steps and LR=0.1 
doesn't converge for extreme OOD commands like (H=5, R=500).

Solution: Train new shield with:
- projection_steps=200 (instead of 50)
- projection_lr=0.5 (instead of 0.1)
- This should allow better convergence to safe regions
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import only what we need to avoid circular dependencies
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def load_shield_architecture(checkpoint_path=None):
    """Load FlowMatchingSafetyModule architecture."""
    from src.models.safety.flow_matching import FlowMatchingSafetyModule
    
    shield = FlowMatchingSafetyModule(
        state_dim=8,
        command_dim=2,
        hidden_dim=256,
        n_layers=4,
        ood_threshold=-4.0,  # Will be calibrated
        projection_method="gradient",
        projection_steps=200,  # INCREASED from 50
        projection_lr=0.5,     # INCREASED from 0.1
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        # Load pre-trained weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            shield.load_state_dict(ckpt["model_state_dict"])
        else:
            shield.load_state_dict(ckpt)
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"Updating projection params: steps=200, lr=0.5")
    
    return shield


def calibrate_threshold(shield, states, commands, percentile=5):
    """Calibrate OOD threshold at given percentile of training log probs."""
    shield.eval()
    
    # Compute log probs for all training data
    log_probs = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch_states = torch.tensor(
                states[i:i+batch_size], 
                dtype=torch.float32
            )
            batch_commands = torch.tensor(
                commands[i:i+batch_size], 
                dtype=torch.float32
            )
            
            lp = shield.get_safety_score(batch_states, batch_commands)
            log_probs.append(lp)
    
    log_probs = torch.cat(log_probs).numpy()
    
    # Set threshold at percentile
    threshold = np.percentile(log_probs, percentile)
    shield.ood_threshold = threshold
    
    return threshold, log_probs


def main():
    print("=" * 60)
    print("IMPROVED FLOW SHIELD TRAINING")
    print("=" * 60)
    
    # Load data
    data_path = "data/lunarlander_expert.npz"
    print(f"\nLoading data: {data_path}")
    
    data = np.load(data_path)
    states = data["states"]
    commands = data["commands"]
    
    print(f"States: {states.shape}")
    print(f"Commands: {commands.shape}")
    
    # Load pretrained shield and update projection params
    pretrained_path = "results/unknown/models/flow_shield.pt"
    shield = load_shield_architecture(checkpoint_path=pretrained_path)
    
    print(f"\nShield configuration:")
    print(f"  projection_method: {shield.projection_method}")
    print(f"  projection_steps: {shield.projection_steps}")
    print(f"  projection_lr: {shield.projection_lr}")
    
    # Calibrate threshold
    print(f"\nCalibrating OOD threshold...")
    threshold, log_probs = calibrate_threshold(shield, states, commands, percentile=5)
    
    print(f"\nLog probability distribution:")
    print(f"  Mean: {log_probs.mean():.4f}")
    print(f"  Std: {log_probs.std():.4f}")
    print(f"  Min: {log_probs.min():.4f}")
    print(f"  Max: {log_probs.max():.4f}")
    print(f"  5th percentile: {np.percentile(log_probs, 5):.4f}")
    print(f"  95th percentile: {np.percentile(log_probs, 95):.4f}")
    print(f"\nOOD threshold set to: {threshold:.4f}")
    
    # Save improved shield
    output_path = "results/unknown/models/flow_shield_improved.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        "ood_threshold": shield.ood_threshold,
        "projection_steps": shield.projection_steps,
        "projection_lr": shield.projection_lr,
        "projection_method": shield.projection_method,
        "log_prob_mean": float(log_probs.mean()),
        "log_prob_std": float(log_probs.std()),
    }
    
    # Add model state dict
    for key, value in shield.state_dict().items():
        checkpoint[key] = value
    
    torch.save(checkpoint, output_path)
    print(f"\nImproved shield saved to: {output_path}")
    
    # Test projection on extreme OOD command
    print("\n" + "=" * 60)
    print("TEST PROJECTION ON EXTREME OOD")
    print("=" * 60)
    
    # Use initial state from env
    import gymnasium as gym
    env = gym.make("LunarLander-v3", continuous=True)
    state, _ = env.reset(seed=42)
    env.close()
    
    # Test command
    test_command = torch.tensor([[5.0, 500.0]])  # Extreme OOD
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Initial log prob
    initial_log_prob = shield.get_safety_score(state_t, test_command)
    print(f"\nOriginal command: H=5.0, R=500.0")
    print(f"Initial log_prob: {initial_log_prob.item():.4f}")
    print(f"Is OOD: {initial_log_prob.item() < shield.ood_threshold}")
    
    # Project
    projected = shield.project(state_t, test_command)
    final_log_prob = shield.get_safety_score(state_t, projected)
    
    print(f"\nProjected command: H={projected[0, 0].item():.1f}, R={projected[0, 1].item():.1f}")
    print(f"Final log_prob: {final_log_prob.item():.4f}")
    print(f"Is safe: {final_log_prob.item() > shield.ood_threshold}")
    print(f"Projection distance: {torch.norm(projected - test_command).item():.2f}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("\nTest improved shield with:")
    print("  python scripts/visualize_with_shield.py \\")
    print("    --shield-path results/unknown/models/flow_shield_improved.pt \\")
    print("    --n-episodes 10 --horizon 5 --target-return 500")


if __name__ == "__main__":
    main()
