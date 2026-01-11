"""
Modify existing shield to use better projection hyperparameters.
This avoids retraining - we just update the projection config.
"""

import torch
from pathlib import Path

# Load existing shield
shield_path = "results/unknown/models/flow_shield.pt"
print("=" * 60)
print("UPDATING SHIELD PROJECTION HYPERPARAMETERS")
print("=" * 60)

ckpt = torch.load(shield_path, map_location="cpu", weights_only=False)

print(f"\nOriginal shield:")
print(f"  threshold: {ckpt['ood_threshold']:.4f}")

# Add improved projection hyperparameters
ckpt["projection_steps"] = 200  # Was 50
ckpt["projection_lr"] = 0.5     # Was 0.1
ckpt["projection_method"] = "gradient"

print(f"\nUpdated projection config:")
print(f"  projection_steps: 200 (was 50)")
print(f"  projection_lr: 0.5 (was 0.1)")
print(f"  projection_method: gradient")

# Save as improved version
output_path = "results/unknown/models/flow_shield_improved.pt"
torch.save(ckpt, output_path)

print(f"\nImproved shield saved to: {output_path}")

# Now update visualize_with_shield.py to load improved version
print("\n" + "=" * 60)
print("UPDATING VISUALIZATION SCRIPT")
print("=" * 60)

# We need to modify how FlowMatchingSafetyModule is created to use these params
print("\nNote: visualize_with_shield.py needs to be updated to:")
print("1. Load projection_steps and projection_lr from checkpoint")
print("2. Pass them to FlowMatchingSafetyModule constructor")
print("\nLet's update visualize_with_shield.py now...")
