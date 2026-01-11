"""
Main evaluation classes for FlowShield-UDRL.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.agent import UDRLPolicy
from ..models.safety import BaseSafetyModule
from .metrics import (
    compute_ood_metrics,
    compute_performance_metrics,
    compute_safety_metrics,
    obedient_suicide_rate,
)


class Evaluator:
    """
    Main evaluator for UDRL policies.
    
    Runs evaluation episodes and computes performance metrics.
    
    Args:
        policy: UDRL policy to evaluate
        env_fn: Function that creates environments
        device: Device for inference
        
    Example:
        >>> evaluator = Evaluator(policy, lambda: gym.make("LunarLander-v2"))
        >>> results = evaluator.evaluate(
        ...     commands=[(50, 200), (100, 250)],
        ...     n_episodes=20
        ... )
    """
    
    def __init__(
        self,
        policy: UDRLPolicy,
        env_fn: Callable,
        safety_module: Optional[BaseSafetyModule] = None,
        device: str = "cuda",
    ):
        self.policy = policy
        self.env_fn = env_fn
        self.safety_module = safety_module
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.policy.to(self.device)
        self.policy.eval()
        
        if self.safety_module:
            self.safety_module.to(self.device)
            self.safety_module.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        commands: List[Tuple[float, float]],
        n_episodes: int = 10,
        max_steps: int = 1000,
        deterministic: bool = True,
        use_safety_filter: bool = True,
        record_trajectories: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate policy with given commands.
        
        Args:
            commands: List of (horizon, target_return) commands
            n_episodes: Episodes per command
            max_steps: Maximum steps per episode
            deterministic: Use deterministic policy
            use_safety_filter: Apply safety module filtering
            record_trajectories: Store full trajectories
            verbose: Print progress
            
        Returns:
            Comprehensive evaluation results
        """
        all_returns = []
        all_lengths = []
        all_target_returns = []
        all_terminations = []
        all_outcomes = []
        all_commands_used = []
        trajectories = []
        
        env = self.env_fn()
        
        for cmd_idx, (horizon, target_return) in enumerate(commands):
            if verbose:
                print(f"Command {cmd_idx + 1}/{len(commands)}: H={horizon}, R={target_return}")
            
            for ep in range(n_episodes):
                result = self._run_episode(
                    env,
                    horizon=horizon,
                    target_return=target_return,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    use_safety_filter=use_safety_filter,
                    record_trajectory=record_trajectories,
                )
                
                all_returns.append(result["total_return"])
                all_lengths.append(result["length"])
                all_target_returns.append(target_return)
                all_terminations.append(result["termination_type"])
                all_outcomes.append((result["length"], result["total_return"]))
                all_commands_used.append((horizon, target_return))
                
                if record_trajectories:
                    trajectories.append(result["trajectory"])
        
        env.close()
        
        # Compute metrics
        performance_metrics = compute_performance_metrics(
            all_returns, all_lengths, all_target_returns
        )
        
        suicide_metrics = obedient_suicide_rate(
            all_commands_used,
            all_outcomes,
            all_terminations,
        )
        
        results = {
            "performance": performance_metrics,
            "suicide": suicide_metrics,
            "raw_returns": all_returns,
            "raw_lengths": all_lengths,
            "terminations": all_terminations,
        }
        
        if record_trajectories:
            results["trajectories"] = trajectories
        
        return results
    
    def _run_episode(
        self,
        env,
        horizon: float,
        target_return: float,
        max_steps: int,
        deterministic: bool,
        use_safety_filter: bool,
        record_trajectory: bool,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        state, info = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        command = torch.tensor([[horizon, target_return]], device=self.device)
        
        total_return = 0.0
        trajectory = [] if record_trajectory else None
        termination_type = "timeout"
        
        for step in range(max_steps):
            # Apply safety filter
            if use_safety_filter and self.safety_module:
                filtered_command, safety_info = self.safety_module(
                    state_tensor, command, return_info=True
                )
            else:
                filtered_command = command
                safety_info = {"is_safe": True}
            
            # Get action
            action = self.policy.get_action(
                state_tensor, filtered_command,
                deterministic=deterministic
            )
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Environment step
            action_scalar = action.squeeze() if action.ndim > 0 else action
            next_state, reward, terminated, truncated, info = env.step(action_scalar)
            
            total_return += reward
            
            if record_trajectory:
                trajectory.append({
                    "state": state,
                    "action": action_scalar,
                    "reward": reward,
                    "command": command.cpu().numpy().squeeze(),
                    "filtered_command": filtered_command.cpu().numpy().squeeze() if torch.is_tensor(filtered_command) else filtered_command,
                    "is_safe": safety_info.get("is_safe", True),
                })
            
            done = terminated or truncated
            
            if done:
                if info.get("fell_in_trap", False) or (terminated and reward < -5):
                    termination_type = "trap"
                elif info.get("reached_goal", False) or (terminated and reward > 0):
                    termination_type = "goal"
                else:
                    termination_type = "terminated"
                break
            
            # Update state and command
            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Decrement horizon and adjust return
            command[0, 0] = max(1, command[0, 0] - 1)
            command[0, 1] = command[0, 1] - reward
        
        return {
            "total_return": total_return,
            "length": step + 1,
            "termination_type": termination_type,
            "trajectory": trajectory,
        }
    
    def sweep_commands(
        self,
        horizon_range: Tuple[int, int, int],
        return_range: Tuple[float, float, int],
        n_episodes_per_command: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate over a grid of commands.
        
        Args:
            horizon_range: (min, max, n_steps) for horizon
            return_range: (min, max, n_steps) for return
            n_episodes_per_command: Episodes per command
            
        Returns:
            Grid of results
        """
        horizons = np.linspace(*horizon_range[:2], horizon_range[2])
        returns = np.linspace(*return_range[:2], return_range[2])
        
        commands = [(h, r) for h in horizons for r in returns]
        
        results = self.evaluate(
            commands=commands,
            n_episodes=n_episodes_per_command,
            **kwargs,
        )
        
        # Reshape to grid
        n_horizons = len(horizons)
        n_returns = len(returns)
        
        return_grid = np.array(results["raw_returns"]).reshape(n_horizons, n_returns, -1)
        
        results["grid"] = {
            "horizons": horizons,
            "returns": returns,
            "mean_returns": return_grid.mean(axis=-1),
            "std_returns": return_grid.std(axis=-1),
        }
        
        return results


class SafetyEvaluator:
    """
    Evaluator specifically for safety module analysis.
    
    Measures OOD detection, projection quality, and comparison
    between different safety methods.
    
    Args:
        safety_module: Safety module to evaluate
        dataset: Dataset for in-distribution reference
        device: Device for computation
    """
    
    def __init__(
        self,
        safety_module: BaseSafetyModule,
        device: str = "cuda",
    ):
        self.safety_module = safety_module
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.safety_module.to(self.device)
        self.safety_module.eval()
    
    @torch.no_grad()
    def evaluate_ood_detection(
        self,
        id_states: torch.Tensor,
        id_commands: torch.Tensor,
        ood_commands: torch.Tensor,
        ood_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance.
        
        Args:
            id_states: In-distribution states
            id_commands: In-distribution commands
            ood_commands: Out-of-distribution commands
            ood_states: States for OOD commands (default: same as ID)
            
        Returns:
            OOD detection metrics
        """
        if ood_states is None:
            ood_states = id_states[:len(ood_commands)]
        
        id_states = id_states.to(self.device)
        id_commands = id_commands.to(self.device)
        ood_commands = ood_commands.to(self.device)
        ood_states = ood_states.to(self.device)
        
        # Get safety scores
        id_scores = self.safety_module.get_safety_score(id_states, id_commands)
        ood_scores = self.safety_module.get_safety_score(ood_states, ood_commands)
        
        return compute_ood_metrics(id_scores, ood_scores)
    
    @torch.no_grad()
    def evaluate_projection(
        self,
        states: torch.Tensor,
        commands: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate command projection quality.
        
        Args:
            states: States to condition on
            commands: Commands to project
            
        Returns:
            Projection metrics
        """
        states = states.to(self.device)
        commands = commands.to(self.device)
        
        # Get safety info including projection
        original_safe = self.safety_module.is_safe(states, commands)
        projected = self.safety_module.project(states, commands)
        projected_safe = self.safety_module.is_safe(states, projected)
        
        # Distances
        projection_distance = torch.norm(commands - projected, dim=-1)
        
        # Safety improvement
        original_scores = self.safety_module.get_safety_score(states, commands)
        projected_scores = self.safety_module.get_safety_score(states, projected)
        
        return {
            "original_safe_rate": original_safe.float().mean().item(),
            "projected_safe_rate": projected_safe.float().mean().item(),
            "mean_projection_distance": projection_distance.mean().item(),
            "max_projection_distance": projection_distance.max().item(),
            "score_improvement": (projected_scores - original_scores).mean().item(),
            "unsafe_to_safe_rate": (projected_safe & ~original_safe).float().mean().item(),
        }
    
    @torch.no_grad()
    def evaluate_sampling(
        self,
        states: torch.Tensor,
        n_samples: int = 100,
        ground_truth_commands: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate command sampling quality.
        
        Args:
            states: States to sample from
            n_samples: Samples per state
            ground_truth_commands: Optional ground truth for comparison
            
        Returns:
            Sampling metrics
        """
        states = states.to(self.device)
        
        samples = self.safety_module.sample_commands(states, n_samples)
        
        # Sample statistics
        sample_mean = samples.mean(dim=1)
        sample_std = samples.std(dim=1)
        
        # Check if samples are safe
        flat_samples = samples.reshape(-1, samples.shape[-1])
        flat_states = states.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, states.shape[-1])
        
        samples_safe = self.safety_module.is_safe(flat_states, flat_samples)
        
        metrics = {
            "sample_mean_horizon": sample_mean[:, 0].mean().item(),
            "sample_mean_return": sample_mean[:, 1].mean().item(),
            "sample_std_horizon": sample_std[:, 0].mean().item(),
            "sample_std_return": sample_std[:, 1].mean().item(),
            "samples_safe_rate": samples_safe.float().mean().item(),
        }
        
        # Compare to ground truth if provided
        if ground_truth_commands is not None:
            ground_truth_commands = ground_truth_commands.to(self.device)
            gt_mean = ground_truth_commands.mean(dim=0)
            gt_std = ground_truth_commands.std(dim=0)
            
            metrics["mean_mismatch"] = torch.norm(sample_mean.mean(dim=0) - gt_mean).item()
            metrics["std_mismatch"] = torch.norm(sample_std.mean(dim=0) - gt_std).item()
        
        return metrics
    
    def compare_methods(
        self,
        methods: Dict[str, BaseSafetyModule],
        states: torch.Tensor,
        id_commands: torch.Tensor,
        ood_commands: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple safety methods.
        
        Args:
            methods: Dictionary of method_name -> safety_module
            states: Test states
            id_commands: In-distribution commands
            ood_commands: Out-of-distribution commands
            
        Returns:
            Comparison results for each method
        """
        results = {}
        
        for name, module in methods.items():
            self.safety_module = module
            self.safety_module.to(self.device)
            self.safety_module.eval()
            
            ood_metrics = self.evaluate_ood_detection(
                states, id_commands, ood_commands
            )
            
            projection_metrics = self.evaluate_projection(
                states, ood_commands
            )
            
            sampling_metrics = self.evaluate_sampling(
                states, n_samples=100, ground_truth_commands=id_commands
            )
            
            results[name] = {
                **{f"ood_{k}": v for k, v in ood_metrics.items()},
                **{f"proj_{k}": v for k, v in projection_metrics.items()},
                **{f"sample_{k}": v for k, v in sampling_metrics.items()},
            }
        
        return results
