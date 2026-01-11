"""
Data Collectors for Offline RL.

Provides utilities for collecting trajectories from environments
using various policies (random, expert, mixed).
"""

from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .trajectory import Trajectory, TrajectoryBuffer


class DataCollector:
    """
    Base collector for gathering trajectories from an environment.
    
    Args:
        env: Gymnasium environment
        max_episode_steps: Maximum steps per episode (overrides env setting)
        
    Example:
        >>> env = make_env("lunarlander")
        >>> collector = DataCollector(env)
        >>> trajectories = collector.collect_random(n_episodes=100)
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
    ):
        self.env = env
        self.max_episode_steps = max_episode_steps
    
    def collect_random(
        self,
        n_episodes: int,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect trajectories using random policy.
        
        Args:
            n_episodes: Number of episodes to collect
            seed: Random seed
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer with collected trajectories
        """
        if seed is not None:
            np.random.seed(seed)
        
        buffer = TrajectoryBuffer()
        iterator = range(n_episodes)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting random trajectories")
        
        for _ in iterator:
            trajectory = self._collect_episode(policy=None)
            buffer.add(trajectory)
        
        return buffer
    
    def collect_with_policy(
        self,
        policy: Callable[[np.ndarray], Any],
        n_episodes: int,
        deterministic: bool = False,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect trajectories using a given policy.
        
        Args:
            policy: Function mapping state -> action
            n_episodes: Number of episodes to collect
            deterministic: Whether to use deterministic actions
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer with collected trajectories
        """
        buffer = TrajectoryBuffer()
        iterator = range(n_episodes)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting with policy")
        
        for _ in iterator:
            trajectory = self._collect_episode(
                policy=policy,
                deterministic=deterministic,
            )
            buffer.add(trajectory)
        
        return buffer
    
    def collect_mixed(
        self,
        expert_policy: Callable[[np.ndarray], Any],
        n_episodes: int,
        expert_ratio: float = 0.2,
        epsilon: float = 0.0,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect a mix of expert and random trajectories.
        
        Args:
            expert_policy: Expert policy function
            n_episodes: Total number of episodes
            expert_ratio: Fraction of expert trajectories
            epsilon: Epsilon-greedy noise for expert (0 = pure expert)
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer with mixed trajectories
        """
        buffer = TrajectoryBuffer()
        n_expert = int(n_episodes * expert_ratio)
        n_random = n_episodes - n_expert
        
        iterator = range(n_episodes)
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting mixed trajectories")
        
        for i in iterator:
            if i < n_expert:
                # Expert trajectory (possibly with epsilon noise)
                if epsilon > 0:
                    policy = lambda s: (
                        expert_policy(s) 
                        if np.random.random() > epsilon 
                        else self.env.action_space.sample()
                    )
                else:
                    policy = expert_policy
            else:
                # Random trajectory
                policy = None
            
            trajectory = self._collect_episode(policy=policy)
            buffer.add(trajectory)
        
        return buffer
    
    def _collect_episode(
        self,
        policy: Optional[Callable[[np.ndarray], Any]] = None,
        deterministic: bool = False,
    ) -> Trajectory:
        """
        Collect a single episode.
        
        Args:
            policy: Policy function (None for random)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Trajectory object
        """
        states = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        obs, info = self.env.reset()
        states.append(obs.copy())
        
        step = 0
        done = False
        
        while not done:
            # Get action
            if policy is None:
                action = self.env.action_space.sample()
            else:
                action = policy(obs)
            
            # Step
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            if not done:
                states.append(obs.copy())
            
            step += 1
            
            # Check max steps
            if self.max_episode_steps and step >= self.max_episode_steps:
                done = True
                dones[-1] = True
        
        return Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=infos,
        )


class ExpertCollector(DataCollector):
    """
    Collector that uses environment-specific expert policy.
    
    For continuous environments, requires explicit expert policy.
    
    Args:
        env: Gymnasium environment
        expert_policy: Optional explicit expert policy
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert_policy: Optional[Callable[[np.ndarray], Any]] = None,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env, max_episode_steps)
        
        # Try to get optimal policy from environment
        if expert_policy is not None:
            self.expert_policy = expert_policy
        elif hasattr(env, 'get_optimal_action'):
            self.expert_policy = lambda s: env.get_optimal_action()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_optimal_action'):
            self.expert_policy = lambda s: env.unwrapped.get_optimal_action()
        else:
            raise ValueError(
                "No expert policy provided and environment doesn't have "
                "get_optimal_action method. Please provide expert_policy."
            )
    
    def collect(
        self,
        n_episodes: int,
        epsilon: float = 0.0,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect expert trajectories.
        
        Args:
            n_episodes: Number of episodes
            epsilon: Epsilon-greedy noise (0 = pure expert)
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer with expert trajectories
        """
        if epsilon > 0:
            policy = lambda s: (
                self.expert_policy(s)
                if np.random.random() > epsilon
                else self.env.action_space.sample()
            )
        else:
            policy = self.expert_policy
        
        return self.collect_with_policy(
            policy=policy,
            n_episodes=n_episodes,
            show_progress=show_progress,
        )


class MixedCollector(DataCollector):
    """
    Collector for mixed-quality datasets.
    
    Creates datasets with varying trajectory quality levels,
    useful for testing offline RL algorithms.
    
    Args:
        env: Gymnasium environment
        expert_policy: Expert policy function
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert_policy: Optional[Callable[[np.ndarray], Any]] = None,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env, max_episode_steps)
        
        # Get expert policy
        if expert_policy is not None:
            self.expert_policy = expert_policy
        elif hasattr(env, 'get_optimal_action'):
            self.expert_policy = lambda s: env.get_optimal_action()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_optimal_action'):
            self.expert_policy = lambda s: env.unwrapped.get_optimal_action()
        else:
            self.expert_policy = None
    
    def collect(
        self,
        n_episodes: int,
        quality_dist: Optional[Dict[str, float]] = None,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect mixed-quality trajectories.
        
        Args:
            n_episodes: Total number of episodes
            quality_dist: Distribution of quality levels
                         {"expert": 0.2, "medium": 0.3, "random": 0.5}
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer with mixed trajectories
        """
        if quality_dist is None:
            quality_dist = {"expert": 0.2, "medium": 0.3, "random": 0.5}
        
        # Validate distribution
        total = sum(quality_dist.values())
        quality_dist = {k: v / total for k, v in quality_dist.items()}
        
        buffer = TrajectoryBuffer()
        iterator = range(n_episodes)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Collecting mixed quality")
        
        for _ in iterator:
            # Sample quality level
            quality = np.random.choice(
                list(quality_dist.keys()),
                p=list(quality_dist.values()),
            )
            
            # Create policy based on quality
            if quality == "expert":
                policy = self.expert_policy
                epsilon = 0.0
            elif quality == "medium":
                policy = self.expert_policy
                epsilon = 0.3
            elif quality == "noisy":
                policy = self.expert_policy
                epsilon = 0.5
            else:  # random
                policy = None
                epsilon = 0.0
            
            # Apply epsilon-greedy if needed
            if policy is not None and epsilon > 0:
                base_policy = policy
                policy = lambda s, bp=base_policy, e=epsilon: (
                    bp(s) if np.random.random() > e else self.env.action_space.sample()
                )
            
            trajectory = self._collect_episode(policy=policy)
            buffer.add(trajectory)
        
        return buffer
    
    def collect_replay_style(
        self,
        n_episodes: int,
        phases: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
    ) -> TrajectoryBuffer:
        """
        Collect trajectories simulating replay buffer growth.
        
        Starts with random policy and improves over time,
        mimicking how a replay buffer would look during training.
        
        Args:
            n_episodes: Total number of episodes
            phases: List of collection phases with epsilon values
            show_progress: Whether to show progress bar
            
        Returns:
            TrajectoryBuffer simulating replay buffer
        """
        if phases is None:
            # Default: gradually improve from random to near-expert
            phases = [
                {"fraction": 0.3, "epsilon": 1.0},   # Pure random
                {"fraction": 0.3, "epsilon": 0.5},   # Half-random
                {"fraction": 0.2, "epsilon": 0.2},   # Mostly expert
                {"fraction": 0.2, "epsilon": 0.05},  # Near-expert
            ]
        
        buffer = TrajectoryBuffer()
        
        total_collected = 0
        for phase in phases:
            n_phase = int(n_episodes * phase["fraction"])
            epsilon = phase["epsilon"]
            
            iterator = range(n_phase)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc=f"Phase (ε={epsilon:.2f})",
                )
            
            for _ in iterator:
                if epsilon >= 1.0 or self.expert_policy is None:
                    policy = None
                else:
                    policy = lambda s, e=epsilon: (
                        self.expert_policy(s)
                        if np.random.random() > e
                        else self.env.action_space.sample()
                    )
                
                trajectory = self._collect_episode(policy=policy)
                buffer.add(trajectory)
                total_collected += 1
        
        return buffer
