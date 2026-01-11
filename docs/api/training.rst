===============
Training Module
===============

.. module:: flowshield.training
   :synopsis: Training loops, datasets, and utilities

This module provides training infrastructure for UDRL agents and safety shields.

Trainer Class
-------------

.. class:: Trainer

   Main training orchestrator for UDRL and shields.
   
   .. method:: __init__(config: DictConfig)
   
      :param config: Hydra configuration object
      
   .. method:: train_udrl(dataset: Dataset, n_epochs: int = 200) -> UDRLAgent
   
      Train UDRL agent on collected data.
      
      :param dataset: Trajectory dataset
      :param n_epochs: Training epochs
      :return: Trained agent
      
   .. method:: train_shield(dataset: Dataset, shield_type: str, n_epochs: int = 100) -> BaseShield
   
      Train safety shield on (state, command) pairs.
      
      :param dataset: Trajectory dataset
      :param shield_type: "quantile", "diffusion", or "flow_matching"
      :param n_epochs: Training epochs
      :return: Trained shield
      
   .. method:: train(dataset: Dataset, train_udrl: bool = True, train_shield: bool = True) -> Tuple[UDRLAgent, BaseShield]
   
      Complete training pipeline.
      
      :param dataset: Trajectory dataset
      :param train_udrl: Whether to train UDRL
      :param train_shield: Whether to train shield
      :return: (agent, shield) tuple
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.training import Trainer
      from omegaconf import OmegaConf
      
      config = OmegaConf.load("configs/train.yaml")
      trainer = Trainer(config)
      
      agent, shield = trainer.train(dataset)

Dataset Classes
---------------

.. class:: TrajectoryDataset

   Dataset of collected trajectories for UDRL training.
   
   .. method:: __init__(data_path: str)
   
      :param data_path: Path to collected data (.npz or .pkl)
      
   .. attribute:: states
      :type: np.ndarray
      
      All states [N, state_dim]
      
   .. attribute:: actions
      :type: np.ndarray
      
      All actions [N, action_dim]
      
   .. attribute:: rewards
      :type: np.ndarray
      
      All rewards [N]
      
   .. attribute:: dones
      :type: np.ndarray
      
      Episode termination flags [N]
      
   .. method:: get_hindsight_batch(batch_size: int) -> Tuple[torch.Tensor, ...]
   
      Sample batch with hindsight commands.
      
      :param batch_size: Number of samples
      :return: (states, actions, commands) tensors
      
   .. method:: get_command_pairs(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]
   
      Sample (state, command) pairs for shield training.
      
      :param batch_size: Number of samples
      :return: (states, commands) tensors
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.training import TrajectoryDataset
      
      dataset = TrajectoryDataset("data/lunarlander_10k.npz")
      print(f"Loaded {len(dataset)} transitions")
      
      states, actions, commands = dataset.get_hindsight_batch(256)

.. class:: CommandDataset

   Dataset of (state, command) pairs for shield training.
   
   .. method:: __init__(trajectory_dataset: TrajectoryDataset)
   
   .. method:: __len__() -> int
   
   .. method:: __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]
   
      :return: (state, command) pair

Data Collection
---------------

.. class:: DataCollector

   Collect trajectories using behavioral policy.
   
   .. method:: __init__(env: gym.Env, policy: Optional[Callable] = None)
   
      :param env: Environment to collect from
      :param policy: Behavioral policy (None = random)
      
   .. method:: collect(n_episodes: int, render: bool = False) -> TrajectoryDataset
   
      Collect specified number of episodes.
      
      :param n_episodes: Episodes to collect
      :param render: Visualize collection
      :return: Collected dataset
      
   .. method:: save(path: str) -> None
   
      Save collected data to file.
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.training import DataCollector
      from flowshield.envs import make_env
      
      env = make_env("lunarlander")
      collector = DataCollector(env)
      
      dataset = collector.collect(n_episodes=10000)
      collector.save("data/lunarlander_10k.npz")

Training Utilities
------------------

.. function:: compute_hindsight_commands(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray

   Compute hindsight commands from trajectory.
   
   For each timestep, computes remaining horizon and return:
   
   .. math::
   
      g_t = (H - t, \sum_{k=t}^{H} r_k)
   
   :param rewards: Reward sequence
   :param dones: Done flags
   :return: Commands [N, 2]

.. function:: sample_future_commands(states: np.ndarray, rewards: np.ndarray, dones: np.ndarray, batch_indices: np.ndarray) -> np.ndarray

   Sample future commands for training.
   
   For each sampled state, sample a random future timestep and compute 
   the command to reach that point.
   
   :param states: All states
   :param rewards: All rewards
   :param dones: Done flags
   :param batch_indices: Indices of sampled states
   :return: Sampled commands [batch, 2]

.. class:: EarlyStopping

   Early stopping based on validation loss.
   
   .. method:: __init__(patience: int = 10, min_delta: float = 1e-4)
   
   .. method:: __call__(val_loss: float) -> bool
   
      :param val_loss: Current validation loss
      :return: True if should stop

.. class:: LearningRateScheduler

   Learning rate scheduling wrapper.
   
   .. method:: __init__(optimizer: torch.optim.Optimizer, scheduler_type: str = "cosine", **kwargs)
   
   .. method:: step(epoch: int) -> None

Logging
-------

.. class:: Logger

   Training logger with TensorBoard support.
   
   .. method:: __init__(log_dir: str, use_wandb: bool = False)
   
   .. method:: log_scalar(name: str, value: float, step: int) -> None
   
   .. method:: log_histogram(name: str, values: np.ndarray, step: int) -> None
   
   .. method:: log_config(config: dict) -> None
   
   .. method:: close() -> None

Checkpointing
-------------

.. function:: save_checkpoint(path: str, agent: UDRLAgent, shield: BaseShield, optimizer: torch.optim.Optimizer, epoch: int) -> None

   Save training checkpoint.

.. function:: load_checkpoint(path: str) -> Dict[str, Any]

   Load training checkpoint.
   
   :return: Dictionary with model, optimizer, and epoch

Configuration
-------------

Training is configured via Hydra YAML files:

.. code-block:: yaml

   # configs/train.yaml
   defaults:
     - env: lunarlander
     - model: udrl
     - shield: flow_matching
   
   # Training
   seed: 42
   n_epochs: 200
   batch_size: 256
   learning_rate: 1e-4
   
   # Logging
   log_dir: "logs/"
   checkpoint_every: 10
   
   # Hardware
   device: "cuda"
   num_workers: 4
