=============
Models Module
=============

.. module:: flowshield.models
   :synopsis: Neural network models for UDRL and safety shields

This module contains all neural network architectures for policy learning and 
density estimation.

UDRL Agent
----------

.. class:: UDRLAgent

   Upside-Down Reinforcement Learning goal-conditioned policy.
   
   The agent learns to produce actions conditioned on state and command:
   
   .. math::
   
      \pi_\theta(a | s, g) = \mathcal{N}(\mu_\theta(s, g), \sigma_\theta(s, g))
   
   where :math:`g = (H, R)` is the hindsight command.
   
   .. method:: __init__(state_dim: int, action_dim: int, command_dim: int = 2, hidden_dim: int = 256, n_layers: int = 4, activation: str = "silu")
   
      :param state_dim: Dimension of state space
      :param action_dim: Dimension of action space  
      :param command_dim: Dimension of command (default 2: horizon, return)
      :param hidden_dim: Width of hidden layers
      :param n_layers: Number of hidden layers
      :param activation: Activation function ("relu", "silu", "gelu")
      
   .. method:: forward(state: torch.Tensor, command: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
   
      Compute action distribution parameters.
      
      :param state: State tensor [batch, state_dim]
      :param command: Command tensor [batch, 2]
      :return: (mean, log_std) action distribution parameters
      
   .. method:: sample(state: torch.Tensor, command: torch.Tensor, deterministic: bool = False) -> torch.Tensor
   
      Sample action from policy.
      
      :param state: State tensor
      :param command: Command tensor
      :param deterministic: If True, return mean action
      :return: Action tensor [batch, action_dim]
      
   .. method:: log_prob(state: torch.Tensor, command: torch.Tensor, action: torch.Tensor) -> torch.Tensor
   
      Compute log probability of action.
      
      :param state: State tensor
      :param command: Command tensor
      :param action: Action tensor
      :return: Log probability [batch]
      
   .. method:: save(path: str) -> None
   
      Save model to file.
      
   .. method:: load(path: str) -> UDRLAgent
      :classmethod:
      
      Load model from file.
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.models import UDRLAgent
      
      agent = UDRLAgent(state_dim=8, action_dim=2)
      
      # Training
      state = torch.randn(32, 8)
      command = torch.tensor([[100, 50]] * 32).float()
      action_target = torch.randn(32, 2)
      
      mean, log_std = agent(state, command)
      loss = agent.compute_loss(state, command, action_target)
      
      # Inference
      action = agent.sample(state, command, deterministic=True)

Base Shield
-----------

.. class:: BaseShield

   Abstract base class for safety shields.
   
   All shields must implement these methods:
   
   .. method:: compute_loss(state: torch.Tensor, command: torch.Tensor) -> torch.Tensor
      :abstractmethod:
      
      Compute training loss.
      
   .. method:: log_prob(state: torch.Tensor, command: torch.Tensor) -> torch.Tensor
      :abstractmethod:
      
      Compute log probability of command given state.
      
   .. method:: is_ood(state: torch.Tensor, command: torch.Tensor, threshold: float) -> torch.Tensor
      :abstractmethod:
      
      Check if command is out-of-distribution.
      
   .. method:: project(state: torch.Tensor, command: torch.Tensor) -> torch.Tensor
      :abstractmethod:
      
      Project OOD command to safe region.

Quantile Shield
---------------

.. class:: QuantileShield(BaseShield)

   Quantile regression-based safety shield.
   
   Estimates conservative return bounds using pinball loss:
   
   .. math::
   
      \mathcal{L}_\tau(\hat{R}, R) = (R - \hat{R}) \cdot (\tau - \mathbb{1}[R < \hat{R}])
   
   .. method:: __init__(state_dim: int, command_dim: int = 2, hidden_dim: int = 256, n_layers: int = 4, tau: float = 0.9)
   
      :param tau: Quantile level (0.9 = 90th percentile)
      
   .. method:: get_quantile(state: torch.Tensor) -> torch.Tensor
   
      Predict quantile bounds for state.
      
      :param state: State tensor
      :return: Quantile bounds [batch, 2]

Diffusion Shield
----------------

.. class:: DiffusionShield(BaseShield)

   DDPM-based density estimation shield.
   
   Uses denoising score matching to learn :math:`p(g|s)`:
   
   .. math::
   
      \mathcal{L} = \mathbb{E}_{t, \epsilon} \| \epsilon - \epsilon_\theta(g_t, t, s) \|^2
   
   .. method:: __init__(state_dim: int, command_dim: int = 2, hidden_dim: int = 256, n_timesteps: int = 1000, beta_schedule: str = "linear")
   
      :param n_timesteps: Number of diffusion steps
      :param beta_schedule: Noise schedule ("linear", "cosine", "quadratic")
      
   .. method:: sample(state: torch.Tensor, n_samples: int = 1) -> torch.Tensor
   
      Sample commands from learned distribution.
      
      :param state: Conditioning state
      :param n_samples: Number of samples
      :return: Sampled commands [n_samples, 2]
      
   .. method:: denoise(noisy_command: torch.Tensor, state: torch.Tensor) -> torch.Tensor
   
      Denoise command using reverse process.

Flow Matching Shield
--------------------

.. class:: FlowMatchingShield(BaseShield)

   OT-CFM based density estimation shield (recommended).
   
   Learns velocity field for continuous normalizing flow:
   
   .. math::
   
      \frac{dg}{dt} = v_\theta(g, t, s)
   
   .. method:: __init__(state_dim: int, command_dim: int = 2, hidden_dim: int = 256, n_layers: int = 4, time_embed_dim: int = 128, sigma_min: float = 1e-4)
   
      :param time_embed_dim: Dimension of time embedding
      :param sigma_min: Minimum noise for stability
      
   .. method:: velocity(g: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor
   
      Compute velocity at point g, time t, conditioned on state.
      
      :param g: Command position [batch, 2]
      :param t: Time in [0, 1]
      :param state: Conditioning state
      :return: Velocity vector [batch, 2]
      
   .. method:: log_prob(state: torch.Tensor, command: torch.Tensor, n_steps: int = 50) -> torch.Tensor
   
      Compute exact log probability via ODE integration.
      
      :param state: Conditioning state
      :param command: Command to evaluate
      :param n_steps: ODE integration steps
      :return: Log probability [batch]
      
   .. method:: sample(state: torch.Tensor, n_samples: int = 1, n_steps: int = 50) -> torch.Tensor
   
      Sample from flow by integrating ODE.
      
      :param state: Conditioning state
      :param n_samples: Number of samples
      :param n_steps: Integration steps
      :return: Samples [n_samples, 2]
      
   .. method:: project(state: torch.Tensor, command: torch.Tensor, n_steps: int = 20, lr: float = 0.1) -> torch.Tensor
   
      Project command via gradient ascent on log probability.
      
      :param state: Conditioning state
      :param command: OOD command to project
      :param n_steps: Gradient steps
      :param lr: Learning rate
      :return: Projected command [batch, 2]
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.models import FlowMatchingShield
      
      shield = FlowMatchingShield(state_dim=8)
      
      # Training
      loss = shield.compute_loss(states, commands)
      
      # OOD Detection
      log_p = shield.log_prob(state, command)
      is_ood = log_p < threshold
      
      # Projection
      if is_ood:
          safe_command = shield.project(state, command)

Network Components
------------------

.. class:: MLP

   Multi-layer perceptron with residual connections.
   
   .. method:: __init__(input_dim: int, output_dim: int, hidden_dim: int = 256, n_layers: int = 4, activation: str = "silu")

.. class:: SinusoidalEmbedding

   Sinusoidal positional embedding for time conditioning.
   
   .. math::
   
      \text{PE}(t, 2i) = \sin(t / 10000^{2i/d})
   
   .. method:: __init__(embed_dim: int)
   
   .. method:: forward(t: torch.Tensor) -> torch.Tensor
   
      :param t: Time values [batch]
      :return: Embeddings [batch, embed_dim]

.. class:: FourierFeatures

   Random Fourier features for command encoding.
   
   .. math::
   
      \phi(g) = [\cos(2\pi B g), \sin(2\pi B g)]
   
   .. method:: __init__(input_dim: int, embed_dim: int, sigma: float = 1.0)
   
   .. method:: forward(x: torch.Tensor) -> torch.Tensor
