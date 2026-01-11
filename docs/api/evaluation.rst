=================
Evaluation Module
=================

.. module:: flowshield.evaluation
   :synopsis: Evaluation functions and metrics

This module provides evaluation utilities for assessing agent and shield performance.

Main Evaluation
---------------

.. function:: evaluate_agent(env: gym.Env, agent: UDRLAgent, shield: Optional[BaseShield] = None, commands: List[Tuple[float, float]] = None, n_episodes: int = 100, threshold: float = -6.0, render: bool = False) -> Dict[str, float]

   Evaluate agent with optional safety shield.
   
   :param env: Environment to evaluate in
   :param agent: Trained UDRL agent
   :param shield: Optional safety shield for OOD protection
   :param commands: List of (horizon, return) commands to test
   :param n_episodes: Number of evaluation episodes
   :param threshold: OOD detection threshold
   :param render: Whether to visualize
   :return: Dictionary of metrics
   
   **Returned metrics:**
   
   - ``mean_return``: Average episode return
   - ``std_return``: Standard deviation of returns
   - ``success_rate``: Fraction of successful episodes
   - ``crash_rate``: Fraction of failed episodes
   - ``mean_length``: Average episode length
   - ``shield_activations``: Number of times shield projected
   - ``ood_detections``: Number of OOD commands detected
   
   **Example:**
   
   .. code-block:: python
   
      from flowshield.evaluation import evaluate_agent
      from flowshield.envs import make_env
      from flowshield.models import UDRLAgent, FlowMatchingShield
      
      env = make_env("lunarlander")
      agent = UDRLAgent.load("agent.pt")
      shield = FlowMatchingShield.load("shield.pt")
      
      # Test with various commands
      commands = [
          (100, 50),   # In-distribution
          (50, 200),   # OOD: high return
          (20, 150),   # OOD: short horizon
      ]
      
      metrics = evaluate_agent(
          env, agent, shield,
          commands=commands,
          n_episodes=100
      )
      
      print(f"Mean return: {metrics['mean_return']:.1f}")
      print(f"Crash rate: {metrics['crash_rate']:.1%}")

Episode Evaluation
------------------

.. function:: run_episode(env: gym.Env, agent: UDRLAgent, command: Tuple[float, float], shield: Optional[BaseShield] = None, threshold: float = -6.0, render: bool = False) -> Dict[str, Any]

   Run single episode and collect detailed statistics.
   
   :param env: Environment
   :param agent: UDRL agent
   :param command: Target (horizon, return)
   :param shield: Optional safety shield
   :param threshold: OOD threshold
   :param render: Visualize episode
   :return: Episode statistics
   
   **Returned dictionary:**
   
   .. code-block:: python
   
      {
          'return': float,           # Total episode return
          'length': int,             # Episode length
          'success': bool,           # Whether successful
          'crash': bool,             # Whether crashed
          'states': np.ndarray,      # State trajectory
          'actions': np.ndarray,     # Action trajectory
          'rewards': np.ndarray,     # Reward sequence
          'commands': List[Tuple],   # Commands used (may change if projected)
          'ood_flags': List[bool],   # OOD detection per step
          'log_probs': List[float],  # Shield log-probs per step
      }

Shield Evaluation
-----------------

.. function:: evaluate_shield(shield: BaseShield, test_dataset: TrajectoryDataset, threshold: float = -6.0) -> Dict[str, float]

   Evaluate shield detection and projection quality.
   
   :param shield: Trained safety shield
   :param test_dataset: Test dataset with known labels
   :param threshold: Detection threshold
   :return: Shield performance metrics
   
   **Metrics:**
   
   - ``detection_rate``: True positive rate for OOD
   - ``false_positive_rate``: False alarm rate
   - ``f1_score``: Harmonic mean of precision/recall
   - ``auc_roc``: Area under ROC curve
   - ``calibration_error``: Probability calibration error
   
   **Example:**
   
   .. code-block:: python
   
      from flowshield.evaluation import evaluate_shield
      
      metrics = evaluate_shield(shield, test_dataset)
      print(f"Detection rate: {metrics['detection_rate']:.1%}")
      print(f"AUC-ROC: {metrics['auc_roc']:.3f}")

.. function:: evaluate_projection(shield: BaseShield, ood_commands: np.ndarray, states: np.ndarray) -> Dict[str, float]

   Evaluate projection quality.
   
   :param shield: Safety shield
   :param ood_commands: Known OOD commands
   :param states: Corresponding states
   :return: Projection metrics
   
   **Metrics:**
   
   - ``mean_distance``: Average distance from original
   - ``mean_log_prob``: Average log-prob of projected
   - ``projection_success``: Fraction projected to in-distribution

Metrics Classes
---------------

.. class:: MetricsTracker

   Track and aggregate evaluation metrics.
   
   .. method:: __init__()
   
   .. method:: update(episode_result: Dict[str, Any]) -> None
   
      Add episode result to tracker.
      
   .. method:: compute() -> Dict[str, float]
   
      Compute aggregate statistics.
      
   .. method:: reset() -> None
   
      Clear tracked metrics.
      
   **Example:**
   
   .. code-block:: python
   
      tracker = MetricsTracker()
      
      for command in commands:
          for _ in range(n_episodes):
              result = run_episode(env, agent, command, shield)
              tracker.update(result)
      
      metrics = tracker.compute()

.. class:: ConfidenceInterval

   Compute confidence intervals via bootstrap.
   
   .. method:: __init__(confidence: float = 0.95, n_bootstrap: int = 1000)
   
   .. method:: compute(values: np.ndarray) -> Tuple[float, float, float]
   
      :param values: Sample values
      :return: (mean, lower_bound, upper_bound)

Statistical Tests
-----------------

.. function:: paired_t_test(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]

   Paired t-test for comparing methods.
   
   :param values_a: Results from method A
   :param values_b: Results from method B
   :return: (t_statistic, p_value)

.. function:: wilcoxon_test(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]

   Non-parametric Wilcoxon signed-rank test.
   
   :param values_a: Results from method A
   :param values_b: Results from method B
   :return: (statistic, p_value)

Visualization
-------------

.. function:: plot_returns(metrics: Dict[str, List[float]], save_path: Optional[str] = None) -> None

   Plot return distribution across methods.
   
   :param metrics: Dictionary mapping method name to returns
   :param save_path: Optional path to save figure

.. function:: plot_ood_detection(shield: BaseShield, states: np.ndarray, commands: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None) -> None

   Visualize OOD detection decision boundary.
   
   :param shield: Trained shield
   :param states: Test states
   :param commands: Test commands
   :param labels: True OOD labels
   :param save_path: Optional save path

.. function:: plot_projection(original: np.ndarray, projected: np.ndarray, density: np.ndarray, save_path: Optional[str] = None) -> None

   Visualize command projection.
   
   :param original: Original OOD commands
   :param projected: Projected safe commands
   :param density: Density heatmap
   :param save_path: Optional save path

Command Line Interface
----------------------

Evaluation can be run from command line:

.. code-block:: bash

   # Basic evaluation
   python scripts/evaluate.py \
       env=lunarlander \
       agent_path=checkpoints/agent.pt \
       shield_path=checkpoints/shield.pt \
       n_episodes=1000
   
   # With custom commands
   python scripts/evaluate.py \
       env=lunarlander \
       commands="[[100,50],[50,200],[20,150]]" \
       threshold=-6.0
   
   # Render visualization
   python scripts/evaluate.py \
       env=lunarlander \
       render=true \
       n_episodes=10
