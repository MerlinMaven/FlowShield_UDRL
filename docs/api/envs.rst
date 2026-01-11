===================
Environments Module
===================

.. module:: flowshield.envs
   :synopsis: Environment wrappers and factory functions

This module provides environment creation and wrappers for continuous control tasks.

Factory Functions
-----------------

.. function:: make_env(env_name: str, **kwargs) -> gym.Env

   Create an environment by name.
   
   :param env_name: Environment identifier ("lunarlander", "highway")
   :param kwargs: Additional environment configuration
   :return: Wrapped gymnasium environment
   
   **Example:**
   
   .. code-block:: python
   
      env = make_env("lunarlander")
      state, _ = env.reset()
      action = env.action_space.sample()
      next_state, reward, done, truncated, info = env.step(action)

.. function:: register_env(name: str, wrapper_cls: Type[ContinuousEnvWrapper]) -> None

   Register a custom environment wrapper.
   
   :param name: Unique environment name
   :param wrapper_cls: Wrapper class to instantiate
   
   **Example:**
   
   .. code-block:: python
   
      register_env("custom", MyCustomEnvWrapper)
      env = make_env("custom")

Base Wrapper
------------

.. class:: ContinuousEnvWrapper

   Base class for continuous control environment wrappers.
   
   .. attribute:: state_dim
      :type: int
      
      Dimension of observation space.
      
   .. attribute:: action_dim
      :type: int
      
      Dimension of action space.
      
   .. attribute:: max_episode_steps
      :type: int
      
      Maximum steps per episode.

   .. method:: reset() -> np.ndarray
   
      Reset environment and return initial observation.
      
      :return: Initial state vector
      
   .. method:: step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]
   
      Execute action in environment.
      
      :param action: Action vector
      :return: (next_state, reward, terminated, truncated, info)
      
   .. method:: normalize_state(state: np.ndarray) -> np.ndarray
   
      Normalize state to zero mean, unit variance.
      
      :param state: Raw state vector
      :return: Normalized state
      
   .. method:: denormalize_state(state: np.ndarray) -> np.ndarray
   
      Convert normalized state back to original scale.
      
      :param state: Normalized state vector
      :return: Original scale state

LunarLander Wrapper
-------------------

.. class:: LunarLanderWrapper(ContinuousEnvWrapper)

   Wrapper for LunarLander-v2 continuous control.
   
   **State space:** 8 dimensions (position, velocity, angle, leg contacts)
   
   **Action space:** 2 dimensions (main engine, side engines)
   
   .. method:: __init__(render_mode: Optional[str] = None)
   
      :param render_mode: "human" for visualization, None for headless
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.envs import LunarLanderWrapper
      
      env = LunarLanderWrapper(render_mode="human")
      state, _ = env.reset()
      
      for _ in range(1000):
          action = policy(state)
          state, reward, done, _, _ = env.step(action)
          if done:
              break

Highway Wrapper
---------------

.. class:: HighwayWrapper(ContinuousEnvWrapper)

   Wrapper for Highway-Env continuous control.
   
   **State space:** 25 dimensions (5 vehicles × 5 features)
   
   **Action space:** 2 dimensions (acceleration, steering)
   
   .. method:: __init__(config: Optional[dict] = None, render_mode: Optional[str] = None)
   
      :param config: Highway-env configuration dict
      :param render_mode: "human" for visualization
      
   **Example:**
   
   .. code-block:: python
   
      from flowshield.envs import HighwayWrapper
      
      config = {
          "observation": {
              "type": "Kinematics",
              "vehicles_count": 5
          },
          "action": {
              "type": "ContinuousAction"
          }
      }
      
      env = HighwayWrapper(config=config)
      state, _ = env.reset()

Utility Functions
-----------------

.. function:: get_env_info(env_name: str) -> dict

   Get environment metadata.
   
   :param env_name: Environment identifier
   :return: Dictionary with state_dim, action_dim, etc.
   
   **Example:**
   
   .. code-block:: python
   
      info = get_env_info("lunarlander")
      # {'state_dim': 8, 'action_dim': 2, 'max_steps': 1000}

.. function:: list_envs() -> List[str]

   List all registered environments.
   
   :return: List of environment names
   
   **Example:**
   
   .. code-block:: python
   
      envs = list_envs()
      # ['lunarlander', 'highway']

Constants
---------

.. data:: LUNAR_STATE_MEAN
   :type: np.ndarray
   
   Mean state values for LunarLander normalization.

.. data:: LUNAR_STATE_STD
   :type: np.ndarray
   
   Standard deviation for LunarLander normalization.

.. data:: HIGHWAY_STATE_MEAN
   :type: np.ndarray
   
   Mean state values for Highway normalization.

.. data:: HIGHWAY_STATE_STD
   :type: np.ndarray
   
   Standard deviation for Highway normalization.
