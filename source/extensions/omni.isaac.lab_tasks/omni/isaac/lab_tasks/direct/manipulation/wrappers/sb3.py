from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import numpy as np


class Sb3VecEnvWrapperRace(Sb3VecEnvWrapper):
    """Wrapper specifically for Reach Avoid Collision Environment"""

    def __init__(self, env: ManagerBasedRLEnv):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        observation_space = self.unwrapped.single_observation_space
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs
