import gymnasium as gym
import numpy as np
import torch


class CarEnv(gym.Wrapper):
    """CarRacing-v3 with continuous actions that speaks torch on `device`.

    The simulation (Box2D physics + pygame rendering) always runs on the CPU;
    this wrapper moves each observation to `device` once, as a uint8 tensor,
    and accepts actions as tensors from any device.

    Observations: uint8 tensor (96, 96, 3) on `device`
    Actions:      [steer, gas, brake], clipped to [-1, 1], [0, 1], [0, 1]
    """

    def __init__(self, device="cpu", domain_randomization=True, render_mode=None):
        env = gym.make("CarRacing-v3", continuous=True,
                       domain_randomize=domain_randomization, render_mode=render_mode)
        super().__init__(env)
        self.device = torch.device(device)

    def _to_torch(self, obs):
        return torch.as_tensor(obs, device=self.device)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_torch(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # A Gaussian policy can sample outside the bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_torch(obs), reward, terminated, truncated, info
