import numpy as np
from stable_baselines3 import PPO

class RLAgent:
    def __init__(self, config):
        self.model = PPO.load(config.get("model_path", "ppo_racecar"))

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
