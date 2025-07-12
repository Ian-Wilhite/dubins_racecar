import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environments.stadium_track import StadiumTrackEnv
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Define training config from config.yaml
env_config = cfg["env"]

def make_env():
    return StadiumTrackEnv(env_config)

if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)

    # Train agent with TensorBoard logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    model.learn(total_timesteps=100000)

    # Save model
    model.save("ppo_racecar")

    # Test the trained agent
    obs, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        trajectory.append((obs[0], obs[1]))

    # Visualize results
    from utils.visualization import plot_trajectory
    # Need to get the track from the environment for plotting
    env.track_polygon, env.outer_boundary, env.inner_boundary = env._generate_track()
    plot_trajectory(env.outer_boundary, trajectory, "rl_trajectory.png")
