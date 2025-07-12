# dubins_racecar/environments/stadium_track.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import gymnasium as gym
from gymnasium import spaces

class StadiumTrackEnv(gym.Env):
    def __init__(self, config):
        super(StadiumTrackEnv, self).__init__()
        self.track_radius = config.get("track_radius", 36.5)
        self.straight_length = config.get("straight_length", 84.39)
        self.lane_width = config.get("lane_width", 1.22)
        self.dt = config.get("dt", 0.1)
        self.max_steps = config.get("max_steps", 1000)
        self.action_space = spaces.Box(low=np.array([-1.0, -3.0]), high=np.array([1.0, 3.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.y = -self.track_radius
        self.theta = np.pi / 2
        self.v = 5.0
        self.step_count = 0
        self.track = self._generate_track()
        self.lap_length = self.track.length
        return self._get_obs(), {}

    def _generate_track(self):
        arc = lambda r, start, end: [
            (r * np.cos(t), r * np.sin(t))
            for t in np.linspace(start, end, num=50)
        ]
        left_arc = arc(self.track_radius, -np.pi / 2, np.pi / 2)
        top = [(x + self.straight_length, y) for x, y in left_arc[::-1]]
        right_arc = arc(self.track_radius, np.pi / 2, 3 * np.pi / 2)
        bottom = [(x - self.straight_length, y) for x, y in right_arc[::-1]]
        return LineString(left_arc + top + right_arc + bottom)

    def step(self, action):
        steering, acceleration = action
        self.v += acceleration * self.dt
        self.theta += steering * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        point = Point(self.x, self.y)
        dist = self.track.project(point)
        reward = self.v * self.dt
        done = dist >= self.lap_length or self.step_count >= self.max_steps
        self.step_count += 1
        return self._get_obs(), reward, done, False, {"distance": dist, "lap_length": self.lap_length}

    def _get_obs(self):
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32)

    def render(self):
        xs, ys = self.track.xy
        plt.figure(figsize=(8, 6))
        plt.plot(xs, ys, 'k--', label="Track")
        plt.plot(self.x, self.y, 'ro', label="Car")
        plt.axis("equal")
        plt.legend()
        plt.title("Racecar on Stadium Track")
        plt.show()

    def evaluate(self):
        print(f"Completed lap at position ({self.x:.2f}, {self.y:.2f})")


# dubins_racecar/planners/dubins_planner.py
import dubins
import numpy as np

def generate_dubins_path(start, goal, turning_radius):
    path = dubins.shortest_path(start, goal, turning_radius)
    configurations, _ = path.sample_many(0.5)
    return np.array(configurations)


# dubins_racecar/agents/optimal_control_agent.py
import numpy as np
from planners.dubins_planner import generate_dubins_path
from utils.geometry import wrap_angle, distance

class OptimalControlAgent:
    def __init__(self, config):
        self.turning_radius = config.get("turning_radius", 10.0)
        self.goal = config.get("goal", (0.0, -36.5, np.pi / 2))
        self.path = None
        self.index = 0

    def plan(self, start):
        self.path = generate_dubins_path(start, self.goal, self.turning_radius)
        self.index = 0

    def act(self, obs):
        x, y, theta, v = obs
        if self.path is None:
            self.plan((x, y, theta))

        if self.index >= len(self.path):
            return np.array([0.0, 0.0], dtype=np.float32)

        target = self.path[self.index]
        dx, dy = target[0] - x, target[1] - y
        desired_theta = np.arctan2(dy, dx)
        steering = wrap_angle(desired_theta - theta)
        acceleration = 1.0 if v < 5.0 else 0.0
        if distance((x, y), (target[0], target[1])) < 1.0:
            self.index += 1
        return np.array([steering, acceleration], dtype=np.float32)

# dubins_racecar/environments/stadium_track.py
[... existing content unchanged ...]


# dubins_racecar/train_and_test_rl.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environments.stadium_track import StadiumTrackEnv

# Define training config
env_config = {
    "track_radius": 36.5,
    "straight_length": 84.39,
    "lane_width": 1.22,
    "dt": 0.1,
    "max_steps": 1000
}

def make_env():
    return StadiumTrackEnv(env_config)

if __name__ == "__main__":
    env = make_env()
    check_env(env, warn=True)

    # Train agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    # Save model
    model.save("ppo_racecar")

    # Test the trained agent
    obs, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        trajectory.append((obs[0], obs[1]))

    # Visualize results
    from utils.visualization import plot_trajectory
    plot_trajectory(env.track, trajectory)
