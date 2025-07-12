import numpy as np
from planners.dubins_planner import generate_dubins_path
from utils.geometry import wrap_angle, distance

class OptimalControlAgent:
    def __init__(self, config):
        self.turning_radius = config.get("turning_radius", 10.0)
        self.goal = config.get("goal", (0.0, -36.5, np.pi / 2))
        self.path = None
        self.index = 0
        self.max_steering_rate_abs = config.get("max_steering_rate_abs", np.pi / 4.0)
        self.max_acceleration = config.get("max_acceleration", 2.0)
        self.min_acceleration = config.get("min_acceleration", -3.0)

    def plan(self, start):
        self.path = generate_dubins_path(start, self.goal, self.turning_radius)
        self.index = 0

    def act(self, obs):
        x, y, theta, v = obs
        if self.path is None:
            self.plan((x, y, theta))

        if self.index >= len(self.path):
            # If path is completed, try to stop the car
            normalized_acceleration = (0.0 - self.min_acceleration) / (self.max_acceleration - self.min_acceleration) * 2.0 - 1.0
            return np.array([0.0, normalized_acceleration], dtype=np.float32)

        target = self.path[self.index]
        dx, dy = target[0] - x, target[1] - y
        desired_theta = np.arctan2(dy, dx)
        
        # Calculate steering rate based on desired change in theta
        steering_rate = wrap_angle(desired_theta - theta) # This needs to be refined for actual steering rate
        
        # Normalize steering rate to [-1, 1]
        normalized_steering_rate = np.clip(steering_rate / self.max_steering_rate_abs, -1.0, 1.0)

        # Simple acceleration logic: accelerate if too slow, decelerate if too fast
        if v < 5.0:
            acceleration = 1.0
        elif v > 10.0:
            acceleration = -1.0
        else:
            acceleration = 0.0
            
        # Normalize acceleration to [-1, 1]
        normalized_acceleration = (acceleration - self.min_acceleration) / (self.max_acceleration - self.min_acceleration) * 2.0 - 1.0
        normalized_acceleration = np.clip(normalized_acceleration, -1.0, 1.0)

        if distance((x, y), (target[0], target[1])) < 1.0:
            self.index += 1
            
        return np.array([normalized_steering_rate, normalized_acceleration], dtype=np.float32)