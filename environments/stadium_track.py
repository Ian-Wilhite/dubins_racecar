import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon, LinearRing
import gymnasium as gym
from gymnasium import spaces

class StadiumTrackEnv(gym.Env):
    def __init__(self, config):
        super(StadiumTrackEnv, self).__init__()
        self.config = config # Store the entire config
        self.track_radius = config.get("track_radius", 36.5)
        self.straight_length = config.get("straight_length", 84.39)
        self.lane_width = config.get("lane_width", 1.22)
        self.dt = config.get("dt", 0.1)
        self.max_steps = config.get("max_steps", 1000)

        # New parameters for agent control and constraints
        self.min_turning_radius = config.get("min_turning_radius", 5.0)
        self.max_acceleration = config.get("max_acceleration", 2.0) # m/s^2
        self.min_acceleration = config.get("min_acceleration", -3.0) # m/s^2 (braking)
        self.max_steering_rate_abs = config.get("max_steering_rate_abs", np.pi / 4.0) # radians/sec (e.g., 45 deg/sec)

        # Action space: [normalized_steering_rate, normalized_acceleration]
        # Both are in [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.trajectory_data = [] # To store [x, y, theta, v] at each step
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = self.config.get("initial_state", {})
        self.x = initial_state.get("x", 0.0)
        self.y = initial_state.get("y", -self.track_radius)
        self.theta = initial_state.get("theta", np.pi / 2)
        self.v = initial_state.get("v", 5.0)
        self.step_count = 0
        self.track_polygon, self.outer_boundary, self.inner_boundary = self._generate_track()
        self.lap_length = self.outer_boundary.length
        
        self.trajectory_data = [] # Clear data for new trial
        self.trajectory_data.append([self.x, self.y, self.theta, self.v]) # Save initial state

        return self._get_obs(), {}

    def _generate_track(self):
        # Define the half-length of the straight sections
        half_straight = self.straight_length / 2.0

        # Define the inner and outer radii for the track lanes
        inner_radius = self.track_radius - self.lane_width / 2.0
        outer_radius = self.track_radius + self.lane_width / 2.0

        # Generate points for the outer boundary (counter-clockwise)
        outer_points = []
        # Bottom straight (left to right)
        outer_points.extend([(x, -outer_radius) for x in np.linspace(-half_straight, half_straight, num=100)])
        # Right arc (bottom to top)
        outer_points.extend([
            (half_straight + outer_radius * np.cos(t), outer_radius * np.sin(t))
            for t in np.linspace(-np.pi / 2, np.pi / 2, num=100)
        ])
        # Top straight (right to left)
        outer_points.extend([(x, outer_radius) for x in np.linspace(half_straight, -half_straight, num=100)])
        # Left arc (top to bottom)
        outer_points.extend([
            (-half_straight + outer_radius * np.cos(t), outer_radius * np.sin(t))
            for t in np.linspace(np.pi / 2, 3 * np.pi / 2, num=100)
        ])
        outer_boundary = LinearRing(outer_points)

        # Generate points for the inner boundary (clockwise to define a hole)
        inner_points = []
        # Bottom straight (right to left)
        inner_points.extend([(x, -inner_radius) for x in np.linspace(half_straight, -half_straight, num=100)])
        # Left arc (bottom to top, clockwise)
        inner_points.extend([
            (-half_straight + inner_radius * np.cos(t), inner_radius * np.sin(t))
            for t in np.linspace(3 * np.pi / 2, np.pi / 2, num=100) 
        ])
        # Top straight (left to right)
        inner_points.extend([(x, inner_radius) for x in np.linspace(-half_straight, half_straight, num=100)])
        # Right arc (top to bottom, clockwise)
        inner_points.extend([
            (half_straight + inner_radius * np.cos(t), inner_radius * np.sin(t))
            for t in np.linspace(np.pi / 2, -np.pi / 2, num=100) 
        ])
        inner_boundary = LinearRing(inner_points)

        # Create a polygon from the outer and inner boundaries
        track_polygon = Polygon(outer_boundary, [inner_boundary])
        
        return track_polygon, outer_boundary, inner_boundary

    def step(self, action):
        normalized_steering_rate, normalized_acceleration = action

        # Apply constraints and scale actions
        actual_steering_rate = np.clip(normalized_steering_rate, -1.0, 1.0) * self.max_steering_rate_abs
        
        # Scale normalized_acceleration from [-1, 1] to [min_acceleration, max_acceleration]
        actual_acceleration = self.min_acceleration + (normalized_acceleration + 1) * \
                              (self.max_acceleration - self.min_acceleration) / 2.0
        actual_acceleration = np.clip(actual_acceleration, self.min_acceleration, self.max_acceleration)

        self.v += actual_acceleration * self.dt
        self.v = max(0.0, self.v) # Ensure velocity doesn't go negative

        self.theta += actual_steering_rate * self.dt

        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        point = Point(self.x, self.y)
        dist = self.outer_boundary.project(point)
        reward = self.v * self.dt
        
        out_of_bounds = not self.track_polygon.contains(point)
        
        done = dist >= self.lap_length or self.step_count >= self.max_steps or out_of_bounds
        
        if out_of_bounds:
            reward -= 100 # Penalize for going out of bounds

        self.step_count += 1
        self.trajectory_data.append([self.x, self.y, self.theta, self.v]) # Save current state
        return self._get_obs(), reward, done, False, {"distance": dist, "lap_length": self.lap_length, "out_of_bounds": out_of_bounds}

    def _get_obs(self):
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32)

    def render(self, filename=None):
        if self.track_polygon:
            x_outer, y_outer = self.outer_boundary.xy
            x_inner, y_inner = self.inner_boundary.xy
            plt.figure(figsize=(8, 6))
            plt.plot(x_outer, y_outer, 'k--', label="Outer Track")
            plt.plot(x_inner, y_inner, 'k--', label="Inner Track")
            plt.plot(self.x, self.y, 'ro', label="Car")
            plt.axis("equal")
            plt.legend()
            plt.title("Racecar on Stadium Track")
            if filename:
                plt.savefig(filename)
            else:
                plt.show()

    def evaluate(self):
        print(f"Completed lap at position ({self.x:.2f}, {self.y:.2f})")

    def save_trial_data(self, filename):
        np.save(filename, np.array(self.trajectory_data))