import numpy as np
import matplotlib.pyplot as plt
import yaml
from environments.stadium_track import StadiumTrackEnv

def plot_trajectory_data(agent_type, track_polygon, outer_boundary, inner_boundary):
    trajectory_data = np.load(f"{agent_type}_trajectory.npy")
    x = trajectory_data[:, 0]
    y = trajectory_data[:, 1]
    theta = trajectory_data[:, 2]
    v = trajectory_data[:, 3]

    # Plot X-Y trajectory with track boundaries
    plt.figure(figsize=(10, 8))
    x_outer, y_outer = outer_boundary.xy
    x_inner, y_inner = inner_boundary.xy
    plt.plot(x_outer, y_outer, 'k--', label="Outer Track")
    plt.plot(x_inner, y_inner, 'k--', label="Inner Track")
    plt.plot(x, y, 'r-', label=f'{agent_type.upper()} Trajectory')
    plt.axis("equal")
    plt.legend()
    plt.title(f'{agent_type.upper()} Agent Trajectory on Stadium Track')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.savefig(f'{agent_type}_trajectory_xy.png')
    plt.close()

    # Plot Velocity over time
    plt.figure(figsize=(10, 6))
    time = np.arange(len(v)) * 0.1 # Assuming dt = 0.1
    plt.plot(time, v, 'b-')
    plt.title(f'{agent_type.upper()} Agent Velocity over Time')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.savefig(f'{agent_type}_velocity.png')
    plt.close()

    # Plot Yaw (theta) over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, theta, 'g-')
    plt.title(f'{agent_type.upper()} Agent Yaw (Theta) over Time')
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw (radians)")
    plt.grid(True)
    plt.savefig(f'{agent_type}_yaw.png')
    plt.close()

if __name__ == "__main__":
    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Initialize environment to get track boundaries
    env = StadiumTrackEnv(cfg["env"])
    track_polygon, outer_boundary, inner_boundary = env._generate_track()

    # Plot data for Optimal Control Agent
    plot_trajectory_data("oc", track_polygon, outer_boundary, inner_boundary)

    # Plot data for RL Agent
    plot_trajectory_data("rl", track_polygon, outer_boundary, inner_boundary)
