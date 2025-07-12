from planners.dubins_path import dubins_path_planning
import numpy as np

def generate_dubins_path(start, goal, turning_radius):
    path_x, path_y, path_yaw, mode, lengths = dubins_path_planning(
        start[0], start[1], start[2], goal[0], goal[1], goal[2], 1.0 / turning_radius
    )
    return np.array([path_x, path_y, path_yaw]).T