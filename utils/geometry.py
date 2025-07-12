
import numpy as np

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
