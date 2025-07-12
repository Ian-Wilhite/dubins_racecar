
import matplotlib.pyplot as plt

def plot_trajectory(track, positions, filename=None):
    xs, ys = track.xy
    px = [p[0] for p in positions]
    py = [p[1] for p in positions]
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, 'k--', label='Track')
    plt.plot(px, py, 'r-', label='Trajectory')
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectory vs Track")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

