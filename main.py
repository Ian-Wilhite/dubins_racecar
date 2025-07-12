from environments.stadium_track import StadiumTrackEnv
from agents.optimal_control_agent import OptimalControlAgent
from agents.rl_agent import RLAgent
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize environment and agent
env = StadiumTrackEnv(cfg["env"])
agent_type = cfg["agent"]["type"]

if agent_type == "oc":
    agent = OptimalControlAgent(cfg["agent"])
else:
    agent = RLAgent(cfg["agent"])

# Run simulation
obs, info = env.reset()

done = False
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render(f"{agent_type}_sim.png")
env.evaluate()
env.save_trial_data(f"{agent_type}_trajectory.npy")
