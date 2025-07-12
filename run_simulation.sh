#!/bin/bash

# Check if an agent type is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <agent_type>"
  echo "<agent_type> can be 'oc' for Optimal Control or 'rl' for Reinforcement Learning."
  exit 1
fi

AGENT_TYPE=$1
CONFIG_FILE="config.yaml"

# Update config.yaml with the selected agent type
# This uses a simple sed command, which might not be robust for all YAML structures.
# A more robust solution would involve a Python script to modify YAML.
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS (BSD sed)
  sed -i '' "s/^  type: .*$/  type: ${AGENT_TYPE}/" "$CONFIG_FILE"
else
  # Linux (GNU sed)
  sed -i "s/^  type: .*$/  type: ${AGENT_TYPE}/" "$CONFIG_FILE"
fi

echo "Running simulation for ${AGENT_TYPE} agent..."
python3 main.py

echo "Generating plots for ${AGENT_TYPE} agent..."
python3 plot_trajectories.py

echo "Simulation and plotting complete for ${AGENT_TYPE} agent."
