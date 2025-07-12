# Dubins Racecar RL vs OC

This project compares **Reinforcement Learning (RL)** with **Optimal Control (OC)** for a racecar navigating closed tracks using Dubins path planning. The initial environment models a standard 400m stadium track, and the goal is to complete a full lap.

---

## 🧠 Project Goals

* **Evaluate** path-following performance of RL agents vs. optimal control.
* **Develop** a modular simulation framework with swappable environments and controllers.
* **Benchmark** trajectories, lap times, and deviations from the ideal path.

---

## 📁 Code Structure

```
dubins_racecar/
├── agents/
│   └── optimal_control_agent.py    # OC agent using Dubins paths
├── environments/
│   └── stadium_track.py           # 400m oval stadium Gym environment
├── planners/
│   └── dubins_planner.py          # Dubins path generator
├── utils/
│   ├── geometry.py                # Math helpers
│   ├── metrics.py                 # Evaluation metrics
│   └── visualization.py           # Plotting utilities
├── train_and_test_rl.py          # Stable Baselines3 RL training + testing
└── README.md                      # Project overview and instructions
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Required packages:

* `stable-baselines3`
* `gymnasium`
* `matplotlib`
* `shapely`
* `dubins`

You can install them manually:

```bash
pip install stable-baselines3 gymnasium matplotlib shapely dubins
```

### 2. Train the RL Agent

```bash
python dubins_racecar/train_and_test_rl.py
```

This will:

* Create the stadium track environment
* Train a PPO agent for 50,000 steps
* Test and visualize the final trajectory

### 3. Run Optimal Control Agent (TBD)

Create a separate script to initialize and evaluate `OptimalControlAgent` from `agents/`.

---

## 📊 Evaluation Metrics (Planned)

* Lap time (seconds)
* Deviation from track centerline
* Reward accumulation (RL only)
* Success rate across different environments

---

## 🔧 Extending the Framework

You can:

* Add new environments to `environments/`
* Swap planners or control policies
* Benchmark more advanced RL agents (e.g., SAC, TD3)
