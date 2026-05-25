# AI Task Offloading for Edge Computing using Deep Reinforcement Learning
A lightweight DQN-powered controller that learns when to run tasks locally on an edge device and when to offload them to the cloud — in real time, on real hardware.

## Overview

Edge devices (smartphones, IoT nodes, AR headsets) operate under tight resource constraints: limited CPU, fixed RAM, and finite battery. Traditional offloading systems use static rules like *"send to cloud if CPU > 80%"* — but a single metric cannot capture the full hardware state.

This project replaces those hardcoded rules with a **Deep Q-Network (DQN)** agent that observes five live system signals and learns an optimal routing policy through trial and error across 5,000 training episodes.

**Results vs. static baseline:**
- Up to **47% lower latency** on heavy tasks
- **~40% faster** execution on simple tasks (kept local instead of needlessly offloaded)
- **Zero RAM violations** across 300 test decisions (vs. 3 with static threshold)

---

## Repository Structure

```
├── universal_env.py       # Environment: reads live hardware via psutil, executes tasks, returns rewards
├── universal_agent.py     # DQN Agent: PyTorch neural network, experience replay, epsilon-greedy policy
├── deploy_anywhere.py     # Main runtime: loads model, routes tasks, adapts to new hardware continuously
├── universal_model.pth    # Pre-trained model weights (ready to deploy — no retraining needed)
└── .gitignore
```

---

## How It Works

The offloading problem is modelled as a **Markov Decision Process (MDP)**:

| Component | Description |
|---|---|
| **State** | 5-element vector: `[CPU %, RAM %, Battery %, Power State, Task Complexity]` |
| **Action** | Binary: `0 = run locally`, `1 = offload to cloud` |
| **Reward** | `-latency - (100 × RAM_penalty) - (200 × Energy_penalty)` |

The **DQN architecture** is a feed-forward neural network:
```
Input (5) → FC Layer (64, ReLU) → FC Layer (64, ReLU) → Output (2 Q-values)
```

At each decision step, the agent picks the action with the higher Q-value. Weights are updated using the **Bellman Equation** with γ = 0.99 and a replay buffer of 10,000 transitions.

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/VidushiChauhan29/AI-Task-Offloading-in-Edge-computing.git
cd AI-Task-Offloading-in-Edge-computing

# Install dependencies
pip install torch psutil numpy
```

---

## Usage

### Run on your machine (uses pre-trained model)

```bash
python deploy_anywhere.py
```

The script will:
1. Load `universal_model.pth` (pre-trained weights)
2. Start reading your machine's live CPU, RAM, and battery
3. Route every incoming task — locally or to the cloud
4. Continuously fine-tune to your hardware's behaviour
5. Save the adapted brain on exit (`Ctrl+C`)

**Sample output:**
```
Task 270 | Epsilon: 0.02 | Level:  8 | CPU: 62.0% | Bat: 78% | ☁️  CLOUD | 0.048s
Task 280 | Epsilon: 0.02 | Level:  2 | CPU: 14.0% | Bat: 78% | 🖥️  LOCAL | 0.021s
```

### Train from scratch

```bash
# (Optional) Retrain on your own hardware
python universal_agent.py
```

---

## Key Design Decisions

**Why DQN over static rules?**
Static thresholds are single-dimensional. A CPU reading of 75% means very different things on a device with 90% RAM usage and 15% battery vs. one that is plugged in and idle. The DQN jointly reasons over all five signals.

**Why γ = 0.99?**
A discount factor close to 1 makes the agent value long-term battery health, not just immediate per-task latency.

**Why penalty values of 100 (RAM) and 200 (Energy)?**
Empirically tuned — values below 50 failed to prevent violations; values above 300 caused the agent to always offload regardless of load.

**Why Experience Replay?**
Without it, sequential training causes catastrophic forgetting. The replay buffer of 10,000 ensures rare but critical events (RAM violation at low battery) are reinforced many times.

---

## Cross-Device Adaptation

The model transfers to new hardware without retraining. When `deploy_anywhere.py` loads on a new machine, epsilon is reset to **0.15** (15% exploration) instead of starting from scratch. The agent uses its pre-trained policy for 85% of decisions and self-calibrates over time, saving the updated brain on exit.

---

## Results

| Metric | Static Threshold | DQN Agent | Improvement |
|---|---|---|---|
| Latency — simple tasks (L1–L3) | 0.027s | 0.013–0.035s | ~40% faster |
| Latency — heavy tasks (L8–L10) | 0.085s | 0.045s | ~47% faster |
| RAM violations (300 tasks) | 3 | 0 | 100% eliminated |
| Unnecessary cloud calls | 100% of L1–L3 | <5% of L1–L3 | 95% reduction |

---

## Tech Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.10 | Core language |
| PyTorch | 2.x | DQN + backpropagation |
| psutil | 5.9.x | Live hardware telemetry |
| NumPy | 1.24+ | State vector construction |
| collections.deque | stdlib | Experience replay buffer |

---

## Limitations & Future Work

- Task complexity is currently a manually assigned integer (1–10). In production, this would be derived from actual payload characteristics (image resolution, sequence length, model parameter count).
- Cloud delay is fixed at ~0.027s. Future work will add real-time network latency monitoring via `ping3`.
- Potential extensions: partial offloading (split task between edge and cloud), Double DQN, live ML workloads instead of simulated delays.

---

## Authors

- **Anmol Sharma** (A25305223118) — Amity University Punjab
- **Vidushi Chauhan** — Amity University Punjab

Supervised by **Dr. Avtar Singh**, Amity School of Engineering and Technology

---

## References

1. Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540).
2. Liu et al. (2024). A deep Q-learning model for sequential task offloading in edge AI systems.
3. Grand View Research (2023). Mobile Edge Computing Market Size & Growth Report, 2030.
