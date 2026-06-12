# Deep Reinforcement Learning: Double DQN for Atari Beam Rider

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

🚀 **Peak Agent Score: 3,560** 
*(Achieved after 2,000,000 training frames. A massive ~10x improvement over the random baseline!)*

This repository contains a complete, from-scratch implementation of a **Deep Q-Network (DQN)** and **Double DQN** agent trained to play the classic Atari 2600 game *Beam Rider*. 

This project was built as a rigorous university assignment focused on the **explainability and mathematical foundations** of Deep Reinforcement Learning. As such, **no high-level black-box RL libraries** (like Stable-Baselines3 or RLlib) were used. The Bellman update, experience replay buffer, preprocessing pipeline, and Convolutional Neural Networks were built entirely from the ground up using PyTorch.

---

## 📊 Results & Performance

After training for 2 million frames (~12 hours on a modern GPU), the Double DQN agent successfully learned complex spatial-temporal features, evasive maneuvers, and shooting accuracy.

| Metric | Score | Context |
| :--- | :--- | :--- |
| **Random Policy Baseline** | 363.9 | Completely random button presses. |
| **Agent Average Reward** | **1058.40** | Average over 10 strict evaluation episodes ($\epsilon = 0.05$). |
| **Agent Peak Reward** | **3560.00** | Best recorded run. The agent mastered basic sector survival. |
| *DeepMind DQN (2015)* | *6846.0* | *For context: the original Nature paper trained for 50M frames.* |

*(A recorded gameplay video of the 3560-point run is available in the `gameplay_videos/` directory).*

---

## 🧠 The Algorithm & Math

The agent learns using **Q-Learning**, a model-free reinforcement learning algorithm. The goal is to learn the optimal action-value function $Q^*(s, a)$, which represents the maximum expected cumulative reward starting from state $s$ and taking action $a$.

### The Bellman Equation
The agent updates its Q-values by minimizing the Mean Squared Error (or Huber Loss) between its current prediction and the Bellman target:
$$Target = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

### Core Architectural Enhancements
To make this work on high-dimensional pixel inputs, three critical enhancements were implemented:

1.  **Convolutional Neural Network (CNN):** A 3-layer CNN processes the stacked visual frames to extract features, followed by fully connected layers that output a Q-value for each of the 9 possible joystick actions.
2.  **Experience Replay Buffer:** Instead of learning from highly correlated sequential frames, transitions $(s, a, r, s', done)$ are stored in a cyclic buffer (size 100,000). The network trains on randomized mini-batches of 32, breaking correlation and stabilizing gradients.
3.  **Target Network:** A secondary "frozen" network ($\theta^-$) evaluates the next state. It is synchronized with the main online network ($\theta$) only every 10,000 steps to prevent the targets from shifting erratically during training.
4.  **Double DQN (DDQN):** Standard DQN suffers from overestimation bias due to the `max` operator. DDQN solves this by decoupling action *selection* from *evaluation*:
    *   *Select* best action using Online Net: $a^* = \arg\max_{a} Q(S_{t+1}, a; \theta)$
    *   *Evaluate* action using Target Net: $Q(S_{t+1}, a^*; \theta^-)$

---

## 🏗️ Codebase Structure

The project was developed in 8 distinct phases, adhering strictly to clean architectural separation:

```text
ASMBeamRider/
├── configs/
│   └── dqn_config.py          # Centralized hyperparameters (no magic numbers)
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py       # Core Bellman update and epsilon-greedy logic
│   │   └── double_dqn_agent.py# Overridden DDQN train step
│   ├── environment/
│   │   └── preprocessing.py   # Custom Gymnasium wrappers (Grayscale, Resize, Skip, Stack)
│   ├── networks/
│   │   └── q_network.py       # PyTorch CNN architecture
│   ├── replay/
│   │   └── replay_buffer.py   # Cyclic experience replay memory
│   ├── training/
│   │   └── trainer.py         # Training loop, TensorBoard logging, Checkpoints
│   └── evaluation/
│       └── evaluator.py       # Deterministic model evaluation
├── train.py                   # Entry point for training
├── evaluate.py                # Entry point for running the test episodes
├── run_ablations.py           # Entry point for Phase 7 Ablation Studies
├── record_video.py            # Generates gameplay mp4s
├── reports/                   # Markdown analysis reports and PNG graphs
├── models/                    # Saved .pth weights
└── logs/                      # TensorBoard event files
```

### The Preprocessing Pipeline
Atari 2600 outputs `210x160` RGB images at 60 FPS. Feeding this directly into a CNN is computationally prohibitive and breaks the Markov assumption (a static frame has no velocity).
*   **Grayscale & Resize:** Reduced to `84x84` grayscale (93% reduction in state space).
*   **Frame Skip & Max-Pool:** Actions are repeated for 4 frames. The last two frames are max-pooled to prevent Atari sprite flickering (bullets disappearing).
*   **Frame Stacking:** The 4 most recent frames are stacked into a single `(4, 84, 84)` tensor, providing the network with the temporal context necessary to perceive motion and velocity.

---

## 🔬 Ablation Studies

To scientifically prove the necessity of the chosen architecture, the repository includes automated ablation studies (`run_ablations.py`) that intentionally break the agent. The results (fully documented in `reports/ablation_study.md`) empirically prove:
*   **Without Target Network:** Loss diverges to infinity; rewards flatline near 200.
*   **Without Replay Buffer:** Catastrophic forgetting; agent overfits to sequential frames and fails to generalize.
*   **Without Frame Stacking:** Agent becomes blind to velocity, failing completely at evasive maneuvers.

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requires Python 3.11, PyTorch, Gymnasium[atari], and TensorBoard).*

### 2. Train the Agent
Hyperparameters can be adjusted in `configs/dqn_config.py`.
```bash
python train.py
```
To monitor training:
```bash
tensorboard --logdir=logs/
```

### 3. Evaluate the Agent
Runs the model saved at `models/dqn_final.pth` with a frozen network and minimal exploration ($\epsilon = 0.05$).
```bash
python evaluate.py
```

### 4. Record a Gameplay Video
Saves `.mp4` recordings of the agent's gameplay to `gameplay_videos/`.
```bash
python record_video.py
```

### 5. Run Ablation Studies
Executes 3 shortened training loops with crippled configurations to reproduce failure states.
```bash
python run_ablations.py
```

---
*Developed as a University Deep Reinforcement Learning Assignment.*
