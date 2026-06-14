# Deep Reinforcement Learning: Double DQN for Atari Beam Rider

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

🚀 **Peak Agent Score: 3,560** 
*(Achieved after 2,000,000 training frames. A massive ~10x improvement over the random baseline!)*

A from-scratch PyTorch implementation of a **Double Deep Q-Network (DDQN)** trained to master the classic Atari 2600 game *Beam Rider*. 

This project was built as a rigorous university assignment focused on the **explainability and mathematical foundations** of Deep Reinforcement Learning. As such, **no high-level black-box RL libraries** (like Stable-Baselines3 or RLlib) were used. The Bellman update, experience replay buffer, preprocessing pipeline, and Convolutional Neural Networks were built entirely from the ground up.

---

## 🏗️ Monorepo Codebase Structure

The project is organized as a structurally precise monorepo, cleanly separating the reusable reinforcement learning framework from the environment-specific code and application scripts:

```text
/
├── apps/
│   └── beam_rider/              # Application entry points and configs
│       ├── configs/             # Centralized hyperparameters
│       ├── train.py             # Entry point for training
│       ├── evaluate.py          # Entry point for running test episodes
│       ├── run_ablations.py     # Automates Phase 7 Ablation Studies
│       ├── record_video.py      # Generates gameplay mp4s
│       ├── generate_plots.py    # Parses TensorBoard logs to PNGs
│       ├── models/              # Saved .pth weights
│       └── logs/                # TensorBoard event files
├── packages/
│   ├── rl_core/                 # Reusable Deep RL algorithms
│   │   ├── agents/              # DQN and Double DQN agent logic
│   │   ├── networks/            # PyTorch CNN architectures
│   │   ├── replay/              # Experience replay memory
│   │   ├── training/            # Training loop orchestration
│   │   └── evaluation/          # Evaluation orchestration
│   └── atari_env/               # Environment specific wrappers
│       └── environment/         # Grayscale, Resize, Skip, Stack
├── docs/
│   └── reports/                 # Markdown analysis reports and graphs
└── assets/
    └── gameplay_videos/         # Rendered best-attempt gameplay
```

---

## 📊 Results & Performance

After training for 2 million frames (~12 hours on a modern GPU), the Double DQN agent successfully learned complex spatial-temporal features, evasive maneuvers, and shooting accuracy.

| Metric | Score | Context |
| :--- | :--- | :--- |
| **Random Policy Baseline** | 363.9 | Completely random button presses. |
| **Agent Average Reward** | **1058.40** | Average over 10 strict evaluation episodes ($\epsilon = 0.05$). |
| **Agent Peak Reward** | **3560.00** | Best recorded run. The agent mastered basic sector survival. |
| *DeepMind DQN (2015)* | *6846.0* | *For context: the original Nature paper trained for 50M frames.* |

*(A recorded gameplay video of the 3560-point run is available in the `assets/gameplay_videos/` directory).*

---

## 🧠 The Algorithm & Math

The agent learns using **Q-Learning**, a model-free reinforcement learning algorithm. The goal is to learn the optimal action-value function $Q^*(s, a)$, which represents the maximum expected cumulative reward starting from state $s$ and taking action $a$.

### The Bellman Equation
The agent updates its Q-values by minimizing the Huber Loss between its current prediction and the Bellman target:
$$Target = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

### Core Architectural Enhancements
1.  **Convolutional Neural Network (CNN):** A 3-layer CNN processes the stacked visual frames to extract features, outputting Q-values for all 9 joystick actions.
2.  **Experience Replay Buffer:** Transitions $(s, a, r, s', done)$ are stored in a cyclic buffer (size 100k). Training on randomized mini-batches of 32 breaks correlation and stabilizes gradients.
3.  **Target Network:** A secondary "frozen" network ($\theta^-$) evaluates the next state. It is synchronized with the main online network ($\theta$) every 10,000 steps.
4.  **Double DQN (DDQN):** Solves standard DQN overestimation bias by decoupling action *selection* from *evaluation*:
    *   *Select* best action using Online Net: $a^* = \arg\max_{a} Q(S_{t+1}, a; \theta)$
    *   *Evaluate* action using Target Net: $Q(S_{t+1}, a^*; \theta^-)$

### The Preprocessing Pipeline
Feeding `210x160` RGB images at 60 FPS directly into a CNN is computationally prohibitive and breaks the Markov assumption (a static frame lacks velocity). The `packages/atari_env` handles:
*   **Grayscale & Resize:** Reduced to `84x84` grayscale (93% reduction in state space).
*   **Frame Skip & Max-Pool:** Actions are repeated for 4 frames. The last two frames are max-pooled to prevent Atari sprite flickering.
*   **Frame Stacking:** The 4 most recent frames are stacked into a `(4, 84, 84)` tensor, providing the network with temporal context (motion and velocity).

---

## 🔬 Ablation Studies
To scientifically prove the necessity of the chosen architecture, the repository includes automated ablation studies (`apps/beam_rider/run_ablations.py`). The results (`docs/reports/ablation_study.md`) empirically prove:
*   **Without Target Network:** Loss diverges to infinity; rewards flatline near 200.
*   **Without Replay Buffer:** Catastrophic forgetting; agent overfits to sequential frames and fails to generalize.
*   **Without Frame Stacking:** Agent becomes blind to velocity, failing completely at evasive maneuvers.

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Agent
Hyperparameters can be adjusted in `apps/beam_rider/configs/dqn_config.py`.
```bash
cd apps/beam_rider
python train.py
```

### 3. Evaluate the Agent
Runs the model saved at `models/dqn_final.pth` with a frozen network and minimal exploration ($\epsilon = 0.05$).
```bash
cd apps/beam_rider
python evaluate.py
```

### 4. Record a Gameplay Video
```bash
cd apps/beam_rider
python record_video.py
```

---
*Developed as a University Deep Reinforcement Learning Assignment.*
