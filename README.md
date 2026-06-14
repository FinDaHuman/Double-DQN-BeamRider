# Deep Reinforcement Learning: Mastering Beam Rider (Unrestricted Branch)

![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-201547?style=for-the-badge&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

🚀 **Current Focus: Defeating the Sector Sentinel with RecurrentPPO**

Welcome to the **Unrestricted RL** branch of the Beam Rider project. 

While the `master` branch contains a strict, from-scratch PyTorch implementation of Double DQN for educational purposes, this branch lifts those academic restrictions. Our new goal is **maximum performance** and solving the game's hard exploration traps using state-of-the-art ecosystem tools.

---

## 🧠 The Challenge: The "Blocker" Trap

During our initial DQN evaluation, we discovered a significant "cognitive wall" for standard agents: The **Sector Sentinel**.

The Sentinel (the boss at the end of each sector) does not attack directly. Instead, it spawns fast-moving indestructible "Green Blockers" directly on the player's position. Defeating the boss requires a complex, delayed-reward sequential strategy:
1. **Wait** on a beam to bait the Blocker.
2. **Dodge** to an adjacent beam right as it locks on.
3. **Shoot** a limited-supply torpedo at the Sentinel.

Because standard DQN relies on 4-frame stacking (which only provides immediate velocity, not long-term memory) and $\epsilon$-greedy exploration (random noise), it statistically fails to discover or execute this multi-step sequence, resulting in repeated deaths.

---

## 🛠️ The Solution: RecurrentPPO (LSTM)

To overcome this trap, we have transitioned from a custom DQN to **Stable-Baselines3** and **sb3-contrib**, implementing a **Recurrent Proximal Policy Optimization (PPO)** agent.

*   **LSTM Memory:** By replacing standard linear layers with a Long Short-Term Memory (LSTM) network, the agent gains true temporal memory. It can remember baiting a blocker dozens of frames ago, allowing it to execute the delayed "wait -> dodge -> shoot" sequence.
*   **Hardware Optimized:** The training script (`apps/beam_rider/train_sb3_lstm.py`) is explicitly configured to run safely on lower-end laptop hardware. It uses `DummyVecEnv` to prevent multiprocessing freezes and utilizes smaller batch sizes to avoid Out-Of-Memory errors.

---

## ⚙️ How to Run

### 1. Install the Unrestricted Dependencies
This branch requires the Stable-Baselines3 ecosystem and compatible environment wrappers.
```bash
pip install "stable-baselines3[extra]>=2.0.0" sb3-contrib "ale-py==0.8.1" "numpy<2.0.0"
```

### 2. Train the LSTM Agent
We have provided a hardware-safe training script. (For best results, allow this to run for at least 10,000,000 timesteps).
```bash
python apps/beam_rider/train_sb3_lstm.py
```

### 3. Evaluate and Record
Watch the agent attempt to defeat the Sector Sentinel.
```bash
python apps/beam_rider/evaluate_sb3.py
```

---
*This branch represents the evolution from educational algorithms to state-of-the-art application.*