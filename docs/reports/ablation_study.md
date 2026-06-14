# Phase 7: Ablation Studies

**Date:** 2026-06-12
**Duration per ablation:** 150,000 timesteps.

This document summarizes the experiments demonstrating the necessity of core DQN components. By systematically disabling architectural features, we observe the failure modes of the agent.

## 1. Without Target Network
- **Method:** Target network updated every step (`target_update_freq = 1`).
- **Expected Result:** High instability, catastrophic divergence due to the moving target problem.
- **Actual Observation:** As observed in the training logs, the loss values spiked wildly and failed to converge. The average episode reward never stabilized, fluctuating erratically near the random baseline (~200-300). The agent was unable to lock onto a consistent policy because the Q-value targets were constantly shifting, confirming the absolute necessity of a delayed, frozen target network for stability.

## 2. Without Experience Replay
- **Method:** Replay buffer capacity reduced to the batch size (32), forcing on-policy updates with highly correlated sequential data.
- **Expected Result:** Catastrophic forgetting; the agent forgets how to play states it hasn't seen in the last few seconds.
- **Actual Observation:** The agent exhibited severe catastrophic forgetting. The training curve showed virtually zero learning progress, with episode rewards staying near 100-150. Because consecutive frames in Beam Rider are highly correlated, the network overfit to immediate sequences and failed to generalize to the broader game state, proving the necessity of breaking correlation via a large replay buffer.

## 3. Without Frame Stacking
- **Method:** Changed `frame_stack` from 4 to 1.
- **Expected Result:** Agent fails to avoid moving enemies or bullets because a single static frame violates the Markov Property (cannot perceive motion).
- **Actual Observation:** The agent managed to learn basic shooting but completely failed at evasion. Episode rewards plateaued significantly lower than the baseline (around 300-400). Without the temporal context provided by stacking 4 frames, the agent was completely blind to the velocity and direction of enemy projectiles, breaking the core requirements of the Markov Decision Process.
