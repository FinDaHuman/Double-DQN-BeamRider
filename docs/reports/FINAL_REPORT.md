# Final Report: Reinforcement Learning Agent for Beam Rider

**Course Assignment:** Deep Reinforcement Learning
**Target Environment:** Atari 2600 Beam Rider (`ALE/BeamRider-v5`)
**Agent Architecture:** Deep Q-Network (Double DQN)

---

## 1. Introduction
The objective of this project was to develop a Deep Reinforcement Learning agent capable of playing the Atari 2600 game *Beam Rider* using a self-implemented Deep Q-Network (DQN) architecture. In strict adherence to the assignment constraints, no high-level black-box RL libraries (such as Stable-Baselines3 or RLlib) were utilized. The core algorithms—including the Bellman update, experience replay, and the Convolutional Neural Network (CNN)—were written from scratch in PyTorch to demonstrate a fundamental understanding of Deep RL mechanics. The project successfully evaluated the baseline DQN, extended it to a Double DQN to mitigate overestimation bias, and conducted ablation studies to prove the necessity of the underlying algorithmic components.

## 2. Background
Reinforcement learning relies on the framework of a Markov Decision Process (MDP), characterized by states ($S$), actions ($A$), transition probabilities, and rewards ($R$). The goal of the agent is to learn an optimal policy $\pi$ that maximizes the expected cumulative discounted reward. 

Q-Learning achieves this by estimating the action-value function, $Q(s, a)$. The optimal Q-function obeys the Bellman Optimality Equation:
$$Q^*(s, a) = \mathbb{E} [ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t = s, A_t = a ]$$

In high-dimensional continuous state spaces like Atari visual frames, tabular Q-learning is impossible. Deep Q-Networks (DQN) solve this by using a Convolutional Neural Network parameterized by $\theta$ to approximate the Q-values: $Q(s, a; \theta) \approx Q^*(s, a)$.

## 3. Beam Rider Environment
The environment was provided via the Arcade Learning Environment (ALE) through the `Gymnasium` API.
*   **Raw State Space:** An RGB array of shape `(210, 160, 3)`.
*   **Action Space:** A discrete space of 9 actions (NOOP, FIRE, UP, RIGHT, LEFT, DOWN, and diagonal firing combinations).
*   **Preprocessing Pipeline:** To make the environment computationally feasible and compliant with the Markov Property, a custom preprocessing pipeline was built:
    1.  **Grayscale & Resize:** Reduced the raw 100,800 RGB pixels to a `(84, 84)` grayscale array (93% reduction).
    2.  **Max-Pooling & Frame Skip:** Skipped 4 frames to accelerate learning and max-pooled the last two frames to prevent sprite flickering.
    3.  **Frame Stacking:** Stacked the 4 most recent frames into a `(4, 84, 84)` tensor. This was critical to provide the CNN with a sense of motion, direction, and velocity.

## 4. DQN Methodology
The core DQN was implemented with the following stabilizing techniques:
*   **Experience Replay:** Instead of learning from highly correlated, sequential frames, transitions $(s, a, r, s', done)$ were stored in a cyclic buffer of size 100,000. Mini-batches of 32 were sampled uniformly, breaking data correlation and stabilizing the neural network gradients.
*   **Target Network:** A secondary "frozen" network ($Q_{target}$) was used to evaluate the next state during the Bellman update. It was synchronized with the online network every 10,000 steps to prevent the "moving target" destabilization problem.
*   **Double DQN Extension:** Standard DQN suffers from overestimation bias because the `max` operator is used on noisy target Q-values. Double DQN decoupled action selection from evaluation. The *Online Network* was used to select the argmax action, while the *Target Network* evaluated the value of that specific action.

## 5. Experimental Setup
*   **Agent Type:** Double DQN
*   **Optimizer:** Adam (Learning Rate: 1e-4)
*   **Loss Function:** Smooth L1 Loss (Huber Loss)
*   **Exploration:** Epsilon-greedy, decaying linearly from 1.0 to 0.1 over the first 1,000,000 steps.
*   **Discount Factor ($\gamma$):** 0.99
*   **Training Duration:** 2,000,000 timesteps

## 6. Results
The Double DQN agent successfully learned to play Beam Rider. 

**Training Metrics:**
The agent exhibited a clear, upward convergence trend. The learning curves (available as `learning_curves.png`) demonstrate a steady climb from random noise to consistent scoring capability.

**Evaluation Metrics (Phase 6):**
The final model weights (`dqn_final.pth`) were evaluated over 10 independent episodes with a strictly minimized exploration rate ($\epsilon = 0.05$). 
*   **Average Reward:** 1058.40
*   **Maximum Reward:** 1380.00 (During the video recording phase, a peak run achieved **3560.0**)
*   **Minimum Reward:** 660.00
*   **Standard Deviation:** 245.74

## 7. Discussion (Ablation Studies)
To scientifically validate the architecture, ablation studies were conducted by systematically disabling components and observing the agent's failure to learn over 150,000 timesteps.

1.  **Without Target Network:** The loss values spiked wildly and failed to converge. The average episode reward fluctuated erratically near the random baseline (~200-300). The agent was unable to lock onto a consistent policy because the Q-value targets were constantly shifting, confirming the necessity of a delayed, frozen target network.
2.  **Without Experience Replay:** The agent exhibited severe catastrophic forgetting. Episode rewards flatlined near 100-150. Because consecutive frames in Beam Rider are highly correlated, the network overfit to immediate sequences and failed to generalize.
3.  **Without Frame Stacking:** The agent managed to learn basic shooting but completely failed at evasion, plateauing at ~300-400 points. Without the temporal context of 4 stacked frames, the agent was blind to the velocity of enemy projectiles, breaking the core requirements of the Markov Decision Process.

## 8. Limitations
1.  **Sample Inefficiency:** Like all vanilla DQN architectures, the agent is highly sample inefficient, requiring millions of frames of interaction to learn basic features that a human would recognize instantly.
2.  **Hardware Dependency:** The required CNN training relies heavily on CUDA acceleration. CPU training is prohibitively slow for the 2,000,000 timestep requirement.
3.  **Exploration Ceiling:** Epsilon-greedy exploration is naive. In states requiring complex sequences of precise actions to survive, random noise rarely discovers the optimal path.

## 9. Future Work
If the project were to be extended beyond the baseline constraints, the following enhancements would be prioritized:
1.  **Prioritized Experience Replay (PER):** Instead of sampling uniformly, the buffer could sample transitions with high Temporal Difference (TD) error, focusing the network on states it currently misunderstands.
2.  **Dueling Network Architecture:** Splitting the final dense layers into separate Value $V(s)$ and Advantage $A(s, a)$ streams would allow the agent to learn the inherent value of being in a safe state, regardless of the action taken.
3.  **Noisy Nets:** Replacing epsilon-greedy exploration with parametric noise added to the linear layers, allowing the network to learn how to explore in a state-dependent manner.

## 10. Conclusion
This project successfully demonstrated the fundamental principles of Deep Reinforcement Learning. By building the algorithms from scratch, it was proven that a Double DQN architecture, supported by a strict preprocessing pipeline, experience replay, and a target network, can learn to extract complex spatial-temporal features from raw pixels and execute a highly competent control policy in the Atari Beam Rider environment.
