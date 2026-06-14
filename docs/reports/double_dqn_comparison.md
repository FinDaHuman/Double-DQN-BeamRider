# Phase 5: Double DQN Comparison

## Motivation: The Overestimation Problem
In standard Q-Learning and DQN, the target for the Bellman update is calculated as:
$$Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-)$$

The fundamental issue is that the same network ($\theta^-$) is used to both **select** the best action (via the $\max$ operator) and **evaluate** the value of that action. Because the network is an estimator and inherently contains noise, taking the maximum over noisy estimates guarantees a positive bias. Over many steps, this bias compounds, leading the agent to massively overestimate the true value of states.

## Implementation: Double DQN
Double DQN (DDQN) elegantly solves this by decoupling selection from evaluation:
1.  **Selection:** We use the *Online Network* ($\theta$) to find the best action for the next state.
2.  **Evaluation:** We use the *Target Network* ($\theta^-$) to calculate the Q-value of that specifically chosen action.

The updated Bellman target is:
$$Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta); \theta^-)$$

By separating these steps, if the Online Network overestimates a suboptimal action and selects it, the Target Network (which has different weights) is unlikely to also overestimate it, thus pulling the target back down to a more realistic value.

## Experimental Setup
*   **Baseline:** Standard DQN (Trained for 2M steps)
*   **Experimental:** Double DQN (Trained for 2M steps)
*   *All other hyperparameters (learning rate, epsilon decay, capacity) remain identical.*

## Results & Comparison
*   *Note for Student: Run the Double DQN training (`python train.py` with `agent_type: DoubleDQN`) and compare the TensorBoard logs.*
*   **Reward Stability:** Does the Double DQN agent experience fewer sudden drops in episode reward compared to the baseline?
*   **Max Q-Values:** (If logged) Double DQN should theoretically show lower, more realistic estimated Q-values than the standard DQN.

## Conclusion
*   *(To be filled after Double DQN training is complete and evaluated)*