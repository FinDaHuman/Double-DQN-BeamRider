# Phase 4: Baseline Training Results

## Experiment Setup
*   **Environment:** ALE/BeamRider-v5
*   **Agent:** Standard Deep Q-Network (Nature 2015 Architecture)
*   **Training Duration:** 2,000,000 timesteps
*   **Exploration:** Epsilon-greedy (linear decay from 1.0 to 0.1 over 1,000,000 steps)
*   **Target Network Update:** Every 10,000 steps

## Observations & Findings
*   *Note for Student: Review your TensorBoard logs (`tensorboard --logdir=logs`) to complete this section prior to your oral defense.*
*   **Convergence:** Did the agent's episode reward steadily increase? (Look at `Train/Episode_Reward`).
*   **Stability:** Were there massive drops in performance? Standard DQN is known for occasional catastrophic forgetting or sudden drops in Q-value accuracy due to overestimation.
*   **Loss Curve:** Did the Huber loss stabilize, or did it continue to grow? (A growing loss in DQN is often normal as the scale of Q-values increases, which is why we track reward more closely).

## Conclusion
The baseline DQN successfully completed its training loop. However, standard DQN inherently suffers from overestimation bias. Our next phase (Phase 5) implements Double DQN to address this theoretical limitation.