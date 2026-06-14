# Phase 0: Literature Review

## 1. Markov Decision Process (MDP)
### What is it?
A Markov Decision Process is a mathematical framework used to describe an environment in reinforcement learning. It formally defines the decision-making problem where outcomes are partly random and partly under the control of a decision-maker (the agent).

### Components
An MDP is defined by a tuple $\langle S, A, P, R, \gamma \rangle$:
*   **$S$ (State Space):** A set of all possible states in the environment.
*   **$A$ (Action Space):** A set of all possible actions the agent can take.
*   **$P$ (Transition Probability):** $P(s' | s, a)$, the probability of transitioning to state $s'$ given the current state $s$ and action $a$. (In deterministic environments like standard Atari, this probability is usually 1 for the resulting state).
*   **$R$ (Reward Function):** $R(s, a, s')$, the immediate reward received after transitioning from $s$ to $s'$ via action $a$.
*   **$\gamma$ (Discount Factor):** A value between $0$ and $1$ that determines the importance of future rewards compared to immediate rewards.

### The Markov Property
The core assumption of an MDP is the **Markov Property**: the future state depends only on the current state and action, not on the sequence of events that preceded it. Mathematically: $P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1} | S_t, A_t)$. 

*Oral Defense Note: In Atari games, a single frame does not satisfy the Markov Property because it lacks velocity/directional information. We stack multiple frames (usually 4) to approximate the Markov Property.*

---

## 2. The Bellman Equation
### What is it?
The Bellman Equation is a fundamental recursive equation in dynamic programming and RL. It decomposes the value function into two parts: the immediate reward and the discounted value of the subsequent state.

### The Equation
For the optimal action-value function $Q^*(s, a)$, the Bellman Optimality Equation is defined as:
$$Q^*(s, a) = \mathbb{E} [ r + \gamma \max_{a'} Q^*(s', a') ]$$
Where:
*   $Q^*(s, a)$ is the maximum expected return starting from state $s$, taking action $a$, and acting optimally thereafter.
*   $r$ is the immediate reward.
*   $\gamma$ is the discount factor.
*   $\max_{a'} Q^*(s', a')$ is the estimated value of the best action in the next state $s'$.

### Why is it important?
It provides the foundation for Q-Learning. Instead of waiting until the end of an episode to evaluate how good an action was, we can "bootstrap" using our current estimate of the next state's value.

---

## 3. Q-Learning
### What is it?
Q-Learning is a model-free, off-policy reinforcement learning algorithm. It aims to learn the optimal action-value function (Q-function) directly, without needing a model of the environment's transition dynamics (the $P$ in MDP).

### How it works
It uses a table (Q-table) to store the Q-value for every state-action pair. After taking an action $a$ in state $s$, receiving reward $r$, and arriving in state $s'$, the Q-value is updated using the Q-Learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

*   $\alpha$: Learning rate.
*   The term $\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$ is the **Temporal Difference (TD) Error**.

### Limitations
Standard Q-Learning relies on a table. If the state space is continuous or highly dimensional (like the $84 \times 84 \times 4$ pixels of an Atari game), a Q-table is impossible to store and update. This leads to the need for Deep Q-Networks.

---

## 4. Deep Q-Network (DQN)
### What is it?
DQN combines Q-Learning with Deep Neural Networks. Instead of a Q-table, it uses a Convolutional Neural Network (CNN) as a function approximator to map raw pixel inputs (states) to Q-values for all possible actions.

### Architecture in Atari
*   **Input:** $84 \times 84 \times 4$ tensor (4 stacked grayscale frames).
*   **Hidden Layers:** Several Convolutional layers to extract spatial features, followed by fully connected layers.
*   **Output:** A vector containing a Q-value for every valid action in the game.

### Loss Function
The network is trained to minimize the Mean Squared Error (MSE) or Huber Loss between the predicted Q-value and the target Q-value (derived from the Bellman Equation):
$$Loss = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

---

## 5. Experience Replay
### What problem does it solve?
In standard reinforcement learning, the agent learns from sequential experiences as it interacts with the environment. This causes two major issues for neural networks:
1.  **Correlated Data:** Sequential frames in a game are highly correlated. Neural networks assume independent and identically distributed (i.i.d) data, and correlated data can cause the network to overfit to local sequences and diverge.
2.  **Forgetting:** Once a rare but important experience occurs, it is used for one update and discarded. The agent might "forget" how to handle that situation.

### How it works
The agent stores its transitions $(s, a, r, s', done)$ into a large cyclic buffer (Experience Replay Buffer). During training, the agent samples a random mini-batch (e.g., 32 transitions) from this buffer.
*   **Advantage 1:** Breaks the correlation between consecutive samples, stabilizing training.
*   **Advantage 2:** Allows the agent to learn from past experiences multiple times, increasing data efficiency.

---

## 6. Target Network
### What problem does it solve?
In the DQN loss function, both the prediction $Q(s, a; \theta)$ and the target $r + \gamma \max_{a'} Q(s', a'; \theta)$ use the *same* neural network weights ($\theta$). Because the target depends on the same weights that are being updated, the target is constantly moving. This creates a "chasing your own tail" effect, leading to severe instability and divergence.

### How it works
DQN introduces a second, identical neural network called the **Target Network** ($\theta^-$). 
*   The **Online Network** ($\theta$) is updated at every training step and is used to select actions.
*   The **Target Network** ($\theta^-$) is used exclusively to calculate the target Q-values during the loss calculation.
*   The Target Network's weights are "frozen" and are only periodically updated (e.g., every 10,000 steps) by copying the weights from the Online Network. This provides a stable target for the loss function.

---

## 7. Atari Learning Environment (ALE)
### What is it?
The ALE is an interface to hundreds of Atari 2600 game environments, widely used as a benchmark for Deep Reinforcement Learning. We interface with it using the `gymnasium` library.

### Beam Rider Specifics
*   **Observation Space:** By default, it's a $210 \times 160 \times 3$ RGB image representing the game screen. We must preprocess this to $84 \times 84$ grayscale.
*   **Action Space:** Discrete. The agent can choose from a predefined set of actions (NOOP, FIRE, LEFT, RIGHT, etc.).
*   **Reward Structure:** The environment returns the change in score as the reward at each time step. The agent must learn to maximize this total score over the episode.