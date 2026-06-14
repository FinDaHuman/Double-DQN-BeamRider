# Phase 1: Environment Analysis

## Overview
This document analyzes the Atari 2600 Beam Rider environment using the Gymnasium `ALE/BeamRider-v5` environment. Understanding the raw environment is the critical first step before building a Deep Reinforcement Learning agent.

## Observation Space
*   **Raw Format:** `Box(0, 255, (210, 160, 3), uint8)`
*   **Description:** The environment returns a 3D NumPy array representing the RGB game screen. 
*   **Dimensions:**
    *   **Height:** 210 pixels
    *   **Width:** 160 pixels
    *   **Channels:** 3 (Red, Green, Blue)
*   **Values:** Each pixel value is an integer between 0 and 255.
*   **Implication for DQN:** This raw shape is $210 \times 160 \times 3 = 100,800$ data points per frame. This is too large for efficient training and includes unnecessary information (like color). We will need to implement a preprocessing pipeline to convert this to $84 \times 84$ grayscale images.

## Action Space
*   **Format:** `Discrete(9)`
*   **Description:** The agent has 9 discrete actions it can take at any given time step.
*   **Action Meanings:**
    0.  `NOOP` (No Operation - Do nothing)
    1.  `FIRE` (Shoot laser)
    2.  `UP` (Move forward in the grid)
    3.  `RIGHT` (Move right)
    4.  `LEFT` (Move left)
    5.  `UPRIGHT` (Move forward and right)
    6.  `UPLEFT` (Move forward and left)
    7.  `RIGHTFIRE` (Move right and shoot)
    8.  `LEFTFIRE` (Move left and shoot)
*   **Implication for DQN:** The output layer of our Q-Network will have exactly 9 neurons, each outputting the estimated Q-value for the corresponding action.

## Reward Structure
*   **Type:** Float.
*   **Description:** The reward represents the change in score from the previous frame. 
*   **Examples:** Destroying an enemy saucer or ship grants positive points. Simply surviving usually grants 0 reward, though advancing levels may give bonuses. 
*   **Implication for DQN:** Because the baseline reward for most steps is 0.0, the environment exhibits sparse rewards. The agent will have to explore sufficiently to encounter enemies and learn that shooting them yields positive value.

## Episode Termination
*   **Conditions:** The episode terminates (`terminated=True` or `truncated=True`) when the player loses all their lives or reaches the maximum frame limit defined by the environment (usually 108,000 frames).
*   **Implication for DQN:** We must properly handle the `done` flag in our Replay Buffer and Bellman Equation. If a state is terminal, the target Q-value is simply the immediate reward ($Q(s,a) = r$), because there is no future state to bootstrap from.
