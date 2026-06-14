# Contributing to Double-DQN-BeamRider

Thank you for your interest in contributing to this educational Deep Reinforcement Learning project! 

This repository was originally created as a university assignment to demonstrate a from-scratch implementation of a Double Deep Q-Network without relying on high-level RL libraries.

## How to Contribute

Since this is an educational baseline repository, contributions are welcome, particularly those that:
1. **Fix Bugs:** Resolve any runtime errors, typos in documentation, or mathematically incorrect Bellman updates.
2. **Improve Documentation:** Enhance the "explainability" of the code (e.g., expanding the docstrings).
3. **Add Optional Extensions:** Add advanced DQN variants (like Dueling DQN, Prioritized Experience Replay, or Noisy Nets) as *optional* extensions, provided they do not overwrite or obscure the baseline Double DQN implementation.

## Pull Request Process
1. Fork the repository.
2. Create a new branch for your feature or bugfix (`git checkout -b feature/amazing-feature`).
3. Ensure your code strictly avoids high-level black-box RL libraries (like Stable-Baselines3). Keep it in pure PyTorch.
4. Commit your changes with clear, descriptive messages (`git commit -m "feat: Add Dueling architecture"`).
5. Push to your branch (`git push origin feature/amazing-feature`).
6. Open a Pull Request on GitHub.

## Code of Conduct
Please be respectful and collaborative in issues and code reviews. This project serves as a learning resource for others, so detailed explanations of *why* a change improves the code are highly encouraged.