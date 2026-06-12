# Evaluation Report

This document records the quantitative evaluation of the trained Beam Rider agents (Phase 6).

## Evaluation Methodology
Evaluation differs from training in that we freeze the network weights and minimize the exploration rate (`epsilon`). 
During training, the agent takes random actions up to 100% of the time initially to discover the environment. 
During evaluation, we set `epsilon` to a very low value (e.g., `0.05`) so the agent relies almost entirely on its learned `Q(s,a)` values (exploitation).

**Metrics Tracked:**
*   **Average Reward:** Represents the expected performance of the policy.
*   **Max Reward:** Represents the peak capability/ceiling of the current policy.
*   **Min Reward:** Indicates worst-case performance or catastrophic failures.
*   **Standard Deviation:** Measures the stability and consistency of the policy across episodes.

---
## Evaluation: DoubleDQN (Final Model)
- **Episodes:** 10
- **Epsilon:** 0.05
- **Average Reward:** 1058.40
- **Max Reward:** 1380.00
- **Min Reward:** 660.00
- **Std Deviation:** 245.74

