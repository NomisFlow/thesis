# Brainstorming and Ideas Log

This document automatically collects ideas, notes, and brainstorming from our sessions.

---

### Session: 2025-11-16

**Topic:** Mitigating Contrastive Loss Failure in TMD Training

During a training run on a robotics task, we observed that the contrastive learning component of TMD was failing (loss stagnant at `log(N)`, accuracy at `1/N`). We brainstormed several strategies to address this.

#### 1. Architectural Changes to Fight Representation Collapse

*   **Asymmetric Architecture (BYOL/SimSiam Style):** Instead of a symmetric contrastive loss, modify the architecture to be asymmetric. Use a "predictor" head on the online network to predict the representations of a stop-gradient target network. This is a proven method to prevent collapse.
*   **Decorrelation Loss (Barlow Twins/VICReg Style):** Add a new loss term to the main `L_TMD` objective. This loss would penalize the cross-correlation between the feature dimensions of the latent vectors, forcing the model to use its full representational capacity and preventing features from collapsing.

#### 2. Data and Sampling Changes for Hard Negatives

*   **Adjust Future Step (`k`) Sampling:** The current method samples `k` from a geometric distribution. For slow-moving robots, this might create positive pairs that are too similar. We should experiment with sampling `k` from a uniform distribution `U(k_min, k_max)` with a higher `k_min` to ensure the positive sample is distinct.

#### 3. Loss Function Rebalancing and Reshaping

*   **Systematic Tuning of `ζ` (zeta):** The `ζ` hyperparameter balances the contrastive and invariance losses. The current dominance of the contrastive loss suggests `ζ` might be too low. We should try systematically increasing `ζ` to give more weight to the geometric invariance losses, which could provide a stronger structural gradient to escape the local minimum.
*   **Implement Loss Annealing for `ζ`:** Instead of a fixed `ζ`, we should implement a schedule. Start with a low `ζ` to prioritize the contrastive loss for building a coarse representation, then gradually increase `ζ` over training. This would shift the focus to refining the representation with the geometric invariance constraints, creating a more stable learning curriculum. This is mentioned as a theoretical possibility in the TMD paper and should be implemented.

---

### Session: 2025-11-16 (cont.)

**Topic:** Trajectory-based Auxiliary Loss for Richer Representations

**Proposal:** Instead of the standard contrastive process on a single state-action pair `(s_i, a_i)` to predict a future state `s_{i+k}`, we should use the entire trajectory segment from `s_i` to `s_{i+k}` to provide more information about dynamics.

**Initial Feasibility Concerns:** A naive implementation that replaces the core representation with a trajectory-level one (e.g., from an RNN or Transformer) creates a mismatch with the single-step policy extraction mechanism used at inference time (`argmin_a d(φ(s, a), ψ(g))`).

**Viable Hybrid Solution:** The proposal is highly viable when framed as an **auxiliary loss** that runs in parallel with the main TMD objective.

1.  **Core TMD Stays:** The main encoders (`ψ`, `φ`), distance network (`d`), and primary loss (`L_TMD`) remain unchanged. The policy extraction method is also preserved.
2.  **Add Auxiliary Loss (`L_aux`):**
    *   A new sequence encoder (e.g., a small Transformer) is added.
    *   During training, this encoder takes a sequence of latent states `(ψ(s_i), ..., ψ(s_{i+k-1}))` and produces a context vector `c`.
    *   A contrastive loss `L_aux` is calculated, where the context `c` must predict the final latent state `ψ(s_{i+k})`.
    *   The total loss becomes `L_total = L_TMD + β * L_aux`.
3.  **Benefit:** The gradients from `L_aux` flow back into the main state encoder `ψ`. This forces `ψ` to learn representations that are not only good for 1-step predictions but are also consistent and predictable when viewed as part of a longer dynamic sequence. This enriches the state representations with long-term dynamics information, which should make the primary TMD task easier and more robust, potentially fixing the contrastive learning failure.

---

### Session: 2025-11-17

**Topic:** Alternative Architecture: History-Conditioned TMD (HC-TMD)

**Proposal:** This is a more fundamental architectural change compared to the auxiliary loss idea. Instead of augmenting the existing TMD encoders, we replace the state encoder (`ψ`) entirely with a Transformer-based architecture that directly processes a history of recent states. The goal is to bake temporal dynamics (like velocity and momentum) directly into the state representation.

#### 1. Proposed Architecture

The core of this idea is an asymmetric encoder design to handle the difference between a dynamic state history and a static goal image.

*   **State History Encoder (`ψ_hist`):**
    *   **Input:** A sequence of the last `H` states, `(s_{t-H}, ..., s_t)`. For image-based environments, this is a sequence of `H` images.
    *   **Mechanism:** A Transformer encoder processes this sequence. Its self-attention mechanism is expected to learn the relationships between recent states to infer properties like motion and momentum.
    *   **Output:** A single, context-rich latent vector `h_t` that represents the current state `s_t` conditioned on its recent history.

*   **Goal Encoder (`ψ_goal`):**
    *   **Input:** A single, static goal state `g`.
    *   **Mechanism:** This remains a standard Convolutional Neural Network (CNN) encoder, identical to the one used in the original TMD. A Transformer is not suitable here as the goal is a static target without a temporal sequence.
    *   **Output:** A latent goal vector `ψ_goal(g)`.

*   **State-Action Encoder (`φ_hist`):**
    *   **Input:** The history-conditioned state vector `h_t` and the current action `a_t`.
    *   **Mechanism:** A simple MLP that combines the state and action representations.
    *   **Output:** A latent state-action vector `φ_hist(h_t, a_t)`.

The final distance calculation for the TMD loss would then become: `d(φ_hist(h_t, a_t), ψ_goal(g))`.

#### 2. Learning Process

The fundamental TMD loss objectives (`L_NCE`, `L_I`, `L_T`) would be retained. However, they would now operate on these more sophisticated, history-aware representations.

A key challenge is ensuring that the **State History Encoder** and the **Goal Encoder** produce representations in a **semantically aligned latent space**. The network must learn to distill a dynamic sequence of observations (`h_t`) into a vector that is directly comparable with a vector from a single, static goal image (`ψ_goal(g)`). The existing TMD contrastive and Bellman-like losses are responsible for enforcing this alignment.

#### 3. Advantages vs. Disadvantages

**Potential Advantages:**

*   **Implicit Dynamics Modeling:** The primary benefit. The Transformer can learn complex physical properties like velocity, acceleration, and momentum directly from raw state sequences, creating a far richer and more accurate state representation.
*   **Handling Partial Observability:** In scenarios where a single frame `s_t` is ambiguous, the history `(s_{t-H}, ..., s_t)` can provide the necessary context to disambiguate the true state of the environment.
*   **Architectural Elegance:** Integrates history directly into the main representation, which can be seen as a more unified solution compared to adding an auxiliary loss.

**Significant Challenges and Risks:**

*   **Representation-Goal Mismatch (High Risk):** This is the most critical challenge. Forcing a Transformer encoder (processing a sequence) and a CNN encoder (processing a single image) to output into the same, comparable latent space is very difficult. If this alignment fails, the learned distance `d(h_t, ψ_goal(g))` will be meaningless, and the policy will fail.
*   **Computational and Memory Cost:** Processing a sequence of `H` images through a Transformer at every training step is significantly more computationally expensive than the current single-image CNN approach. This will dramatically slow down training and increase VRAM requirements.
*   **Inference Latency:** At execution time, the policy must maintain a buffer of the last `H` states and perform a full Transformer forward pass for every action. This could introduce significant latency, making it unsuitable for real-time robotic control.
*   **Sample Inefficiency:** Transformers are known to be data-hungry. This architecture would likely require a much larger and more diverse offline dataset to be effective.

#### 4. Comparison with the Auxiliary Loss Idea

*   **HC-TMD** is a **full replacement** of the state representation model. It's a high-risk, high-reward strategy that fundamentally changes the architecture.
*   The **Auxiliary Loss** approach is a **conservative enhancement**. It keeps the proven TMD architecture intact while adding a secondary task to enrich the representations. This is lower risk and computationally cheaper.

#### 5. Strategies to Mitigate Representation Mismatch in HC-TMD

Given the high risk of representation mismatch between the history-based state encoder and the static goal encoder, the following strategies can be employed to enforce alignment in the latent space.

*   **1. Shared Image Encoder Backbone (Most Promising):**
    *   **Concept:** Instead of two entirely separate encoders, use a single, shared **base CNN encoder** (e.g., a ResNet stem) as the first stage for both pathways.
    *   **Implementation:**
        *   **State History Encoder (`ψ_hist`):** Each image in the history sequence `(s_{t-H}, ..., s_t)` is first passed through the shared CNN. The resulting sequence of feature vectors is then fed into the Transformer.
        *   **Goal Encoder (`ψ_goal`):** The single goal image `g` is passed through the *exact same* shared CNN to get its feature vector.
    *   **Benefit:** This forces both representations to be built from the same foundational visual vocabulary, making the alignment task significantly easier for the subsequent Transformer and MLP layers.

*   **2. Explicit Representation Alignment Loss:**
    *   **Concept:** Add an auxiliary loss term that explicitly minimizes the distance between the two different representations of the same physical state.
    *   **Implementation:** For a trajectory ending at state `s_T`, compute both the history-based representation `h_T = ψ_hist(s_{T-H}, ..., s_T)` and the static goal representation `g_T = ψ_goal(s_T)`. Add a loss term, `L_align`, that pulls these two vectors together (e.g., using MSE or cosine similarity). The total loss becomes `L_total = L_TMD + γ * L_align`.
    *   **Benefit:** Provides a direct, unambiguous supervisory signal to the model to align the two latent spaces.

*   **3. Projection Heads:**
    *   **Concept:** Add a small, separate MLP (a "projection head") on top of both the history encoder and the goal encoder. The final TMD distance `d(...)` is calculated using the outputs of these projectors.
    *   **Benefit:** This is a standard technique from self-supervised learning that can improve stability. It gives the main encoders more freedom to learn rich features while the projection heads specialize in transforming them into a space where the distance metric is calculated.

*   **4. Coordinated Data Augmentation:**
    *   **Concept:** Apply the *exact same* random data augmentation (e.g., random crop, color jitter) to all images in the history sequence `(s_{t-H}, ..., s_t)` *and` to the goal image `g` within the same training sample.
    *   **Benefit:** Encourages the encoders to learn features that are invariant to the same set of visual transformations, naturally leading to better alignment.

*   **Recommended Implementation Strategy:**
    For a practical first attempt, a combination of the following is recommended:
    1.  **Start with the Shared Image Encoder Backbone.** This is the most fundamental and likely most impactful change.
    2.  **Add the Explicit Representation Alignment Loss.** This provides a strong, direct signal to solve the exact problem we're worried about.
    3.  **Apply Coordinated Data Augmentation.** This is a simple and computationally cheap way to improve robustness and aid alignment.

#### 6. Architectural Choice for the History Encoder: Transformer vs. RNN/LSTM

While the HC-TMD proposal specifies a Transformer, it is worth considering alternatives like LSTMs or GRUs for the `ψ_hist` sequence encoder.

*   **RNN/LSTM-based Approach:**
    *   **Mechanism:** The encoder would maintain a hidden state `h`. At each step `t`, it would update its state `h_t = RNN(h_{t-1}, s_t)`, where `s_t` is the current observation's feature vector from the shared CNN. This `h_t` becomes the history-conditioned representation.
    *   **Key Advantage - Inference Efficiency:** This is the most significant benefit. RNNs are highly efficient at inference time. To get the next state representation, you only need the previous hidden state and the new observation. This avoids the costly operation of re-processing the entire history window at every single timestep, which is a major concern for real-time robotics.
    *   **Trade-off:** The primary drawback is the potential for weaker modeling of long-range dependencies compared to a Transformer's self-attention mechanism. RNNs can struggle with the "vanishing gradient" problem and may have difficulty weighting the importance of events that occurred far back in the history sequence.

*   **Transformer-based Approach (Recap):**
    *   **Mechanism:** Processes the entire history sequence `(s_{t-H}, ..., s_t)` in parallel to compute `h_t`.
    *   **Key Advantage - Representational Power:** Superior at capturing complex, long-range dependencies and relationships between all points in the history window. Highly parallelizable during training on modern hardware.
    *   **Trade-off:** High computational and memory cost, both in training and inference. The inference latency is a critical concern.

*   **Other Alternatives (e.g., TCNs):**
    *   Temporal Convolutional Networks (TCNs) use causal convolutions and could offer a middle ground, providing parallelizable training and a large receptive field without the quadratic complexity of Transformers.

*   **Recommendation for Initial Implementation:**
    *   **Start with an LSTM or GRU.** This is the most pragmatic and lowest-risk starting point. It directly addresses the critical "Inference Latency" problem of the Transformer-based design. The potential limitations in modeling very long-range dependencies are an acceptable trade-off for a first implementation.
    *   **Path Forward:** If the LSTM-based model proves to be limited by its ability to capture temporal patterns, it can then be "upgraded" to a more powerful (and expensive) Transformer architecture. This creates a clear, iterative development path.

---

### Session: 2025-11-22

**Topic:** Boosting / Ensemble Trajectory Stitching

**Proposal:** Use an ensemble of "weak learners" (short-horizon predictors or policies) to generate multiple potential future trajectories, then aggregate and stitch them to form a robust long-horizon plan. This draws inspiration from AdaBoost (combining weak learners) and existing Model-Based RL approaches (like PETS).

#### 1. Concept Translation to TMD/Offline RL

*   **"Weak Learner" $ightarrow$ Short-Horizon Dynamics Model:** Instead of a single policy attempting to solve the entire task, we use "weak learners" to predict the *outcome* of actions (dynamics: $s_t, a_t ightarrow s_{t+n}$) rather than the distance.
*   **"Redoing Experiment" $ightarrow$ Ensemble Generation:** Run this dynamics prediction multiple times (or use an ensemble of models) to generate a *distribution* of possible near-future states. This averages out the noise in the transition dynamics.
*   **"Averaging Out" $ightarrow$ Robust Trajectory Segments:** Aggregate these predictions to form a "canonical," high-confidence trajectory segment. This gives us a reliable piece of movement we can trust.
*   **"Stitching" $ightarrow$ TMD-Guided Optimization:** Use the **Base TMD** (which predicts *steps/distance*) as the Critic.
    *   The Ensemble *proposes* where we can go (Generation).
    *   The Base TMD *judges* if that destination is closer to the goal (Evaluation).
    *   We stitch the segments that the TMD Critic deems most valuable.

#### 2. Implementation Pathways

*   **Path A: Inference-Time Planning (Recommended):**
    *   Use the trained TMD model as a critic/heuristic.
    *   At each step, roll out $K$ short trajectories using a base policy or random shooting.
    *   Score each trajectory using the TMD distance to the goal.
    *   Execute the first action of the best trajectory (or an average of the top $k$).
    *   This effectively "stitches" the best short-term moves to solve the long-term task.

*   **Path B: Data Augmentation (Offline):**
    *   Use the ensemble approach to generate synthetic "stitched" trajectories from the offline dataset *before* training the final policy.
    *   Add these optimized trajectories to the replay buffer to help the actor learn better behaviors.

#### 3. Relevance to Current Issues

This approach is orthogonal to the "Contrastive Loss Failure" but highly relevant to the overall goal of robust navigation. If the contrastive loss is fixed (via KAN-6), the TMD metric becomes reliable. This "Boosting/Ensemble" idea is then the perfect mechanism to *exploit* that reliable metric for superior performance.

---

### Session: 2025-11-22

**Topic:** Action Stacking for Improved Dynamics Modeling

**Proposal:** Explicitly incorporate prior actions into the current action decision-making process to model inertia and momentum, leading to smoother, more physically compliant, and predictable robot behavior. This will help both dynamics prediction and representation learning.

#### 1. Why it Works

*   **Physics Compliance:** Robots have mass and cannot change actions instantaneously. Conditioning on $a_{t-1}$ ensures actions are physically plausible.
*   **Smoother Control:** Reduces jerky "bang-bang" control, making actions and resulting state transitions more predictable for the TMD critic.
*   **Implicit State Information:** Previous actions provide context about the robot's current momentum/velocity, which might not be fully captured by a single static state observation ($s_t$).

#### 2. Concrete Application: Action Stacking (Recommended)

*   **Concept:** Augment the policy network's input to include the previous action(s). This allows the network to *learn* the optimal way to integrate past actions with the current state to produce a new action.
*   **Implementation:**
    *   **Current Policy Input:** `latent_state = ψ(s_t)`
    *   **New Policy Input:** `[ψ(s_t), a_{t-1}]` (concatenating the latent state with the previous action).
    *   The policy MLP would then receive this combined vector to output $a_t$.
*   **Advantages:**
    *   **Low Computational Cost:** Simple concatenation, minimal overhead.
    *   **Learned Integration:** The network autonomously decides how much influence $a_{t-1}$ should have, adapting to different phases of movement (e.g., high influence during steady motion, low during sharp turns).
    *   **Flexibility:** Can easily extend to include more past actions (e.g., `[ψ(s_t), a_{t-1}, a_{t-2}, ...]`) or their derivatives.

#### 3. Considerations

*   **Initial Action:** For the very first step ($t=0$), $a_{t-1}$ will be undefined. A common solution is to initialize it to zero or a learned parameter.
*   **State Dependency:** It's crucial that the current action ($a_t$) still heavily depends on the current state ($s_t$) and the goal ($g$). Action stacking provides *additional context*, not a replacement for state information.

This approach should make the learned policy outputs more stable and physically grounded, which can be beneficial for the dynamics models involved in the "Boosting/Ensemble" approach and the overall predictability for TMD.

---

### Session: 2025-11-24

**Topic:** Interval Quasimetric Embeddings (IQE) for TMD

**Proposal:** Replace or augment the current Metric Residual Network (MRN) distance with Interval Quasimetric Embeddings (IQE). This is based on the suggestion to use IQE as a more robust property-enforcing building block for quasimetric learning.

**Reference:** "Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings" (arXiv:2211.15120).

#### 1. Analysis of Existing Implementation
We identified that `tmd.py` already contains an `iqe_distance` method.
*   **Status:** Present but disabled (`use_iqe=False` by default).
*   **Logic:** Implements the "Deep Quasimetric" logic using `argsort` and `cumsum` to calculate the intersection of intervals in the latent space.
*   **Reduction:** Uses a learned convex combination (`alpha`) of `mean` and `max` over components, which matches the "maxmean" reduction from the literature.

#### 2. Detailed Implementation Plan (TDD-Driven) - Branch: `feat/KAN-14-iqe-distance`

**Phase 1: Verification & Testing**
*   **Create Test Suite:** Create `tmd-release/tests/test_iqe.py`.
*   **Test Case 1: Basic Properties:**
    *   `d(x, x) == 0`
    *   `d(x, y) >= 0`
    *   `d(x, y)` != `d(y, x)` (Asymmetry check).
    *   Triangle Inequality check: `d(x, z) <= d(x, y) + d(y, z)` (Should hold approximately or exactly depending on formulation).
*   **Test Case 2: Gradient Flow:** Ensure gradients can propagate through the `argsort` / `take_along_axis` operations (JAX handles this, but verification is needed).
*   **Test Case 3: Numerical Stability:** Test with very small and very large values.

**Phase 2: Integration & Experimentation**
*   **Configuration:** Enable `use_iqe=True` in a config file or command line flag.
*   **Hyperparameters:** The number of components (`components=8` default) might need tuning for IQE specifically.
*   **Jira:** Tracked in `KAN-14`.

#### 3. Caveats & Risks
*   **Performance:** The `argsort` operation in JAX/XLA can be slower than simple matrix multiplications used in MRN. We need to monitor step time.
*   **Latent Dimension:** IQE splits the latent dimension $D$ into $K$ components. $D$ must be divisible by $K$. The current default $D=512, K=8$ is fine ($64$ dims/component), but if we change $D$, we must be careful.
