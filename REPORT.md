# Improving RL on the constrained Goddard rocket — report

**Goal.** Reproduce Section 7 of *CARL: Chat Assisted Reinforcement Learning*
(Das et al., arXiv:2610.31014) and show a measurable improvement over its
baseline RL agent.

**Setup.** 18 held-out scenarios: `m_fuel_0 ∈ {500, 2000, 4000} kg` ×
`Qmax ∈ {5, 7.5, 10, 15, 30, 45} kPa`, with `Tmax = 200 kN`, `cd = 1.5 m²`,
`m_dry = 1000 kg`. Dynamics are integrated with RK4 at `dt = 0.1 s`, splitting the
step exactly at fuel exhaustion and binary-searching the thrust whenever a step
would violate `q = ½ρv² <= Qmax` (clamped steps pay a 5% drag surcharge). This
single evaluator is used for every method.

**Training.** PPO with a Beta policy head over throttle, separate 256×256 tanh
actor/critic trunks, `γ = 1`, GAE(0.95), two epochs, target KL 0.05. The reward
is `Δh / 1000`, which telescopes to apogee. Domain randomisation is wider than
the evaluation grid and the 18 evaluation pairs are excluded. Budget per run:
120 updates × 512 steps × 256 environments (~16 M transitions), 3 seeds per
ablation.

## Results

| Method | Mean apogee (m) | % of best known | Light | Nominal | Heavy |
| --- | ---: | ---: | ---: | ---: | ---: |
| CasADi optimum (upper bound) | 22,766 | 100.0 | 100.0 | 100.0 | 100.0 |
| MC-Local-Greedy | 22,468 | 98.8 | 98.9 | 98.9 | 98.6 |
| Unconstrained Optimal | 20,026 | 93.8 | 100.0 | 98.7 | 82.9 |
| **R4 residual action** | **18,074** | **89.3** | **99.6** | **94.9** | 73.4 |
| R2 burnout penalty | 18,532 | 85.4 | 88.7 | 88.4 | **78.9** |
| R6 Q-cap + mode eval | 18,363 | 84.6 | 88.0 | 87.5 | 78.2 |
| R3 BC warm start | 17,944 | 83.3 | 88.4 | 84.9 | 76.5 |
| R0 unimodal Beta | 16,592 | 76.9 | 81.4 | 78.4 | 71.0 |
| R1 boundary Beta | 16,587 | 76.9 | 81.4 | 78.3 | 71.0 |
| R5 potential shaping | 16,582 | 76.8 | 81.1 | 78.3 | 71.0 |
| Bang-Bang | 16,427 | 75.8 | 79.3 | 77.6 | 70.4 |

RL numbers are means over three seeds; the evaluator already matches the
paper's Bang-Bang column to ≤ 1 m on 17 of 18 scenarios (see
`docs/reproduction.md`).

## Findings

1. **The residual action is the dominant improvement.** Parameterising the
   action as `u = clip(u_singular + δ)` lifts the agent from 76.9% to 89.3% of
   best-known, with 99.6% on light fuel and 94.9% on nominal. It reaches
   15,756 m on `N_M` (same as the unconstrained optimum) and 3,490 m on the
   light scenarios, versus 2,387–3,194 m for R0/R1 and 3,372 m for Bang-Bang.
2. **The burnout penalty, not the residual action, fixes heavy fuel.** R2 and
   R6 reach 34–35 km on `H_M`/`H_Lo`, where R4 stalls at 26–27 km. Heavy
   episodes are long (up to 200 s) and the reward per step is a 1,000-m
   normalised altitude increment, so a policy that burns early loses altitude in
   dense air; the one-sided penalty encodes exactly that.
   The two fixes are complementary and a combined variant is the obvious next
   step.
3. **The boundary-mode head is not a free win at this budget.** After 120
   updates R0 and R1 both sit at 76.9%, i.e. bang-bang quality; their difference
   shows up in the policy shape (R1 keeps mass near `u = 1` and a broad body,
   R0 concentrates on a single bump) and in entropy, not in apogee. The paper's
   separation required 500–1000 updates.
4. **BC warm start is a fast way to a mediocre policy.** It begins at a
   near-deterministic singular-arc imitation (entropy −2.3 nats) and ends at
   83.3%: better than R0/R1, worse than R2, and it loses the exploration that
   would let it improve further.
5. **Potential shaping did not help here.** With `γ = 1`, the potential-based
   term telescopes to a constant only if the potential is zero on terminal
   states; we enforce that, but the effect is still neutral (76.8%), because the
   altitude reward is already dense.
6. **Q-cap and mode evaluation** (R6) is the best heavy-fuel variant apart from
   R2 and removes the Beta train/eval mismatch, at a small cost on light fuel.

## Limitations

* 120 updates is roughly a quarter of the paper's budget; absolute numbers are
  below the paper's CARL column (which took 500–1000 updates). All comparisons
  here are internal and share the same budget.
* The paper does not specify how `t_switch` is chosen for the Unconstrained
  Optimal baseline; three heavy/tight rows cannot be reproduced and are
  reported as measured.
* The residual scale (0.5 of the throttle range) was not tuned. A smaller scale
  would likely preserve the heavy-fuel behaviour of R2 while keeping the
  light-fuel gains of R4.
* Three seeds give a coarse view of variance; the per-seed spread is plotted in
  `results/figures/seed_spread.png`.

## Artefacts

* `results/baselines.csv`, `results/eval_rl.csv`, `results/ablation_table.csv`
* `results/figures/` (learning curves, entropy, policy shape, ablation bars,
  heatmap, seed spread, trajectory profile)
* `results/checkpoints/` (21 PPO checkpoints, one per ablation and seed)
* `notebooks/demo.ipynb`, `web/index.html`, `web/slides.html`
