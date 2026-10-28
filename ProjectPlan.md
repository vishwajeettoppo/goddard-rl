# Improving RL for the Constrained Goddard Rocket
Reproduction and improvement of results from "CARL: Chat Assisted Reinforcement
Learning" (Das et al., arXiv:2610.31014), Section 7.

## 1. Goal
Maximize rocket apogee under a dynamic-pressure limit (Qmax) using RL, then show
measurable improvement over a baseline RL agent.

Deliverables:
- Working environment, baselines, and PPO agents
- Ablation table showing each improvement
- Plots (altitude, velocity, throttle vs time)
- Live demo and slides

## 2. Repo layout
```
goddard-rl/
├── README.md
├── requirements.txt
├── configs/
│   ├── scenarios.yaml        # 18 eval scenarios (Table 5)
│   └── train.yaml            # PPO + randomization ranges
├── src/
│   ├── env.py                # RK4 dynamics, Qmax clamp, apogee termination
│   ├── baselines.py          # Bang-Bang, MC-Local-Greedy, Unconstrained Optimal
│   ├── ppo.py                # PPO with Beta policy head
│   ├── features.py           # observation builders (base, +burnout, +energy)
│   ├── train.py              # training entry point (flags select ablation)
│   ├── evaluate.py           # rolls out all methods on 18 scenarios
│   └── plots.py
├── results/                  # csv tables, checkpoints, figures
├── notebooks/demo.ipynb      # live demo for the presentation
└── reference/carl/           # fork of github.com/srdas/carl (read-only reference)
```

## 3. Setup
```bash
git clone <your-repo> goddard-rl && cd goddard-rl
git clone https://github.com/srdas/carl reference/carl   # reference only
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pyyaml pandas torch numba tqdm
```
Optional (true optimality benchmark): `pip install casadi`

## 4. Workflow

### Phase 1: Environment (src/env.py)
Implement from paper Sections 7.1-7.2:
- State (h, v, m). Dynamics: h' = v; v' = (T - D)/m - g; m' = -T/ce
- Drag D = cd * rho(h) * v|v|, rho = 1.225 * exp(-h/8500)
- Constants: g=9.81, Tmax=200000 N, ce=0.01*Tmax, m_dry=1000 kg, cd=1.5, dt=0.1 s
- RK4 integration; split the step exactly when fuel runs out
- Qmax: q = 0.5*rho*v^2. Before each step, if throttle would violate Qmax at t+dt,
  binary-search the largest feasible T. Clamped steps get a 5% drag surcharge.
- Terminate at apogee (v goes from >0 to <=0). Reward = peak altitude.
- Speed: vectorize over many parallel envs in numpy or use numba.

Validation checkpoint: with u=1 (Bang-Bang), results should be close to Table 7
(e.g., 500 kg / 5000 Pa -> ~3372 m).

### Phase 2: Baselines (src/baselines.py)
1. Bang-Bang: u=1 until fuel is gone.
2. MC-Local-Greedy: 30-segment piecewise-constant throttle; sample 500 schedules,
   refine the top 8 with coordinate descent (2 coarse passes + 1 fine pass, grid step 0.05).
3. Unconstrained Optimal: full thrust, then singular arc (paper eq. 8), then coast.
   The paper does not give t_switch; find it as the first time the eq. 8 thrust
   falls below Tmax. Validate against the unconstrained column of Table 7
   (3505 / 15805 / 50165 m).

### Phase 3: Baseline RL (src/ppo.py, train.py)
- PPO actor-critic; actor outputs alpha, beta = softplus(x) + eps; u ~ Beta(alpha, beta)
- Eval is deterministic: u = alpha / (alpha + beta)
- Reward: delta_h / H per step, gamma = 1 (telescopes to final altitude)
- Domain randomization over mfuel0, Tmax, cd, Qmax with ranges WIDER than the eval grid.
  Never train on the 18 eval scenarios.
- Large batches, parallel envs, advantage normalization, entropy logging.
- Run R0 (eps=1, unimodal) and R1 (eps=0.01, boundary mode).
  Log policy entropy per update; expect R0 to collapse.

### Phase 4: Improvements (one at a time)
| ID | Change | Why |
|----|--------|-----|
| R2 | + burnout features and one-sided early-burnout penalty (paper's revisit 1) | Prevents burning fuel early in thick air |
| R3 | + behavior-cloning warm start from MC-Local-Greedy or the singular-arc law | Paper admits it did not use this; targets heavy-fuel gap |
| R4 | Residual action: u = clip(u_singular + delta) | Bakes in bang-arc-bang structure |
| R5 | Replace burnout penalty with potential-based shaping (energy potential) | Preserves the optimal policy |
| R6 | (optional) deterministic-policy method (SAC/TD3) or mode-based eval | Fixes Beta train/eval mismatch for U-shaped policies |

Also audit: the paper's quadratic t_target = 75x - 8x^2 (x = mfuel0/1000) goes
negative near x ~ 9.4. Fit any such calibration on training-distribution scenarios only.

### Phase 5: Evaluation (src/evaluate.py)
- Roll out every method once per scenario (the dynamics are deterministic)
- Train each RL variant with 5+ seeds; report mean +/- std
- Metrics: apogee (m); % of best-known per scenario; worst-case scenario;
  constraint violations; runtime per rollout
- Report heavy fuel + tight Qmax scenarios separately
- Optional: direct collocation (CasADi) as a true optimum reference

### Phase 6: Presentation
1. Problem: specification gap (why RL fails on underspecified problems)
2. CARL idea: 10 steps, human owns Steps 1-7, AI owns 8-9, revisit step
3. Our reproduction: baseline numbers vs the paper
4. Failure demo: R0 entropy collapse vs R1 (plot entropy and altitude)
5. Improvements: ablation table, trajectory plots
6. Limitations and takeaways
Live demo (notebook): pick a scenario, run all methods, plot the three-panel profile.

## 5. Suggested timeline
- Week 1: env + baselines matching Table 7 non-RL rows
- Week 2: PPO + Beta head, reproduce R0/R1
- Week 3: improvements R2-R5, ablation runs
- Week 4: seeds, plots, report, slides, demo rehearsal

## 6. Known risks
- t_switch and several hyperparameters are not specified in the paper
- Heavy-fuel episodes are long (~150 s = 1500 steps); vectorize or training will crawl
- Absolute numbers may not match Table 7 exactly; report your own and explain gaps
- The paper's text claims CARL is within 3.1% of the best method in every scenario,
  but Table 7 shows larger gaps in heavy-fuel cases (3.1% looks like an average)
- Confirm scope (Goddard track vs GBWM vs framework) with your professor early

## 7. References
- Paper: arXiv:2610.31014 (Sections 7.1-7.7, Tables 5-7, Appendix C)
- Goddard problem: Tsiotras & Kelley 1991; Bonnans, Martinon, Trelat 2008
- PPO: Schulman et al. 2017; reward shaping: Ng, Harada, Russell 1999