# Reference results: what reproduces and what does not

This note records the measured deviation between our shared evaluator and
Table 7 of Das et al., *CARL: Chat Assisted Reinforcement Learning*
(arXiv:2610.31014), Section 7. All numbers are produced by
`scripts/run_baselines.py` and stored in `results/baselines.csv`; regenerate
them with:

```bash
.venv/bin/python scripts/run_baselines.py --no-resume
```

## Shared evaluator

`src/env.py` implements the paper's Section 7.2 evaluator: RK4 over `(h, v, m)`
at `dt = 0.1 s`, exact splitting of the step at fuel exhaustion, and a binary
search over thrust whenever the nominal step would violate `q <= Qmax`. Clamped
steps pay a 5% drag surcharge. Every baseline, every RL rollout and every
figure in this repository uses exactly that integrator.

Two implementation details are not stated in the paper and were calibrated
against Table 7:

* **Bisection depth.** `n_bisect = 8` (the paper only says "binary search").
  Deeper searches change the Bang-Bang column by less than 1 m.
* **Surcharge placement.** The surcharge applies to the whole clamped step
  (search and final advance). Applying it only during the search is
  inconsistent because the accepted thrust can then violate `Qmax`.

## Bang-Bang

| fuel (kg) | ours (m) | paper (m) | delta |
| --- | --- | --- | --- |
| 500 | 3371.9 / 3201.5 / 3024.0 / 2735.5 / 2265.3 / 2074.8 | 3372 / 3201 / 3024 / 2735 / 2265 / 2076 | ≤ 1.2 m |
| 2000 | 15004.9 / 14626.7 / 13873.6 / 12361.0 / 9461.2 / 8032.8 | 15005 / 14627 / 13874 / 12362 / 9461 / 8033 | ≤ 1.0 m |
| 4000 | **41967.8** / 43366.8 / 41337.2 / 35476.6 / 24136.8 / 19361.9 | **42083** / 43367 / 41337 / 35477 / 24137 / 19362 | 0.27% on `H_MT`, ≤ 0.5 m elsewhere |

`H_MT` is the single scenario where our result is outside measurement noise.
We tried the plausible variants (surcharge on/off, search depth 8--30, final
step at `lo`/`hi`, mass integrated inside vs outside RK4, clamping on the
pre-step state) and none reproduces 42083 exactly; the closest is 41970.
The gap is reported rather than tuned away.

## Unconstrained Optimal

The paper gives the singular arc (equation 8) but not the switch time
`t_switch`. The plan's suggestion ("first time the equation-8 thrust drops
below `Tmax`") switches at `t = 0` because equation 8 evaluates to `m g` at
rest, so it is not usable. We instead choose `t_switch` by a 1-D search on the
*unconstrained* model (`Qmax -> infinity`), maximising apogee, then replay that
policy in the constrained evaluator. This reproduces:

| fuel (kg) | ours | paper |
| --- | --- | --- |
| 500 | 3504.5 (all Qmax) | 3505 |
| 2000 | 14443.3 / 15716.7 / 15804.8 (x4) | 14443 / 15717 / 15805 |
| 4000 | 20066.7 / 30972.7 / 44529.6 / **50165.3** (x3) | 27966 / 39307 / 46878 / **50165** |

The three heavy-fuel / tight-Qmax rows are not reproducible from the published
information: the paper's values lie between our bang-arc-bang replay and a
fully Q-saturated policy, which suggests a different (undisclosed) switching
rule or a different treatment of the constraint on the singular arc. We keep
the honest, oracle-free definition and report the gap.

## MC-Local-Greedy

Our optimiser uses 30 segments over `4 * t_full`, 500 candidates, and
coordinate descent on the top eight (two coarse passes at 0.2, one fine pass at
0.05), evaluated in parallel with Numba. Wall time is 1--2 s per scenario
versus 20--233 s in the paper, because every candidate is rolled out inside one
parallel kernel.

| scenario | ours | paper | ratio |
| --- | --- | --- | --- |
| `L_MT` | 3487 | 3500 | 0.996 |
| `N_ST` | 15615 | 15754 | 0.991 |
| `H_T` | 49807 | 49707 | 1.002 |
| `H_MT` | 43127 | 44023 | 0.980 |

## CasADi optimum

`src/baselines.py::solve_casadi` solves a free-final-time multiple-shooting
transcription (RK4 dynamics, `q <= Qmax` at every node, `v(T) = 0`) warm-started
from the Bang-Bang rollout. It is a true upper bound on the continuous problem,
so it is slightly above the best rollout (`H_MT`: 44823 m against a best
rollout of 43127 m). The open-loop replayed control is stored separately in
`replay_apogee` because Q-limited arcs are open-loop unstable and the replay is
not a fair "method" rollout.

## Environment notes

* Episodes stop at apogee or after `2000` steps (200 s), matching the paper.
  Heavy-fuel episodes sometimes hit the 200 s cap before true apogee, so their
  reported altitude is the value at truncation, exactly as in Table 7.
* The training distribution is strictly wider than the evaluation grid:
  `m_fuel_0 in [300, 5000]`, `Tmax in [1e5, 3e5]`, `cd in [1, 2]`,
  `Qmax in [3500, 60000]`, with exact evaluation pairs excluded and
  thrust-to-weight `>= 1.1`.
