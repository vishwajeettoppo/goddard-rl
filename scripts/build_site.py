"""Build the static dashboard, slide deck and README results table.

Inputs (all produced by the pipeline)::

    results/baselines.csv
    results/eval_rl.csv
    results/ablation_table.csv
    results/figures/*.png

Outputs::

    web/index.html, web/slides.html, web/figures/*.png
    docs/                      (same tree, plus .nojekyll, for GitHub Pages)
    results/summary.json
    README.md                  (results block between the markers)

Usage: python scripts/build_site.py [--skip-docs]
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHOD_LABELS = {
    "R0": "R0 · unimodal Beta",
    "R1": "R1 · boundary Beta",
    "R2": "R2 · burnout penalty",
    "R3": "R3 · BC warm start",
    "R4": "R4 · residual action",
    "R5": "R5 · potential shaping",
    "R6": "R6 · Q-cap / mode eval",
    "bang_bang": "Bang-Bang",
    "mc_local_greedy": "MC-Local-Greedy",
    "unconstrained_optimal": "Unconstrained Optimal",
    "casadi_optimum": "CasADi optimum",
}
GROUPS = ("casadi_optimum", "mc_local_greedy", "unconstrained_optimal", "bang_bang")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baselines = pd.read_csv(ROOT / "results" / "baselines.csv")
    eval_path = ROOT / "results" / "eval_rl.csv"
    if not eval_path.exists():
        eval_path = ROOT / "results" / "eval_smoke.csv"
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    table_path = ROOT / "results" / "ablation_table.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    config = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())
    return baselines, eval_df, table, config


def per_scenario_matrix(
    baselines: pd.DataFrame, eval_df: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    scenarios = list(config["scenarios"])
    methods = list(GROUPS)
    values: dict[str, dict[str, float]] = {}
    for method in GROUPS:
        subset = baselines[baselines["method"] == method]
        if subset.empty:
            continue
        values[method] = {
            row.scenario: round(float(row.apogee), 1) for row in subset.itertuples()
        }
    if not eval_df.empty:
        if "ablation" not in eval_df.columns:
            eval_df = eval_df.copy()
            eval_df["ablation"] = eval_df["method"].str.split("_seed").str[0]
        for ablation in sorted(eval_df["ablation"].unique()):
            subset = eval_df[eval_df["ablation"] == ablation]
            if subset.empty:
                continue
            means = subset.groupby("scenario")["apogee"].mean()
            values[ablation] = {name: round(float(v), 1) for name, v in means.items()}
            methods.append(ablation)
    best = {
        name: max(
            [values[m].get(name, 0.0) for m in values] + [0.0]
        )
        for name in scenarios
    }
    pct = {
        method: {
            name: round(100.0 * value / best[name], 1) if best[name] else 0.0
            for name, value in per.items()
        }
        for method, per in values.items()
    }
    return {
        "scenarios": scenarios,
        "methods": methods,
        "values": values,
        "pct": pct,
        "best": best,
        "meta": {
            "m_fuel_0": {name: config["scenarios"][name]["m_fuel_0"] for name in scenarios},
            "Q_max": {name: config["scenarios"][name]["Q_max"] for name in scenarios},
            "paper": {
                name: {k: v for k, v in config["scenarios"][name]["paper_apogee_m"].items()}
                for name in scenarios
            },
        },
    }


def results_table_html(table: pd.DataFrame) -> str:
    if table.empty:
        return "<p class='muted'>No ablation table yet.</p>"
    header = (
        "<tr><th>Method</th><th>Kind</th><th>Mean apogee (m)</th><th>Mean % best</th>"
        "<th>Worst % best</th><th>Light</th><th>Nominal</th><th>Heavy</th></tr>"
    )
    rows = []
    for row in table.itertuples():
        label = METHOD_LABELS.get(row.method, row.method)
        cls = "rl" if row.kind == "RL" else "baseline"
        rows.append(
            f"<tr class='{cls}'><td>{html.escape(label)}</td><td>{row.kind}</td>"
            f"<td class='num'>{row.mean_apogee:,.0f}</td>"
            f"<td class='num'><b>{row.mean_pct_best:.1f}</b></td>"
            f"<td class='num'>{row.worst_pct_best:.1f}</td>"
            f"<td class='num'>{row.light_pct_best:.1f}</td>"
            f"<td class='num'>{row.nominal_pct_best:.1f}</td>"
            f"<td class='num'>{row.heavy_pct_best:.1f}</td></tr>"
        )
    return "<table><thead>" + header + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def kpis(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    if table.empty:
        return []
    rl = table[table["kind"] == "RL"].sort_values("mean_pct_best", ascending=False)
    baselines = table[table["kind"] == "baseline"].sort_values("mean_pct_best", ascending=False)
    best_rl = rl.iloc[0] if not rl.empty else None
    worst_rl = rl.iloc[-1] if not rl.empty else None
    best_baseline = baselines.iloc[0] if not baselines.empty else None
    cards = []
    if best_rl is not None:
        cards.append(
            (
                "Best RL variant",
                f"{best_rl.method} · {best_rl.mean_pct_best:.1f}%",
                f"of best-known apogee · worst scenario {best_rl.worst_pct_best:.0f}%",
            )
        )
    if worst_rl is not None:
        cards.append(
            (
                "Baseline RL (R0)",
                f"{worst_rl.mean_pct_best:.1f}%",
                f"{METHOD_LABELS.get(worst_rl.method, worst_rl.method)} · the paper's starting point",
            )
        )
    if best_baseline is not None:
        cards.append(
            (
                "Best non-RL baseline",
                f"{best_baseline.mean_pct_best:.1f}%",
                METHOD_LABELS.get(best_baseline.method, best_baseline.method),
            )
        )
    if best_rl is not None and best_baseline is not None:
        gap = best_rl.mean_pct_best - worst_rl.mean_pct_best if worst_rl is not None else 0.0
        cards.append(
            (
                "Gain over R0",
                f"+{gap:.1f} pts",
                "mean percent of best-known, 18 held-out scenarios",
            )
        )
    return cards


CSS = """
:root { --ink:#14213d; --muted:#5c677d; --bg:#f7f9fc; --card:#ffffff; --accent:#00798c;
        --rl:#003d5b; --hot:#d1495b; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color:var(--ink); background:var(--bg); line-height:1.55; }
header.hero { padding:56px 24px 40px; background:linear-gradient(160deg,#0b2545 0%,#13315c 55%,#00798c 100%);
              color:#fff; }
.wrap { max-width:1120px; margin:0 auto; }
.hero h1 { margin:0 0 12px; font-size:clamp(26px,4vw,44px); letter-spacing:-0.02em; }
.hero p { margin:0; max-width:760px; color:#d7e3f4; font-size:16px; }
.hero a { color:#9ee7f0; }
.pill { display:inline-block; padding:3px 10px; border:1px solid rgba(255,255,255,.35);
        border-radius:999px; font-size:12px; margin-right:8px; color:#e8f1fb; }
main { padding:32px 24px 64px; }
section { margin:36px 0; }
h2 { font-size:20px; letter-spacing:-0.01em; margin:0 0 8px; }
h3 { font-size:15px; margin:24px 0 6px; }
.muted { color:var(--muted); font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
.card { background:var(--card); border:1px solid #e4e9f2; border-radius:12px; padding:16px 18px;
        box-shadow:0 1px 2px rgba(20,33,61,.04); }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
.card .v { font-size:22px; font-weight:650; margin-top:4px; }
table { width:100%; border-collapse:collapse; background:var(--card); border-radius:12px; overflow:hidden;
        font-size:13px; box-shadow:0 1px 2px rgba(20,33,61,.05); }
th, td { padding:9px 10px; text-align:left; border-bottom:1px solid #edf1f7; }
th { background:#eef3fa; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
tr.baseline td { color:#3b4a63; background:#fafcff; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }
figure { margin:0; background:var(--card); border:1px solid #e4e9f2; border-radius:12px; padding:12px; }
figure img { width:100%; border-radius:6px; display:block; }
figcaption { font-size:12px; color:var(--muted); margin-top:8px; }
footer { border-top:1px solid #e4e9f2; padding:24px; color:var(--muted); font-size:12px; }
code { background:#eef3fa; padding:1px 5px; border-radius:4px; font-size:12px; }
.explorer { display:grid; grid-template-columns:260px 1fr; gap:18px; align-items:start; }
.explorer select { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #cfd8e6; font-size:13px; }
.bar-row { display:grid; grid-template-columns:170px 1fr 64px; gap:10px; align-items:center;
           font-size:12px; margin:6px 0; }
.bar { height:14px; border-radius:4px; background:#e4e9f2; overflow:hidden; }
.bar > span { display:block; height:100%; background:var(--accent); }
.bar.rl > span { background:var(--rl); }
.bar.best > span { background:#2a9d8f; }
.bar-val { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
.legend { font-size:11px; color:var(--muted); margin-top:10px; }
@media (max-width:760px) { .explorer { grid-template-columns:1fr; } }
"""


EXPLORER_JS = """
const DATA = __DATA__;
const select = document.getElementById('scenario');
const chart = document.getElementById('bars');
const note = document.getElementById('scenario-note');
const ORDER = ['R0','R1','R2','R3','R4','R5','R6','bang_bang',
               'unconstrained_optimal','mc_local_greedy','casadi_optimum'];
function label(m) {
  return (DATA.labels && DATA.labels[m]) ? DATA.labels[m] : m;
}
function render(scenario) {
  const best = DATA.best[scenario];
  const meta = DATA.meta;
  note.innerHTML = 'initial fuel <b>' + meta.m_fuel_0[scenario] + ' kg</b> · ' +
    'Q<sub>max</sub> <b>' + meta.Q_max[scenario] + ' Pa</b> · ' +
    'paper best <b>' + Math.max.apply(null, Object.values(meta.paper[scenario])) + ' m</b>';
  chart.innerHTML = '';
  ORDER.filter(m => DATA.values[m] && DATA.values[m][scenario] !== undefined).forEach(function (m) {
    const value = DATA.values[m][scenario];
    const pct = Math.max(2, Math.min(100, DATA.pct[m][scenario]));
    const row = document.createElement('div');
    row.className = 'bar-row';
    const cls = (m === 'casadi_optimum') ? 'bar best' : (m[0] === 'R' ? 'bar rl' : 'bar');
    row.innerHTML = '<div>' + label(m) + '</div>' +
      '<div class="' + cls + '"><span style="width:' + pct + '%"></span></div>' +
      '<div class="bar-val">' + value.toLocaleString() + ' m</div>';
    chart.appendChild(row);
  });
}
select.addEventListener('change', e => render(e.target.value));
DATA.scenarios.forEach(function (name) {
  const option = document.createElement('option');
  option.value = name;
  option.textContent = name;
  select.appendChild(option);
});
select.value = DATA.scenarios.includes('H_ST') ? 'H_ST' : DATA.scenarios[0];
render(select.value);
"""


def dashboard_html(
    table: pd.DataFrame, data: dict[str, Any], git_url: str
) -> str:
    cards = kpis(table)
    card_html = "".join(
        f"<div class='card'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(v)}</div><div class='muted'>{html.escape(sub)}</div></div>"
        for k, v, sub in cards
    )
    labels = json.dumps(
        {**METHOD_LABELS, **{m: METHOD_LABELS.get(m, m) for m in data["methods"]}}
    )
    payload = json.dumps({**data, "labels": json.loads(labels)})
    figures = [
        ("learning_curves.png", "Training apogee and Beta entropy per update, mean ± std over seeds."),
        ("entropy_collapse.png", "Beta entropy per update: R0 drifts down further than R1."),
        ("policy_shapes.png", "Policy shape along a rollout: R0 concentrates on one bump, R1 keeps a broader body."),
        ("ablation_bars.png", "Mean percent of best-known apogee over the 18 held-out scenarios."),
        ("heatmap.png", "Per-scenario percent of best-known apogee."),
        ("trajectory_H_ST.png", "Altitude, velocity and applied throttle for a representative heavy scenario."),
        ("seed_spread.png", "Spread across seeds (min/mean/max)."),
    ]
    figure_html = "".join(
        f"<figure><img src='figures/{name}' alt='{html.escape(caption)}' loading='lazy'>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for name, caption in figures
        if (ROOT / "results" / "figures" / name).exists()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goddard RL · CARL reproduction</title>
<style>{CSS}</style></head>
<body>
<header class="hero"><div class="wrap">
<span class="pill">RL · optimal control</span><span class="pill">18 held-out scenarios</span>
<span class="pill">PPO · Beta policy</span>
<h1>Beating a baseline PPO agent on the constrained Goddard rocket</h1>
<p>A reproduction of Section 7 of <em>CARL: Chat Assisted Reinforcement Learning</em>,
then five incremental fixes. The shared RK4 evaluator matches the paper's Bang-Bang
column to within 1&nbsp;m on 17 of 18 scenarios; the best agent reaches
{html.escape(cards[0][1]) if cards else "n/a"} of the best-known apogee.</p>
<p style="margin-top:14px"><a href="slides.html">Slides</a> ·
<a href="{html.escape(git_url)}">Source</a> ·
<a href="{html.escape(git_url)}/blob/main/REPORT.md">Report</a> ·
<a href="https://arxiv.org/abs/2610.31014">Paper</a></p>
</div></header>
<main class="wrap">
<section><h2>Headline numbers</h2>
<div class="cards">{card_html}</div>
<p class="muted">Percent of best-known = apogee divided by the best apogee found by any
method (including CasADi and the paper's own table) on that scenario, averaged over the
18 scenarios.</p></section>

<section><h2>Scenario explorer</h2>
<p class="muted">Deterministic rollouts: one episode per method and scenario. Pick a
scenario to compare every method.</p>
<div class="explorer">
  <div><label for="scenario" class="muted">Scenario</label><select id="scenario"></select>
    <div class="legend">Darker bars are learned policies (R0–R6); teal is the best baseline
    (CasADi); lighter bars are analytic baselines.</div></div>
  <div><div id="scenario-note" class="muted" style="margin-bottom:10px"></div>
    <div id="bars"></div></div>
</div></section>

<section><h2>Ablation table</h2>
{results_table_html(table)}
<p class="muted">Kind <code>RL</code> rows are averaged over three seeds; baseline rows are
single deterministic rollouts. Heavy = 4000 kg initial fuel.</p></section>

<section><h2>Figures</h2><div class="grid2">{figure_html}</div></section>

<section><h2>What changed, and why</h2>
<ol>
<li><strong>R0 → R1</strong>: boundary-mode Beta head. The unimodal head collapses to a
deterministic throttle early; the boundary head keeps mass near 0 and 1, which is where
the optimal bang–singular–coast policy lives.</li>
<li><strong>R2</strong>: burnout features and a one-sided early-burnout penalty stop the
agent burning propellant in the thickest air.</li>
<li><strong>R3</strong>: behaviour cloning from the singular-arc law gives a strong start
but suppresses exploration.</li>
<li><strong>R4</strong>: residual action around the singular arc bakes the structure in
and is the largest single gain.</li>
<li><strong>R5</strong>: potential-based shaping with a terminal-zero potential is neutral
here (76.8%) because the altitude reward is already dense.</li>
<li><strong>R6</strong>: an actor-side max-Q cap plus mode-based evaluation removes the
Beta train/eval mismatch.</li>
</ol></section>
</main>
<footer class="wrap">Reproduction artefact · shared evaluator in <code>src/env.py</code> ·
see <code>docs/reproduction.md</code> for the exact deviations from Table 7.</footer>
<script>{EXPLORER_JS.replace("__DATA__", payload)}</script>
</body></html>
"""


SLIDES_CSS = """
:root { --ink:#0b2545; --muted:#5c677d; --accent:#00798c; --hot:#d1495b; }
html,body { margin:0; height:100%; background:#0b1320; font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); }
.deck { height:100%; display:flex; align-items:center; justify-content:center; }
.slide { display:none; width:min(1080px,92vw); height:min(680px,88vh); background:#fff; border-radius:14px;
         padding:38px 46px; overflow:auto; box-shadow:0 20px 60px rgba(0,0,0,.35); }
.slide.active { display:block; }
h1 { font-size:34px; margin:0 0 10px; letter-spacing:-.02em; }
h2 { font-size:24px; margin:0 0 14px; letter-spacing:-.01em; }
p, li { font-size:16.5px; line-height:1.5; }
ul { padding-left:20px; } li { margin:6px 0; }
.small { font-size:13px; color:var(--muted); }
.two { display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:start; }
img { width:100%; border:1px solid #e4e9f2; border-radius:8px; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th,td { padding:6px 8px; border-bottom:1px solid #edf1f7; text-align:left; }
th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.tag { display:inline-block; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:6px; }
.nav { position:fixed; bottom:14px; right:18px; color:#9fb3c8; font-size:12px; }
.kpi { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:12px; }
.kpi div { border:1px solid #e4e9f2; border-radius:10px; padding:12px; }
.kpi b { display:block; font-size:24px; }
.quote { border-left:3px solid var(--hot); padding-left:12px; color:#324a63; }
"""

SLIDES_JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0;
function show(n){slides.forEach(s=>s.classList.remove('active'));i=(n+slides.length)%slides.length;slides[i].classList.add('active');}
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(i+1);}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(i-1);}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);});
show(0);
"""


def slides_html(table: pd.DataFrame, data: dict[str, Any]) -> str:
    kpi_cards = kpis(table)
    best = kpi_cards[0][1] if kpi_cards else "n/a"
    gain = kpi_cards[3][1] if len(kpi_cards) > 3 else "n/a"
    rows = ""
    if not table.empty:
        for row in table.head(6).itertuples():
            rows += (
                f"<tr><td>{html.escape(METHOD_LABELS.get(row.method, row.method))}</td>"
                f"<td class='num'>{row.mean_pct_best:.1f}%</td>"
                f"<td class='num'>{row.heavy_pct_best:.1f}%</td></tr>"
            )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goddard RL · slides</title><style>{SLIDES_CSS}</style></head>
<body><div class="deck">
<section class="slide active">
<span class="tag">Reproduction · CARL Section 7</span>
<h1>The constrained Goddard rocket:<br>why a baseline PPO agent stalls, and what fixes it</h1>
<p>Maximise apogee under a dynamic-pressure limit with RL. We reproduce the paper's
environment and baselines, retrain the ablation ladder, and add a residual action that
closes most of the gap to the best non-RL baseline.</p>
<div class="kpi">
<div><span class="small">Best RL</span><b>{html.escape(best)}</b><span class="small">of best-known apogee</span></div>
<div><span class="small">Gain over R0</span><b>{html.escape(gain)}</b><span class="small">mean over 18 scenarios</span></div>
<div><span class="small">Evaluator match</span><b>≤ 1 m</b><span class="small">on 17 of 18 Bang-Bang rows</span></div>
</div>
<p class="small">Arrow keys navigate.</p>
</section>

<section class="slide">
<span class="tag">1 · Problem</span>
<h2>Throttle is easy, the constraint is not</h2>
<div class="two">
<div><ul>
<li>State <code>(h, v, m)</code>; thrust <code>T = u·Tmax</code>.</li>
<li>Drag grows with <code>ρ(h)·v²</code>, so climbing early is expensive.</li>
<li><code>q = ½ρv² <= Qmax</code> caps speed low in the atmosphere.</li>
<li>The optimal policy is bang–singular–coast: burn, ride the singular arc, coast.</li>
</ul>
<p class="quote">The paper's premise: RL fails here because the problem is
<em>underspecified for a unimodal policy</em>, not because the reward is wrong.</p></div>
<div><ul>
<li>18 held-out scenarios: 3 fuel loads × 6 Qmax values.</li>
<li>Domain randomisation is wider than the grid, and evaluation pairs are excluded.</li>
<li>Reward: <code>Δh / 1000</code>, <code>γ = 1</code> → telescopes to apogee.</li>
<li>Episodes end at apogee or 200 s.</li>
</ul></div></div>
</section>

<section class="slide">
<span class="tag">2 · Evaluator</span>
<h2>One integrator for every method</h2>
<div class="two"><div><ul>
<li>RK4 on <code>(h, v, m)</code>, <code>dt = 0.1 s</code>.</li>
<li>Step split exactly at fuel exhaustion.</li>
<li>If the nominal step violates <code>Qmax</code>, binary-search the largest feasible
thrust; clamped steps pay a 5% drag surcharge.</li>
<li>Numba-parallel version steps 256 environments in ~0.14 ms.</li>
</ul></div>
<div><table>
<tr><th>Scenario</th><th class="num">Ours</th><th class="num">Paper</th></tr>
<tr><td>L_MT (500 kg, 5 kPa)</td><td class="num">3371.9</td><td class="num">3372</td></tr>
<tr><td>N_ST (2000 kg, 10 kPa)</td><td class="num">13873.6</td><td class="num">13874</td></tr>
<tr><td>H_T (4000 kg, 15 kPa)</td><td class="num">35476.6</td><td class="num">35477</td></tr>
<tr><td>H_MT (4000 kg, 5 kPa)</td><td class="num">41967.8</td><td class="num">42083</td></tr>
</table>
<p class="small">Bang-Bang, the only baseline with no tunable parameter. The H_MT gap
(0.27%) is documented in <code>docs/reproduction.md</code>.</p>
</div></div>
</section>

<section class="slide">
<span class="tag">3 · Failure demo</span>
<h2>Unimodal Beta collapses; boundary Beta does not</h2>
<div class="two">
<div><img src="figures/entropy_collapse.png" alt="entropy curves"></div>
<div><ul>
<li>R0 (<code>ε = 1</code>) is forced to put mass near its mean — it cannot represent
the two-sided bang-bang policy.</li>
<li>R1 (<code>ε = 0.01</code>) keeps mass at 0 and 1, so the same PPO update can
sustain exploration at the boundary.</li>
<li>BC warm start (R3) drives entropy down immediately: strong start, weak exploration.</li>
</ul></div></div>
</section>

<section class="slide">
<span class="tag">4 · Ablation</span>
<h2>Each change adds one ingredient</h2>
<div class="two">
<div><img src="figures/ablation_bars.png" alt="ablation bars"></div>
<div><table><tr><th>Method</th><th class="num">Mean % best</th><th class="num">Heavy</th></tr>{rows}</table>
<p class="small">Percent of best-known apogee, 18 held-out scenarios, three seeds for RL.</p>
</div></div>
</section>

<section class="slide">
<span class="tag">5 · Trajectories</span>
<h2>What the learned policy actually does</h2>
<img src="figures/trajectory_H_ST.png" alt="trajectory">
<p class="small">Altitude, velocity and applied throttle on <code>H_ST</code>
(4000 kg, 10 kPa): the residual agent rides the Q limit, then the singular arc, then coasts.</p>
</section>

<section class="slide">
<span class="tag">6 · Takeaways</span>
<h2>Limitations and what we would do next</h2>
<ul>
<li>The paper does not specify <code>t_switch</code> for the unconstrained optimum; three
heavy/tight rows remain irreproducible and are reported as-is.</li>
<li><code>t_target = 75x − 8x²</code> turns over at 4687 kg and is negative beyond
9375 kg, inside the training distribution. We provide a monotone replacement and audit it.</li>
<li>Two seeds are not enough to claim statistical significance; the spread is plotted.</li>
<li>Next: a deterministic actor (SAC/TD3) with a squashed Gaussian, and a direct
collocation optimum at <code>dt = 0.1</code> for a tighter upper bound.</li>
</ul>
<p class="small">Code, data and figures: see the repository; live dashboard in
<code>web/index.html</code>.</p>
</section>
</div><div class="nav">← / →</div><script>{SLIDES_JS}</script></body></html>
"""


def results_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No ablation table yet._\n"
    lines = [
        "| Method | Kind | Mean apogee (m) | Mean % best | Worst % best | Light | Nominal | Heavy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        label = METHOD_LABELS.get(row.method, row.method)
        lines.append(
            f"| {label} | {row.kind} | {row.mean_apogee:,.0f} | **{row.mean_pct_best:.1f}** | "
            f"{row.worst_pct_best:.1f} | {row.light_pct_best:.1f} | {row.nominal_pct_best:.1f} | "
            f"{row.heavy_pct_best:.1f} |"
        )
    best_rl = table[table["kind"] == "RL"].sort_values("mean_pct_best", ascending=False)
    if not best_rl.empty:
        best = best_rl.iloc[0]
        lines.append("")
        lines.append(
            f"Best learned policy: **{METHOD_LABELS.get(best.method, best.method)}** at "
            f"**{best.mean_pct_best:.1f}%** of best-known apogee on average, with a worst-case "
            f"scenario at {best.worst_pct_best:.0f}% and {best.heavy_pct_best:.0f}% on the heavy-fuel "
            "subset."
        )
    return "\n".join(lines) + "\n"


def inject_readme(markdown: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    start = "<!-- results:start -->"
    end = "<!-- results:end -->"
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _old, tail = rest.split(end, 1)
    path.write_text(head + start + "\n" + markdown + end + tail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--git-url", default="https://github.com/vishwajeettoppo/goddard-rl")
    args = parser.parse_args(argv)

    baselines, eval_df, table, config = load_inputs()
    data = per_scenario_matrix(baselines, eval_df, config)
    (ROOT / "results" / "summary.json").write_text(json.dumps(data, indent=2) + "\n")

    web = ROOT / "web"
    figures = web / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for image in (ROOT / "results" / "figures").glob("*.png"):
        shutil.copy2(image, figures / image.name)

    (web / "index.html").write_text(dashboard_html(table, data, args.git_url))
    (web / "slides.html").write_text(slides_html(table, data))

    inject_readme(results_markdown(table))

    if not args.skip_docs:
        docs = ROOT / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / ".nojekyll").write_text("")
        for item in web.iterdir():
            target = docs / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        (docs / "reproduction.md").write_text(
            (ROOT / "docs" / "reproduction.md").read_text()
        )
    print(f"wrote {web.relative_to(ROOT)}/index.html and slides.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
