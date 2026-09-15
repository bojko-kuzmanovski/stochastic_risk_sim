# Stochastic Risk Simulator

A discrete-event simulation engine combined with a probabilistic temporal-logic verifier, used to
model and formally analyze risk scenarios (startup runway, credit default, fraud, supply-chain
disruption, and more) as populations of agents moving through probabilistic state machines. Built as
part of a UNAM Matemáticas thesis on stochastic risk simulation.

## What it does

Agents move through **automata** (state machines) whose transitions are driven by configurable
probability distributions and timed/triggered events. After each run, a **PATL verifier** analytically
checks probabilistic claims about the system — e.g. "probability of reaching state X is ≥ 40%" — from
snapshots taken during the run, rather than by re-sampling. Every scenario (agents, environments,
automata, events, distributions, predicates) is plain JSON — none of it is hardcoded.

## Example scenario definition

A minimal scenario needs a distribution, an agent that uses it, an automaton that consumes it in a
transition, and a PATL predicate to verify (trimmed from the full schema — see `schemas/`):

```json
// distributions.json
{
  "distribution_name": "default_probability",
  "family": "beta",
  "params": { "alpha": 2.0, "beta": 8.0 },
  "output_type": "float",
  "truncation": { "min": 0.0, "max": 1.0 }
}
```

```json
// agents.json
{
  "quantity": 1,
  "agent_type": "Borrower",
  "params": {
    "default_risk": { "type": "probabilistic", "distribution": "default_probability" },
    "balance": { "type": "deterministic", "value": 10000 }
  },
  "automata": ["credit_lifecycle"]
}
```

```json
// automata.json
{
  "automaton_name": "credit_lifecycle",
  "states": { "initial": "CURRENT", "final": ["DEFAULTED", "PAID_OFF"] },
  "params": {},
  "transitions": [
    {
      "from": "CURRENT",
      "threshold_value": {
        "$risk_draw": { "type": "probabilistic", "distribution": "default_probability" }
      },
      "thresholds": [
        { "threshold_case": [{ "variable": "$risk_draw", "operator": ">=", "value": 0.5 }], "to": "DEFAULTED" },
        { "threshold_case": [{ "variable": "$risk_draw", "operator": "<",  "value": 0.5 }], "to": "PAID_OFF" }
      ]
    }
  ]
}
```

```json
// patl.json
{
  "observations": [
    {
      "automaton_name": "credit_lifecycle",
      "trigger_state": "CURRENT",
      "predicates": [
        {
          "predicate_id": "PR_DEFAULT_RISK_ABOVE_20PCT",
          "type": "reachability",
          "coalition_quantifier": "exists",
          "coalition": [
            { "agent_type": "Borrower", "max_agents": 1,
              "automata": [{ "automaton_name": "credit_lifecycle", "target_states": ["DEFAULTED"] }] }
          ],
          "probability_bound": 0.20,
          "probability_operator": ">="
        }
      ]
    }
  ]
}
```

This says: *"whenever a Borrower is in `CURRENT`, verify that the probability of it eventually
reaching `DEFAULTED` is at least 20%."* Real scenarios also add `environments.json` (relationships
between agents) and `events.json` (timers/triggers).

## Why analytical, not Monte Carlo?

The verification step does not estimate probabilities by re-running the simulation and counting outcomes.
From each snapshot it builds a bounded game between the coalition and the adversaries of the predicate and
evaluates the PATL_b operator

    <<C>>_k^{op d} psi  iff  exists sigma_C (observation-based, memory at most k)  for all sigma_A :  P(psi) op d

* Path formulas are bounded to `delta` rounds: `until` (phi1 U phi2), `release` (phi1 R phi2) and `next`
  (X phi), over state formulas built from `true`, `target` (membership of the coalition in its target final
  states), `prop` (an automaton and final states of a group), `not`, `and`, `or`. `reachability` and
  `invariance` are shorthands for `true U target` and `false R not target`.
* A round activates every participant once: each runs a complete atomic automaton session (the triggering
  agent finishes its session in progress; the others start the automaton their strategy selects). The
  activation order inside a round is uniformly random and the value averages over it, so results do not
  depend on agent names.
* Each probabilistic threshold becomes one branch per case under the first matching case rule, weighted by
  its exact mass (CDF, Poisson support, categorical label). Nothing is sampled. When a sampled continuous value
  is used later by the automaton, each case region is split into `--quantiles` equal-mass cells, each passing
  its conditional expectation on (converges to the exact kernel as cells grow). Continuous distributions with
  integer output are treated as discrete.
* Coalition strategies are observation-based with memory at most `k`: deterministic tables `act` and `Delta`
  over (mode, observation), plus memoryless randomized strategies (optimized on a refined grid when a single
  coalition member has a choice). Adversaries are unrestricted; their best response is computed by backward
  induction, choosing at the start of each round without seeing the realization of the coalition's mixture.
* The direction follows the bound: for `>=`/`>` the coalition maximizes and the adversaries minimize; for
  `<=`/`<` the reverse. Non-participants stay frozen and events emitted during verification are not
  propagated. Verdicts are decided on the exact value with a 1e-9 tolerance; the CSV shows four decimals
  and the strategy class that attained the value (`deterministic` or `mixed`).

A predicate whose automaton is not assigned to the agent type, whose cases do not cover the support, or
whose game exceeds the size limits, is reported as `ERROR` with its reason.

## Project structure

| Path | Purpose |
|---|---|
| `configs/<domain>/<scenario>/` | Scenario definitions (agents, environments, automata, events, distributions, PATL predicates) as JSON, per business domain (banking-finance, cybersecurity, e-commerce, fintech, startups, supply-chain) |
| `schemas/` | JSON Schemas used to validate every config file at load time |
| `core/` | The simulation engine: agents, automata, environments, events, distributions, and the shared expression language configs use |
| `des/` | Discrete-event orchestration for a single simulation run |
| `patl/` | Snapshot capture and the probabilistic temporal-logic verifier |
| `metrics/` | Collects run statistics and writes them to CSV |
| `analytics/` | Standalone script for descriptive statistics over completed runs |
| `data/` | Output CSVs and snapshots from simulation runs (gitignored) |
| `main.py` | Entry point: runs one or many simulations end-to-end |

## Architecture & data flow

```
configs/*.json  ──validate──▶  schemas/*.schema.json
      │
      ▼
┌───────────────────────── one simulation run (one process) ───────────┐
│ core/: Distributions (seeded), Environments, Automata, Agents         │
│      │                                                                │
│      ▼                                                                │
│  DES phase (des/)                     patl/snapshot_manager.py        │
│  calendar Q ordered by time; clock T  ─▶ capture() on trigger states  │
│  jumps to the next event, up to T_max ──▶ data/.snapshots/run_<id>/   │
│  events → agent queues Q_i → up to K atomic automaton sessions        │
│      │                                                                │
│      ▼                                                                │
│  PATL phase: read snapshots (batched) → PATLVerifier                  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
metrics/metrics_writer.py (main process) ──▶ <name>_summary.csv, _des.csv, _patl.csv
      │
      ▼
analytics/analyze_scenario.py  (optional, standalone descriptive stats)
```

DES (stochastic simulation) and PATL (analytical verification) are two separate phases within each run.
The DES follows the event calendar model: static events are first scheduled at `t0 ~ rho` for every agent of
their type and rescheduled at `T + dt, dt ~ rho` each time they are dispatched; dynamic events emitted by an
automaton enter the calendar at the current instant. When the next event lies beyond `T_max`, dispatching
stops and the remaining queues are drained. Run `i` is seeded with `seed * 1000000 + i`, so a run can be
reproduced exactly. `--threads` runs independent simulations in parallel processes.

## Requirements

- Python 3.10 or later (developed and tested on 3.14)
- Key dependencies: `numpy`, `scipy` (distribution sampling and CDFs), `pandas` (analytics), `jsonschema` (config validation), `tqdm` (progress bar) — see `requirements.txt` for the full list

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run a simulation (results are written to `data/`):

```bash
python3 main.py --config-dir configs/startups/runway-risk --output runway_risk_base \
    --runs 100 --time 60 --threads 10 --distributions distributions_1_base.json
```

- `--runs`: number of independent simulations (1–1000)
- `--time`: simulated time horizon `T_max` per run, in model time units (10–500)
- `--threads`: runs executed in parallel processes (1–10)
- `--seed`: base seed (default 1)
- `--config-dir` (required): scenario directory with the six JSON files. The engine contains no domain
  logic; any scenario that validates against `schemas/` runs unchanged. `tools/build_runway_config.py` is an
  optional authoring script that produced the runway JSON files; nothing in the engine imports it.
- `--distributions`: distributions file inside the scenario directory (default `distributions.json`)
- `--queue-batch`: `K`, events an agent takes from its queue per instant (default 5)
- `--memory`: memory bounds `k` of coalition strategies, e.g. `1` or `1,2` (each 1–4, default 1)
- `--event-latency`: delay of dynamic events emitted by automata (default 0). Events published with
  `write_ch_event` reach the other channel participants that implement the signal after the channel's
  `latency` metadata.
- `--trace`: see *Tracing a run*

Analyze results for a completed scenario:

```bash
python3 analytics/analyze_scenario.py --scenario <name>
```

### Tracing a run

`--trace` writes one JSONL file per run to `data/traces/<name>_run<i>.jsonl`. Components can be combined:
`des` (calendar instants, dispatched events), `agents` (queueing, atomic sessions), `automata` (every
transition with its threshold value and chosen case, every call and emitted event), `snapshots`,
`patl` (participants, strategy sequences, value per sequence, result) and `patl_rounds` (every game node:
adversary choice, round outcomes with their probability). Use `all` for everything, ideally with one short run.

```bash
python3 main.py --output trace_demo --runs 1 --time 12 --trace all --distributions distributions_1_base.json
python3 analytics/trace_report.py data/traces/trace_demo_run1.jsonl
python3 analytics/trace_report.py data/traces/trace_demo_run1.jsonl --automaton runway_lifecycle
python3 analytics/trace_report.py data/traces/trace_demo_run1.jsonl --predicate PR_STOCHASTIC_DEMAND_STABILITY_VIA_PROFILE --limit 2
```

### Tests

```bash
python3 -m pytest -q tests
```

`tests/test_patl_semantics.py` checks the verifier against games whose value can be computed by hand
(the 5/7 retention example, operator direction under uniform activation order, memory k=2 against k=1,
randomized memoryless strategies, imperfect information of the coalition, an unrestricted adversary,
quantile cells, bounded until/release/next, exact Poisson support, integer and categorical outputs,
reported errors) plus property-based checks (closed form 1-(1-p)^delta, duality, monotonicity in delta,
independence from agent names). `tests/test_des_engine.py` checks calendar ordering, reproducibility by
seed, static activation times, channel latency, an end-to-end PATL run and that the engine contains no
names from any configuration. `tests/test_des_audit.py` covers agent ids and removal, sampling and
periodicity validation, explicit errors and the empty value, static events and emission, snapshots,
channels, directed relations, the per-instant bound K, case coverage at load time and categorical parameters.

## Output & interpretation

Each run produces three CSV files under `data/`, all sharing the `<name>` prefix:

| File | Contents |
|---|---|
| `<name>_summary.csv` | One row per metric: seed, simulated time reached, declared config counts (agents, automata, predicates, etc.) plus run performance (`elapsed_des_sec`, `elapsed_patl_sec`). Columns: `run_id, category, key, subkey, value` |
| `<name>_des.csv` | Runtime telemetry from the simulation itself: distribution samples drawn, agent/environment actions taken, automaton executions, events fired, snapshots captured. Columns: `run_id, runtime_section, entity_key, metric_subkey, execution_value` |
| `<name>_patl.csv` | One row per verified predicate per snapshot. Columns: `run_id, automaton_name, trigger_state, agent_id, predicate_id, value, bound, operator, memory_k, result, reason` |

To read a PATL result: `value` is the value of the property (the optimal probability of the path formula
for the coalition against its adversaries), not a statistical p-value. `result` is `SATISFIED` or `VIOLATED`
depending on whether `value` meets `bound`/`operator` from `patl.json`, or `ERROR` when the predicate could
not be verified, with the cause in `reason`.

## Use cases

| Domain | Example questions |
|---|---|
| Banking & Finance | Probability a loan reaches default; probability a money-muling ring evades detection |
| Cybersecurity | Probability of privilege escalation succeeding before it's flagged; probability an anomaly generates a false alert |
| E-commerce | Probability of a stockout during a demand spike; probability of losing a customer to an intermittent stockout |
| Fintech | Probability fraud goes undetected; probability risk-drift breaches a compliance threshold |
| Startups | Probability of exhausting runway before closing a funding round; probability an investor walks away after a market shock |
| Supply Chain | Probability a supplier delay cascades into a stockout; probability of recovering from a demand shift within N periods |
