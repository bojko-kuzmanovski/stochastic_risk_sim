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

The verification step does not estimate probabilities by re-running the simulation thousands of times
and counting outcomes. Instead, it reads each distribution's CDF directly and combines them through a
depth-bounded dynamic program over the automaton's reachable states. This gives an **exact, deterministic
probability value** for every predicate — no sampling error, no trial count to tune for convergence, and
the same `p_value` every time you verify the same snapshot.

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
┌───────────────────────── one simulation run ─────────────────────────┐
│ core/: Distributions, Environments, Automata, Agents built from config│
│      │                                                                │
│      ▼                                                                │
│  DES phase (des/)                     patl/snapshot_manager.py        │
│  agents.start() + scheduler.start() ─▶ capture() on observed states   │
│  runs for --time seconds               ──▶ data/.snapshots/run_<id>/  │
│  agents react to events → automata step (may emit new events)         │
│      │                                                                │
│      ▼                                                                │
│  metrics/metrics_writer.py ──▶ <name>_summary.csv, <name>_des.csv     │
│      │                                                                │
│      ▼                                                                │
│  PATL phase: read snapshots (batched) → PATLVerifier (thread pool)    │
│      │                                                                │
│      ▼                                                                │
│  metrics/metrics_writer.py ──▶ <name>_patl.csv                        │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
analytics/analyze_scenario.py  (optional, standalone descriptive stats)
```

DES (stochastic simulation) and PATL (analytical verification) are two separate phases within each run;
`--threads` lets multiple runs' DES/PATL phases overlap.

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
python3 main.py --output <name> --runs 100 --time 60 --threads 10
```

- `--runs`: number of independent simulations (1–1000)
- `--time`: simulated seconds per run (10–500)
- `--threads`: concurrent runs (1–10)

Analyze results for a completed scenario:

```bash
python3 analytics/analyze_scenario.py --scenario <name>
```

> **Note:** `main.py` currently runs the `configs/startups/runway-risk` scenario by default. To run a
> different domain/scenario, edit the config path in `main.py`.

## Output & interpretation

Each run produces three CSV files under `data/`, all sharing the `<name>` prefix:

| File | Contents |
|---|---|
| `<name>_summary.csv` | One row per metric: declared config counts (agents, automata, predicates, etc.) plus run performance (`elapsed_des_sec`, `elapsed_patl_sec`). Columns: `run_id, category, key, subkey, value` |
| `<name>_des.csv` | Runtime telemetry from the simulation itself — distribution samples drawn, agent/environment actions taken, automaton executions, events fired. Columns: `run_id, runtime_section, entity_key, metric_subkey, execution_value` |
| `<name>_patl.csv` | One row per verified predicate per run — the actual formal-verification results. Columns: `run_id, automaton_name, trigger_state, agent_id, predicate_id, p_value, bound, operator, result` |

To read a PATL result: `result` is `SATISFIED` or `VIOLATED` depending on whether the exact computed
`p_value` meets the `bound`/`operator` threshold from `patl.json` (e.g. `>= 0.20`) — a pass/fail flag
reported alongside the probability itself, so you can see how close a case was, not just whether it passed.

## Use cases

| Domain | Example questions |
|---|---|
| Banking & Finance | Probability a loan reaches default; probability a money-muling ring evades detection |
| Cybersecurity | Probability of privilege escalation succeeding before it's flagged; probability an anomaly generates a false alert |
| E-commerce | Probability of a stockout during a demand spike; probability of losing a customer to an intermittent stockout |
| Fintech | Probability fraud goes undetected; probability risk-drift breaches a compliance threshold |
| Startups | Probability of exhausting runway before closing a funding round; probability an investor walks away after a market shock |
| Supply Chain | Probability a supplier delay cascades into a stockout; probability of recovering from a demand shift within N periods |
