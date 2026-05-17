import asyncio
import sys

if sys.version_info >= (3, 14):
    original_del = asyncio.BaseEventLoop.__del__
    
    def safe_del(self):
        try:
            original_del(self)
        except Exception:
            pass
    
    asyncio.BaseEventLoop.__del__ = safe_del


import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent))

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Events

from sim_engine.discrete_event_engine import DiscreteEventSimulator
from sim_engine.event_scheduler import EventScheduler

from patl_engine.patl_verifier import PATLVerifier
from patl_engine.snapshot_manager import SnapshotManager

from metrics.metrics_collector import MetricsCollector


def parse_args():
    parser = argparse.ArgumentParser(description="DES + PATL Simulation")
    parser.add_argument("--output", type=str, default=None,
                        help="Base name for output CSV files (default: timestamp)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent simulations")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Max simulation time per run (seconds)")
    return parser.parse_args()


def load_configs(config_dir: Path) -> dict:
    import json
    configs = {}
    for name in ["distributions", "environments", "automata", "agents", "events", "patl"]:
        path = config_dir / f"{name}.json"
        with open(path, "r") as f:
            configs[name] = json.load(f, object_pairs_hook=OrderedDict)
    return configs


def write_summary_rows(csv_path: str, run_id: int, seed: int, elapsed_des: float,
                       elapsed_patl: float, metrics: MetricsCollector,
                       configs: dict, write_header: bool):
    """
    Write summary in long format: one row per data point.
    Columns: run_id, section, key, subkey, value
    Sections: config, runtime
    """
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "section", "key", "subkey", "value"])

        def w(section, key, subkey, value):
            writer.writerow([run_id, section, key, subkey, value])

        # --- Run metadata ---
        w("run", "seed", "", seed)
        w("run", "elapsed_des_sec", "", round(elapsed_des, 3))
        w("run", "elapsed_patl_sec", "", round(elapsed_patl, 3))

        # --- Config: Distributions ---
        for dist in configs["distributions"]:
            name = dist["distribution_name"]
            family = dist["family"]
            w("config", "distribution", name, family)
            for param, val in dist["params"].items():
                w("config", f"distribution.{name}", param, val)

        # --- Config: Agents ---
        for agent in configs["agents"]:
            atype = agent["agent_type"]
            qty = agent["quantity"]
            w("config", "agent", atype, qty)
            for aut in agent.get("automata", []):
                w("config", f"agent.{atype}", "automaton", aut)

        # --- Config: Automata ---
        for aut in configs["automata"]:
            name = aut["automaton_name"]
            initial = aut["states"]["initial"]
            final_count = len(aut["states"]["final"])
            transition_count = len(aut.get("transitions", []))
            w("config", "automaton", name, initial)
            w("config", f"automaton.{name}", "final_states_count", final_count)
            w("config", f"automaton.{name}", "transition_count", transition_count)

        # --- Config: Environments ---
        for env in configs["environments"]:
            etype = env["environment_type"]
            qty = env.get("quantity", 1)
            w("config", "environment", etype, qty)
            member_count = len(env.get("members", []))
            rel_count = len(env.get("relations", []))
            ch_count = len(env.get("channels", []))
            w("config", f"environment.{etype}", "members", member_count)
            w("config", f"environment.{etype}", "relations", rel_count)
            w("config", f"environment.{etype}", "channels", ch_count)

        # --- Config: Events ---
        for evt in configs["events"]:
            if evt.get("event_category") == "static":
                signal = evt["signal"]
                agent_type = evt["agent_type"]
                periodicity = evt["periodicity"]
                w("config", "event_static", signal, agent_type)
                if periodicity["type"] == "deterministic":
                    w("config", f"event_static.{signal}", "periodicity_value", periodicity["value"])
                    w("config", f"event_static.{signal}", "periodicity_type", "deterministic")
                else:
                    w("config", f"event_static.{signal}", "periodicity_distribution", periodicity.get("distribution", "?"))
                    w("config", f"event_static.{signal}", "periodicity_type", "probabilistic")

        # --- Config: PATL predicates ---
        for obs in configs["patl"].get("observations", []):
            aut = obs["automaton_name"]
            trigger = obs["trigger_state"]
            for pred in obs.get("predicates", []):
                pid = pred["predicate_id"]
                w("config", "patl_predicate", pid, aut)
                w("config", f"patl_predicate.{pid}", "type", pred["type"])
                w("config", f"patl_predicate.{pid}", "trigger_state", trigger)
                w("config", f"patl_predicate.{pid}", "probability_bound", pred["probability_bound"])
                w("config", f"patl_predicate.{pid}", "probability_operator", pred.get("probability_operator", ">="))
                w("config", f"patl_predicate.{pid}", "max_depth", pred.get("max_depth", ""))

        # --- Runtime totals ---
        summary = metrics.get_summary()
        w("runtime", "actions_total", "", summary["actions_total"])
        w("runtime", "automata_exec_total", "", summary["automata_exec_total"])
        w("runtime", "distrib_samples_total", "", summary["distrib_samples_total"])
        w("runtime", "events_total", "", summary["events_total"])
        w("runtime", "snapshots_total", "", summary["snapshots_total"])

        f.flush()


def write_des_rows(csv_path: str, run_id: int, metrics: MetricsCollector, write_header: bool):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "metric_type", "key", "subkey", "value"])

        data = metrics.to_dict()

        # distribution_usage: {dist_name: {family: count}}
        for dist_name, families in data.get("distribution_usage", {}).items():
            for family, count in families.items():
                writer.writerow([run_id, "distribution_usage", dist_name, family, count])

        # environment_actions: {env_type: {action: count}}
        for env_type, actions in data.get("environment_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "environment_action", env_type, action, count])

        # agent_actions: {agent_type: {action: count}}
        for agent_type, actions in data.get("agent_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "agent_action", agent_type, action, count])

        # automaton_executions: {aut_name: {result: {state: count}}}
        for aut_name, results in data.get("automaton_executions", {}).items():
            for result_type, states in results.items():
                for state, count in states.items():
                    writer.writerow([run_id, "automaton_execution", aut_name, f"{result_type}:{state}", count])

        # events_generated: {static: count, dynamic: count}
        events = data.get("events_generated", {})
        for event_type, count in events.items():
            writer.writerow([run_id, "event_generated", event_type, "", count])

        # snapshots_captured: {aut_name: {state: count}}
        for aut_name, states in data.get("snapshots_captured", {}).items():
            for state, count in states.items():
                writer.writerow([run_id, "snapshot_captured", aut_name, state, count])

        f.flush()


def write_patl_rows(csv_path: str, run_id: int, patl_results: list, write_header: bool):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "automaton_name", "trigger_state", "agent_id",
                             "predicate_id", "p_value", "bound", "operator", "result"])
        for snap, results in patl_results:
            for r in results:
                writer.writerow([
                    run_id, snap["automaton_name"], snap["state"], snap["agent_id"],
                    r["predicate_id"], r["p_value"], r["bound"], r["operator"], r["result"]
                ])
        f.flush()


async def main():
    args = parse_args()

    # Output directory and base name
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output if args.output else datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv = output_dir / f"{base_name}_summary.csv"
    des_csv = output_dir / f"{base_name}_des.csv"
    patl_csv = output_dir / f"{base_name}_patl.csv"

    # Remove existing files for this base_name
    for f in [summary_csv, des_csv, patl_csv]:
        f.unlink(missing_ok=True)

    # Load configs once
    config_dir = Path(__file__).parent / "configs/startups/runway-risk"
    configs = load_configs(config_dir)

    from tqdm import tqdm
    pbar = tqdm(
        total=args.runs,
        desc="⠋ Simulations",
        unit="run",
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    )

    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    for run_id in range(1, args.runs + 1):
        seed = int(time.time() * 1000) + run_id

        # --- BUILD ---
        metrics = MetricsCollector(enabled=True)
        distributions = Distributions(configs["distributions"], metrics)
        environments = Environments(configs["environments"], distributions, metrics)
        automata = Automata(configs["automata"], distributions, metrics)
        agents = Agents(configs["agents"], distributions, metrics)

        automata.set_objects(agents, environments)

        snapshot_manager = SnapshotManager(configs["patl"], metrics)
        snapshot_manager.set_objects(agents, environments)
        agents.set_objects(automata, snapshot_manager)

        events = Events(configs["events"], distributions)
        static_events = [e for e in events.data if e["event_category"] == "static"]
        event_scheduler = EventScheduler(static_events, agents, snapshot_manager)

        sim = DiscreteEventSimulator(
            distributions, environments, automata, agents, events,
            event_scheduler, snapshot_manager
        )

        # --- Spinner runs during both DES and PATL ---
        spin_idx = 0
        t0 = time.time()

        async def run_all():
            # DES
            await sim.run_simulation(max_time=args.time)
            # PATL
            nonlocal patl_results, elapsed_patl
            snapshots = snapshot_manager.get_all_snapshots()
            patl_verifier = PATLVerifier(automata, distributions, metrics)
            t1 = time.time()
            patl_results = []
            for snap in snapshots:
                key = (snap["automaton_name"], snap["state"])
                predicates = snapshot_manager.data.get(key, [])
                if predicates:
                    results = patl_verifier.verify(snap, predicates)
                    patl_results.append((snap, results))
            elapsed_patl = time.time() - t1

        async def spinner_loop():
            nonlocal spin_idx
            while not run_task.done():
                pbar.set_description(f"{spinner[spin_idx % len(spinner)]} Simulations")
                spin_idx += 1
                await asyncio.sleep(0.06)

        patl_results = []
        elapsed_patl = 0.0

        run_task = asyncio.create_task(run_all())
        spin_task = asyncio.create_task(spinner_loop())

        await run_task
        spin_task.cancel()
        try:
            await spin_task
        except asyncio.CancelledError:
            pass

        elapsed_des = time.time() - t0 - elapsed_patl

        # --- WRITE ---
        write_header = (run_id == 1)
        write_summary_rows(str(summary_csv), run_id, seed, elapsed_des, elapsed_patl,
                           metrics, configs, write_header)
        write_des_rows(str(des_csv), run_id, metrics, write_header)
        write_patl_rows(str(patl_csv), run_id, patl_results, write_header)

        # --- CLEANUP ---
        del sim, event_scheduler, snapshot_manager
        del agents, automata, environments, distributions, events, metrics

        pbar.update(1)

    pbar.set_description("✔ Simulations")
    pbar.close()
    print(f"\nResults written to: data/{base_name}_summary.csv, data/{base_name}_des.csv, data/{base_name}_patl.csv")


if __name__ == "__main__":
    asyncio.run(main())