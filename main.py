import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
# Los esquemas se cargan con rutas relativas al proyecto.
os.chdir(PROJECT_DIR)

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Events

from des.discrete_event_engine import DiscreteEventSimulator
from des.event_scheduler import EventScheduler

from patl.patl_verifier import PATLVerifier
from patl.snapshot_manager import SnapshotManager

from metrics.metrics_collector import MetricsCollector
from metrics.metrics_writer import MetricsWriter


def parse_args():
    parser = argparse.ArgumentParser(description="DES + PATL Simulation")
    parser.add_argument("--output", type=str, default=None,
                        help="Base name for output CSV files (default: timestamp)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent simulations (1-1000)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Simulated time horizon T_max per run, in model time units (10-500)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of simulations run in parallel processes (1-10)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Base seed; run i uses seed * 1000000 + i")
    parser.add_argument("--config-dir", type=str, default="configs/startups/runway-risk",
                        help="Scenario directory with the six JSON configuration files")
    parser.add_argument("--distributions", type=str, default="distributions.json",
                        help="Distributions file inside --config-dir (e.g. distributions_1_base.json)")
    parser.add_argument("--queue-batch", type=int, default=5,
                        help="K: maximum events an agent extracts from its queue per instant")
    parser.add_argument("--memory", type=int, default=1,
                        help="Default memory bound k of coalition strategies (1-4)")
    return parser.parse_args()


def validate_args(args):
    if not (1 <= args.runs <= 1000):
        raise ValueError(f"--runs must be between 1 and 1000, got {args.runs}")
    if not (10 <= args.time <= 500):
        raise ValueError(f"--time must be between 10 and 500, got {args.time}")
    if not (1 <= args.threads <= 10):
        raise ValueError(f"--threads must be between 1 and 10, got {args.threads}")
    if args.queue_batch < 1:
        raise ValueError(f"--queue-batch must be at least 1, got {args.queue_batch}")
    if not (1 <= args.memory <= 4):
        raise ValueError(f"--memory must be between 1 and 4, got {args.memory}")
    if args.threads > args.runs:
        args.threads = args.runs


def load_configs(config_dir: Path, distributions_file: str) -> dict:
    configs = {}
    for name in ["distributions", "environments", "automata", "agents", "events", "patl"]:
        path = config_dir / (distributions_file if name == "distributions" else f"{name}.json")
        with open(path, "r") as f:
            configs[name] = json.load(f, object_pairs_hook=OrderedDict)
    return configs


def run_single_simulation(run_id: int, configs: dict, max_time: float, output_dir: Path, seed: int,
                          queue_batch: int, memory: int, keep_objects: bool = False):
    metrics = MetricsCollector(enabled=True)
    distributions = Distributions(configs["distributions"], metrics, seed=seed)
    environments = Environments(configs["environments"], distributions, metrics)
    automata = Automata(configs["automata"], distributions, metrics)
    agents = Agents(configs["agents"], distributions, metrics)

    automata.set_objects(agents, environments)

    snapshot_manager = SnapshotManager(configs["patl"], metrics, disk_dir=output_dir / ".snapshots" / f"run_{run_id}")
    snapshot_manager.set_objects(agents, environments)
    agents.set_objects(automata, snapshot_manager)

    events = Events(configs["events"], distributions)
    static_events = [e for e in events.data if e["event_category"] == "static"]
    event_scheduler = EventScheduler(static_events, agents, distributions)

    sim = DiscreteEventSimulator(
        distributions, environments, automata, agents, events,
        event_scheduler, snapshot_manager, queue_batch=queue_batch
    )

    # --- DES ---
    t0 = time.time()
    final_T = sim.run_simulation(max_time=max_time)
    elapsed_des = time.time() - t0

    # --- PATL ---
    # El verificador trabaja con su propio intérprete de autómatas y no muestrea.
    t0 = time.time()
    verifier_automata = Automata(configs["automata"], distributions, MetricsCollector(enabled=False))
    verifier = PATLVerifier(verifier_automata, distributions, configs=configs, default_memory=memory)

    patl_rows = []
    for batch in snapshot_manager.read_and_delete(batch_size=50):
        for snap in batch:
            predicates = snapshot_manager.data.get((snap["automaton_name"], snap["state"]), [])
            if not predicates:
                continue
            for r in verifier.verify(snap, predicates):
                patl_rows.append([
                    run_id, snap["automaton_name"], snap["state"], snap["agent_id"],
                    r["predicate_id"], r["value"], r["bound"], r["operator"], r["memory_k"],
                    r["result"], r["reason"]
                ])
    elapsed_patl = time.time() - t0
    snapshot_manager.clear_all_snapshots()

    result = {
        "run_id": run_id,
        "seed": seed,
        "final_time": final_T,
        "elapsed_des": elapsed_des,
        "elapsed_patl": elapsed_patl,
        "metrics": metrics.to_dict(),
        "patl_spec": snapshot_manager.data,
        "patl_rows": patl_rows,
    }
    if keep_objects:
        result["objects"] = (metrics, distributions, environments, agents, automata, events, snapshot_manager)
    return result


def main():
    args = parse_args()
    validate_args(args)

    output_dir = PROJECT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output if args.output else datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = MetricsWriter(output_dir, base_name)

    configs = load_configs(PROJECT_DIR / args.config_dir, args.distributions)

    from tqdm import tqdm
    pbar = tqdm(total=args.runs, desc="Simulations", unit="run",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

    def seed_for(run_id):
        return args.seed * 1_000_000 + run_id

    def record(result):
        writer.write_simulation_results(result, configs)
        writer.write_patl_rows(result["patl_rows"])
        pbar.update(1)

    last = None
    if args.threads == 1:
        for run_id in range(1, args.runs + 1):
            last = run_single_simulation(run_id, configs, args.time, output_dir, seed_for(run_id),
                                         args.queue_batch, args.memory, keep_objects=(args.runs == 1))
            record(last)
    else:
        with ProcessPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(run_single_simulation, run_id, configs, args.time, output_dir,
                                   seed_for(run_id), args.queue_batch, args.memory)
                       for run_id in range(1, args.runs + 1)]
            for future in as_completed(futures):
                record(future.result())

    pbar.close()

    if args.runs == 1 and last is not None:
        metrics, distributions, environments, agents, automata, events, snapshot_manager = last["objects"]
        metrics.print_report(distributions, environments, agents, automata, events, snapshot_manager)
        print(f"\nSimulated time reached: {last['final_time']:.2f}")
        print(f"DES: {last['elapsed_des']:.1f}s | PATL: {last['elapsed_patl']:.1f}s")

    print(f"\nResults written to: data/{base_name}_summary.csv, data/{base_name}_des.csv, data/{base_name}_patl.csv")


if __name__ == "__main__":
    main()
