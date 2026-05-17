import asyncio
import sys
import concurrent.futures

if sys.version_info >= (3, 14):
    original_del = asyncio.BaseEventLoop.__del__
    
    def safe_del(self):
        try:
            original_del(self)
        except Exception:
            pass
    
    asyncio.BaseEventLoop.__del__ = safe_del


import argparse
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
from metrics.metrics_writer import MetricsWriter


def parse_args():
    parser = argparse.ArgumentParser(description="DES + PATL Simulation")
    parser.add_argument("--output", type=str, default=None,
                        help="Base name for output CSV files (default: timestamp)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent simulations (1-1000)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Max simulation time per run in seconds (10-500)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of parallel simulations (1-10)")
    return parser.parse_args()


def validate_args(args):
    if not (1 <= args.runs <= 1000):
        raise ValueError(f"--runs must be between 1 and 1000, got {args.runs}")
    if not (10 <= args.time <= 500):
        raise ValueError(f"--time must be between 10 and 500, got {args.time}")
    if not (1 <= args.threads <= 10):
        raise ValueError(f"--threads must be between 1 and 10, got {args.threads}")
    if args.threads > args.runs:
        args.threads = args.runs


def load_configs(config_dir: Path) -> dict:
    import json
    configs = {}
    for name in ["distributions", "environments", "automata", "agents", "events", "patl"]:
        path = config_dir / f"{name}.json"
        with open(path, "r") as f:
            configs[name] = json.load(f, object_pairs_hook=OrderedDict)
    return configs


async def run_single_simulation(run_id: int, configs: dict, max_time: float, writer: MetricsWriter):
    seed = int(time.time() * 1000) + run_id

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

    # --- DES ---
    t0 = time.time()
    await sim.run_simulation(max_time=max_time)
    elapsed_des = time.time() - t0

    # --- PATL (run in thread executor to avoid blocking event loop) ---
    snapshots = snapshot_manager.get_all_snapshots()
    patl_verifier = PATLVerifier(automata, distributions, metrics)

    t0 = time.time()
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        patl_results = await loop.run_in_executor(
            pool,
            lambda: _run_patl_sync(snapshots, snapshot_manager, patl_verifier)
        )
    elapsed_patl = time.time() - t0

    # --- WRITE (thread-safe) ---
    await writer.write_summary_rows(run_id, seed, elapsed_des, elapsed_patl, metrics, configs)
    await writer.write_des_rows(run_id, metrics)
    await writer.write_patl_rows(run_id, patl_results)


def _run_patl_sync(snapshots, snapshot_manager, patl_verifier):
    """Synchronous PATL verification for thread executor."""
    patl_results = []
    for snap in snapshots:
        key = (snap["automaton_name"], snap["state"])
        predicates = snapshot_manager.data.get(key, [])
        if predicates:
            results = patl_verifier.verify(snap, predicates)
            patl_results.append((snap, results))
    return patl_results


async def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output if args.output else datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = MetricsWriter(output_dir, base_name)

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
    sem = asyncio.Semaphore(args.threads)

    async def run_with_semaphore(run_id):
        async with sem:
            await run_single_simulation(run_id, configs, args.time, writer)
        pbar.update(1)
        pbar.refresh()

    spin_idx = 0

    async def spinner_loop():
        nonlocal spin_idx
        while not all_tasks_done:
            pbar.set_description(f"{spinner[spin_idx % len(spinner)]} Simulations")
            pbar.refresh()
            spin_idx += 1
            await asyncio.sleep(0.06)

    tasks = [asyncio.create_task(run_with_semaphore(i)) for i in range(1, args.runs + 1)]
    all_tasks_done = False

    spin_task = asyncio.create_task(spinner_loop())
    await asyncio.gather(*tasks)
    all_tasks_done = True
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass

    pbar.set_description("✔ Simulations")
    pbar.close()
    print(f"\nResults written to: data/{base_name}_summary.csv, data/{base_name}_des.csv, data/{base_name}_patl.csv")


if __name__ == "__main__":
    asyncio.run(main())