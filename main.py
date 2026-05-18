import asyncio
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import concurrent.futures

if sys.version_info >= (3, 14):
    original_del = asyncio.BaseEventLoop.__del__
    
    def safe_del(self):
        try:
            original_del(self)
        except Exception:
            pass
    
    asyncio.BaseEventLoop.__del__ = safe_del


sys.path.insert(0, str(Path(__file__).parent))

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
    configs = {}
    for name in ["distributions", "environments", "automata", "agents", "events", "patl"]:
        path = config_dir / f"{name}.json"
        with open(path, "r") as f:
            configs[name] = json.load(f, object_pairs_hook=OrderedDict)
    return configs


async def run_single_simulation(run_id: int, configs: dict, max_time: float, writer: MetricsWriter, output_dir: str):
    seed = int(time.time() * 1000) + run_id

    metrics = MetricsCollector(enabled=True)
    distributions = Distributions(configs["distributions"], metrics)
    environments = Environments(configs["environments"], distributions, metrics)
    automata = Automata(configs["automata"], distributions, metrics)
    agents = Agents(configs["agents"], distributions, metrics)

    automata.set_objects(agents, environments)

    snapshot_manager = SnapshotManager(configs["patl"], metrics, disk_dir=output_dir / ".snapshots" / f"run_{run_id}")
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

    # --- WRITE summary and des IMMEDIATELY, free memory ---
    await writer.write_summary_rows(run_id, seed, elapsed_des, 0.0, metrics, configs)
    await writer.write_des_rows(run_id, metrics)
    metrics_for_report = metrics

    # --- PATL ---
    t0 = time.time()
    
    def _verify_single(snap, automata_obj, dists, snapshot_data):
        verifier = PATLVerifier(automata_obj, dists)
        key = (snap["automaton_name"], snap["state"])
        predicates = snapshot_data.get(key, [])
        if not predicates:
            return None
        results = verifier.verify(snap, predicates)
        return (snap, results)

    loop = asyncio.get_running_loop()
    total_processed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for batch in snapshot_manager.read_and_delete(batch_size=50):
            futures = [
                loop.run_in_executor(pool, _verify_single, snap, automata, distributions, snapshot_manager.data)
                for snap in batch
            ]
            batch_results = []
            for coro in asyncio.as_completed(futures):
                result = await coro
                if result is not None:
                    batch_results.append(result)
            
            if batch_results:
                await writer.write_patl_rows(run_id, batch_results)
                total_processed += len(batch_results)
    
    elapsed_patl = time.time() - t0
    snapshot_manager.clear_all_snapshots()

    return {
        "metrics": metrics_for_report,
        "distributions": distributions,
        "environments": environments,
        "agents": agents,
        "automata": automata,
        "events": events,
        "snapshot_manager": snapshot_manager,
        "elapsed_des": elapsed_des,
        "elapsed_patl": elapsed_patl,
        "snapshots_processed": total_processed
    }


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

    last_objects = {}

    async def run_with_semaphore(run_id):
        async with sem:
            result = await run_single_simulation(run_id, configs, args.time, writer, output_dir)
        pbar.update(1)
        if args.runs == 1:
            last_objects["result"] = result

    spin_idx = 0

    async def spinner_loop():
        nonlocal spin_idx
        while not all_tasks_done:
            pbar.set_description(f"{spinner[spin_idx % len(spinner)]} Simulations")
            pbar.refresh()
            spin_idx += 1
            await asyncio.sleep(0.06)

    all_tasks_done = False
    spin_task = asyncio.create_task(spinner_loop())

    tasks = [asyncio.create_task(run_with_semaphore(i)) for i in range(1, args.runs + 1)]
    await asyncio.gather(*tasks)
    all_tasks_done = True
    spin_task.cancel()
    try:
        await spin_task
    except asyncio.CancelledError:
        pass

    pbar.set_description("✔ Simulations")
    pbar.close()

    if args.runs == 1 and "result" in last_objects:
        result = last_objects["result"]
        result["metrics"].print_report(
            result["distributions"],
            result["environments"],
            result["agents"],
            result["automata"],
            result["events"],
            result["snapshot_manager"]
        )
        print(f"\nSnapshots processed: {result['snapshots_processed']}")
        print(f"DES: {result['elapsed_des']:.1f}s | PATL: {result['elapsed_patl']:.1f}s")

    print(f"\nResults written to: data/{base_name}_summary.csv, data/{base_name}_des.csv, data/{base_name}_patl.csv")


if __name__ == "__main__":
    asyncio.run(main())