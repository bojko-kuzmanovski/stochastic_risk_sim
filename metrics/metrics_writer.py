import csv
import asyncio
from pathlib import Path


class MetricsWriter:
    """
    Thread-safe CSV writer for simulation results.
    Centralizes all file writes with an asyncio lock to prevent
    concurrent write corruption when running multiple simulations.
    """

    def __init__(self, output_dir: Path, base_name: str):
        self._lock = asyncio.Lock()

        self._output_dir = output_dir
        self._base_name = base_name

        self._summary_path = output_dir / f"{base_name}_summary.csv"
        self._des_path = output_dir / f"{base_name}_des.csv"
        self._patl_path = output_dir / f"{base_name}_patl.csv"

        # Track whether header has been written for each file
        self._summary_header_written = False
        self._des_header_written = False
        self._patl_header_written = False

        # Remove existing files for this base_name
        for f in [self._summary_path, self._des_path, self._patl_path]:
            f.unlink(missing_ok=True)

    # --- Public API ---

    async def write_summary_rows(self, run_id: int, seed: int,
                                 elapsed_des: float, elapsed_patl: float,
                                 metrics, configs: dict):
        async with self._lock:
            write_header = not self._summary_header_written
            self._summary_header_written = True
            _write_summary_csv(str(self._summary_path), run_id, seed,
                              elapsed_des, elapsed_patl, metrics, configs, write_header)

    async def write_des_rows(self, run_id: int, metrics):
        async with self._lock:
            write_header = not self._des_header_written
            self._des_header_written = True
            _write_des_csv(str(self._des_path), run_id, metrics, write_header)

    async def write_patl_rows(self, run_id: int, patl_results: list):
        async with self._lock:
            write_header = not self._patl_header_written
            self._patl_header_written = True
            _write_patl_csv(str(self._patl_path), run_id, patl_results, write_header)


# --- Private CSV writers ---

def _write_summary_csv(csv_path: str, run_id: int, seed: int,
                       elapsed_des: float, elapsed_patl: float,
                       metrics, configs: dict, write_header: bool):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "section", "key", "subkey", "value"])

        def w(section, key, subkey, value):
            writer.writerow([run_id, section, key, subkey, value])

        w("run", "seed", "", seed)
        w("run", "elapsed_des_sec", "", round(elapsed_des, 3))
        w("run", "elapsed_patl_sec", "", round(elapsed_patl, 3))

        for dist in configs["distributions"]:
            name = dist["distribution_name"]
            family = dist["family"]
            w("config", "distribution", name, family)
            for param, val in dist["params"].items():
                w("config", f"distribution.{name}", param, val)

        for agent in configs["agents"]:
            atype = agent["agent_type"]
            qty = agent["quantity"]
            w("config", "agent", atype, qty)
            for aut in agent.get("automata", []):
                w("config", f"agent.{atype}", "automaton", aut)

        for aut in configs["automata"]:
            name = aut["automaton_name"]
            initial = aut["states"]["initial"]
            final_count = len(aut["states"]["final"])
            transition_count = len(aut.get("transitions", []))
            w("config", "automaton", name, initial)
            w("config", f"automaton.{name}", "final_states_count", final_count)
            w("config", f"automaton.{name}", "transition_count", transition_count)

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

        summary = metrics.get_summary()
        w("runtime", "actions_total", "", summary["actions_total"])
        w("runtime", "automata_exec_total", "", summary["automata_exec_total"])
        w("runtime", "distrib_samples_total", "", summary["distrib_samples_total"])
        w("runtime", "events_total", "", summary["events_total"])
        w("runtime", "snapshots_total", "", summary["snapshots_total"])

        f.flush()


def _write_des_csv(csv_path: str, run_id: int, metrics, write_header: bool):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "metric_type", "key", "subkey", "value"])

        data = metrics.to_dict()

        for dist_name, families in data.get("distribution_usage", {}).items():
            for family, count in families.items():
                writer.writerow([run_id, "distribution_usage", dist_name, family, count])

        for env_type, actions in data.get("environment_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "environment_action", env_type, action, count])

        for agent_type, actions in data.get("agent_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "agent_action", agent_type, action, count])

        for aut_name, results in data.get("automaton_executions", {}).items():
            for result_type, states in results.items():
                for state, count in states.items():
                    writer.writerow([run_id, "automaton_execution", aut_name, f"{result_type}:{state}", count])

        for event_type, count in data.get("events_generated", {}).items():
            writer.writerow([run_id, "event_generated", event_type, "", count])

        for aut_name, states in data.get("snapshots_captured", {}).items():
            for state, count in states.items():
                writer.writerow([run_id, "snapshot_captured", aut_name, state, count])

        f.flush()


def _write_patl_csv(csv_path: str, run_id: int, patl_results: list, write_header: bool):
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