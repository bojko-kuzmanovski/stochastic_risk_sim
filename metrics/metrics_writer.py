import csv
from pathlib import Path


PATL_HEADER = ["run_id", "automaton_name", "trigger_state", "agent_id", "predicate_id",
               "value", "bound", "operator", "memory_k", "result", "reason"]


class MetricsWriter:
    """
    CSV writer designed for analytical consumption (Pandas, SQL).
    Replicates the exact structured hierarchy of MetricsCollector.print_report()
    into structured CSV records, guaranteeing zero JSON/OrderedDict dumps.
    Runs are written from the main process as they finish, so no locking is required.
    """

    def __init__(self, output_dir: Path, base_name: str):
        self._output_dir = output_dir
        self._base_name = base_name

        self._summary_path = output_dir / f"{base_name}_summary.csv"
        self._des_path = output_dir / f"{base_name}_des.csv"
        self._patl_path = output_dir / f"{base_name}_patl.csv"

        # Reset states
        for f in [self._summary_path, self._des_path, self._patl_path]:
            f.unlink(missing_ok=True)

        with open(self._summary_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "category", "key", "subkey", "value"])
        with open(self._des_path, "w", newline="") as f:
            csv.writer(f).writerow(["run_id", "runtime_section", "entity_key", "metric_subkey", "execution_value"])
        with open(self._patl_path, "w", newline="") as f:
            csv.writer(f).writerow(PATL_HEADER)

    def write_simulation_results(self, result: dict, configs: dict):
        """Writes the summary and DES rows of one finished run."""
        _write_summary_csv(str(self._summary_path), result, configs)
        _write_des_csv(str(self._des_path), result["run_id"], result["metrics"])

    def write_patl_rows(self, rows: list):
        """Writes one row per verified predicate per snapshot."""
        with open(str(self._patl_path), "a", newline="") as f:
            csv.writer(f).writerows(rows)


def _write_summary_csv(csv_path: str, result: dict, configs: dict):
    run_id = result["run_id"]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        def row(category, key, subkey, value):
            writer.writerow([run_id, category, key, subkey, value])

        # METADATA & PERFORMANCE
        row("metadata", "seed", "", result["seed"])
        row("metadata", "simulated_time_reached", "", round(result["final_time"], 6))
        row("performance", "elapsed_des_sec", "", round(result["elapsed_des"], 3))
        row("performance", "elapsed_patl_sec", "", round(result["elapsed_patl"], 3))

        # ==========================================
        # 🧱 BUILDTIME METRICS
        # ==========================================

        # Distributions Buildtime
        dists = configs.get("distributions", [])
        row("buildtime_summary", "distributions_total_declared", "", len(dists))

        for dist_body in dists:
            if not isinstance(dist_body, dict):
                continue

            dist_name = dist_body.get("distribution_name", "unknown")
            family = dist_body.get("family", "unknown")
            row("buildtime_detail", "distribution.family", dist_name, family)

            # Recursive extraction of distribution params/categories to prevent dict dumps
            if "params" in dist_body and isinstance(dist_body["params"], dict):
                for p_key, p_val in dist_body["params"].items():
                    if p_key == "categories" and isinstance(p_val, list):
                        for cat in p_val:
                            if isinstance(cat, dict) and "label" in cat:
                                label = cat["label"]
                                for attr, val in cat.items():
                                    if attr != "label":
                                        row("buildtime_detail", f"distribution_category.{dist_name}", f"{label}.{attr}", val)
                    else:
                        row("buildtime_detail", f"distribution_param.{dist_name}", p_key, p_val)

        # Environments Buildtime
        envs = configs.get("environments", [])
        row("buildtime_summary", "environments_total_declared", "", len(envs))
        for env in envs:
            etype = env.get("environment_type", "unknown")
            row("buildtime_detail", "environment_quantity", etype, env.get("quantity", 1))
            row("buildtime_detail", f"environment_structure.{etype}", "members_count", len(env.get("members", [])))
            row("buildtime_detail", f"environment_structure.{etype}", "relations_count", len(env.get("relations", [])))
            row("buildtime_detail", f"environment_structure.{etype}", "channels_count", len(env.get("channels", [])))

        # Agents Buildtime
        agents = configs.get("agents", [])
        row("buildtime_summary", "agents_total_declared", "", len(agents))
        for agent in agents:
            atype = agent.get("agent_type", "unknown")
            row("buildtime_detail", "agent_quantity", atype, agent.get("quantity", 0))
            for aut in agent.get("automata", []):
                row("buildtime_detail", f"agent_capabilities.{atype}", "assigned_automaton", aut)

        # Automata Buildtime
        automata = configs.get("automata", [])
        row("buildtime_summary", "automata_total_declared", "", len(automata))
        for aut in automata:
            aname = aut.get("automaton_name", "unknown")
            row("buildtime_detail", f"automaton_states.{aname}", "initial_state", aut.get("states", {}).get("initial", ""))
            row("buildtime_detail", f"automaton_states.{aname}", "final_states_count", len(aut.get("states", {}).get("final", [])))
            row("buildtime_detail", f"automaton_structure.{aname}", "transitions_count", len(aut.get("transitions", [])))

        # Static Events Buildtime
        events = configs.get("events", [])
        static_events = [e for e in events if e.get("event_category") == "static"]
        row("buildtime_summary", "static_events_total_declared", "", len(static_events))
        for evt in static_events:
            sig = evt.get("signal", "unknown")
            row("buildtime_detail", f"static_event_target.{sig}", "agent_type", evt.get("agent_type", ""))
            row("buildtime_detail", f"static_event_trigger.{sig}", "periodicity_type", evt.get("periodicity", {}).get("type", ""))

        # PATL Specification Buildtime
        patl_data = result["patl_spec"]
        total_preds = sum(len(preds) for preds in patl_data.values())
        row("buildtime_summary", "patl_predicates_total_declared", "", total_preds)
        row("buildtime_summary", "patl_observations_total_declared", "", len(patl_data))
        for (aut_name, state_name), preds in patl_data.items():
            for pred in preds:
                pid = pred.get("predicate_id", "unknown")
                row("buildtime_detail", f"patl_specification.{aut_name}.{state_name}", pid, pred.get("type", "reachability"))


def _write_des_csv(csv_path: str, run_id: int, data: dict):
    """
    Writes the runtime telemetry directly into its own transactional CSV log.
    Ensures that every execution path matches the granular tree layout.
    """
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # 1.🎲 Distribution Runtime Usage
        for dist_name, families in data.get("distribution_usage", {}).items():
            for family, count in families.items():
                writer.writerow([run_id, "distribution_usage", family, dist_name, count])

        # 2.🌍 Environment Runtime Actions
        for env_type, actions in data.get("environment_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "environment_action", env_type, action, count])

        # 3.🤖 Agent Runtime Actions
        for agent_type, actions in data.get("agent_actions", {}).items():
            for action, count in actions.items():
                writer.writerow([run_id, "agent_action", agent_type, action, count])

        # 4.🎯 Automata Executions & Terminal States Reachability (Granular matching to print_report)
        for aut_name, results in data.get("automaton_executions", {}).items():
            for result_type, states in results.items():
                for state, count in states.items():
                    writer.writerow([run_id, "automaton_execution", aut_name, f"{result_type}:{state}", count])

        # 5.📅 Events Generation Frequency
        for event_type, count in data.get("events_generated", {}).items():
            writer.writerow([run_id, "event_generated", "system", event_type, count])

        # 6.📸 PATL Verification Engine Snapshots
        for aut_name, states in data.get("snapshots_captured", {}).items():
            for state, count in states.items():
                writer.writerow([run_id, "snapshot_captured", aut_name, state, count])
