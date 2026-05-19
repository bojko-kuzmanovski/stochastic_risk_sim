import sys
import json
import asyncio
from jsonschema import validate
from typing import Any, Optional

from core.utils.evaluator import resolve_value, call_method, resolve_ephemeral


class Automata:
    def __init__(self, config_data, distributions, metrics_collector, events_obj=None):
        schema_path = "schemas/automata.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)

        names = [d["automaton_name"] for d in config_data]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"automaton_name must be unique. Duplicates: {list(set(duplicates))}"
            )

        self.data = config_data
        self.distributions = distributions
        self.metrics_collector = metrics_collector
        self.agents = None
        self.environments = None
        self.events = events_obj

    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments

    def create_session(self, signal: str, event: dict, async_mode=True) -> Optional["AutomatonSession"]:
        automaton_def = next(
            (a for a in self.data if a["automaton_name"] == signal), None
        )
        if not automaton_def:
            return None
        return AutomatonSession(self, automaton_def, event, async_mode)


class AutomatonSession:
    def __init__(self, automata: Automata, automaton_def: dict, event: dict, async_mode=True):
        self.automata = automata
        self.automaton_def = automaton_def
        self.automaton_name = automaton_def["automaton_name"]
        self.event = event
        self.current_state = automaton_def["states"]["initial"]
        self.final_states = set(automaton_def["states"].get("final", []))
        self.async_mode = async_mode
        self.ctx = {}
        self._init_params()

    def _init_params(self):
        for k, vdef in self.automaton_def.get("params", {}).items():
            self.ctx[k] = resolve_value(vdef, self.automata.distributions)

    def _resolve(self, vdef: Any) -> Any:
        if isinstance(vdef, dict) and "target" in vdef:
            return self._exec_logic(vdef)
        if isinstance(vdef, dict) and "type" in vdef:
            return resolve_value(vdef, self.automata.distributions)
        return resolve_ephemeral(vdef, self.ctx)

    def _exec_logic(self, logic_def: dict) -> Any:
        target = logic_def["target"]
        method = logic_def["method"]
        param_keys = logic_def.get("params", [])

        if target == "system":
            pipeline_def = logic_def["params"]
            initial = self._resolve(pipeline_def["initial_value"])
            operations = []
            for op_def in pipeline_def.get("operations", []):
                operations.append({
                    "operator": op_def["operator"],
                    "with": self._resolve(op_def.get("with", 0))
                })
            return call_method(target, method, [initial, operations],
                             self.automata.agents, self.automata.environments)

        resolved_keys = [resolve_ephemeral(k, self.ctx) for k in param_keys]
        args = [self.ctx.get(k) if isinstance(k, str) and k.startswith("$") else k for k in resolved_keys]
        event_obj = self.event if target == "events" else None
        return call_method(target, method, args, self.automata.agents, self.automata.environments, event_obj)

    def step(self, forced_X=None) -> Optional[str]:
        if self.current_state in self.final_states:
            self.automata.metrics_collector.record_automaton_execution(
                self.automaton_name, "success", self.current_state)
            return None

        transition = next(
            (t for t in self.automaton_def["transitions"] if t["from"] == self.current_state), None)
        if not transition:
            self.automata.metrics_collector.record_automaton_execution(
                self.automaton_name, "failure", self.current_state)
            return None

        tv = transition["threshold_value"]
        tv_key = next(iter(tv))
        tv_def = tv[tv_key]
        X = forced_X if forced_X is not None else self._resolve(tv_def)
        self.ctx[tv_key] = X

        chosen = None
        for th in transition.get("thresholds", []):
            if self._check_case(th["threshold_case"]):
                chosen = th
                break

        if not chosen:
            print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                  f"no threshold_case matched for value {X}", file=sys.stderr)
            sys.exit(1)

        self.current_state = chosen["to"]
        self._apply_actions(chosen.get("actions", []))
        return self.current_state

    def _check_case(self, conditions: list) -> bool:
        for cond in conditions:
            var_name = resolve_ephemeral(cond["variable"], self.ctx)
            op = cond["operator"]
            expected = resolve_ephemeral(cond["value"], self.ctx) if isinstance(cond["value"], str) else cond["value"]
            actual = self.ctx.get(var_name)
            if actual is None and var_name not in self.ctx:
                print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                      f"undefined ephemeral variable '{var_name}' in threshold_rule", file=sys.stderr)
                sys.exit(1)
            try:
                if op == "<" and not (actual < expected):
                    self._fatal_check(var_name, actual, expected, op)
                elif op == "<=" and not (actual <= expected):
                    self._fatal_check(var_name, actual, expected, op)
                elif op == ">" and not (actual > expected):
                    self._fatal_check(var_name, actual, expected, op)
                elif op == ">=" and not (actual >= expected):
                    self._fatal_check(var_name, actual, expected, op)
                elif op == "==" and not (actual == expected):
                    self._fatal_check(var_name, actual, expected, op)
                elif op == "!=" and not (actual != expected):
                    self._fatal_check(var_name, actual, expected, op)
                else:
                    return False
            except TypeError:
                self._fatal_check(var_name, actual, expected, op)
        return True

    def _fatal_check(self, var_name, actual, expected, op):
        print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
              f"threshold_rule failed: {var_name} {op} {expected} "
              f"(actual={actual}, type={type(actual).__name__})", file=sys.stderr)
        sys.exit(1)

    def _apply_actions(self, actions: list):
        for action_obj in actions:
            if "update_params" in action_obj:
                for k, vdef in action_obj["update_params"].items():
                    self.ctx[k] = self._resolve(vdef)

            elif "action_required" in action_obj:
                for k, vdef in action_obj["action_required"].items():
                    result = self._resolve(vdef)
                    self.ctx[k] = result

            elif "event_emit" in action_obj:
                event_array = action_obj["event_emit"]
                resolved = [resolve_ephemeral(k, self.ctx) for k in event_array]
                signal = self.ctx.get(resolved[0]) if resolved[0].startswith("$") else resolved[0]
                agent_id = self.ctx.get(resolved[1]) if resolved[1].startswith("$") else resolved[1]

                event_data = {
                    "event_category": "dynamic",
                    "signal": signal,
                    "agent_id": agent_id
                }
                if len(resolved) > 2:
                    env_id = self.ctx.get(resolved[2]) if resolved[2].startswith("$") else resolved[2]
                    event_data["env_id"] = env_id
                if len(resolved) > 3:
                    channel_id = self.ctx.get(resolved[3]) if resolved[3].startswith("$") else resolved[3]
                    event_data["channel_id"] = channel_id

                if self.automata.events is not None:
                    if not any(
                        ev.get("event_category") == "dynamic"
                        and ev.get("signal") == signal
                        and ev.get("agent_id") == agent_id
                        for ev in self.automata.events.data
                    ):
                        print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                              f"dynamic event not registered in Events: signal='{signal}', "
                              f"agent_id='{agent_id}'", file=sys.stderr)
                        sys.exit(1)

                if agent_id is not None and self.async_mode:
                    asyncio.create_task(self.automata.agents.receive_event(agent_id, event_data))