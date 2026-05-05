import json
from jsonschema import validate
from typing import Any, Optional
import asyncio

from sim_core.utils.evaluator import resolve_args, call_method, resolve_value


class Automata:
    def __init__(self, config_data, distributions, metrics_collector):
        # Load schema file
        schema_path = "schemas/automata.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Validate against schema
        validate(instance=config_data, schema=schema)

        # Validate unique automaton_name
        names = [d["automaton_name"] for d in config_data]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"automaton_name must be unique. Duplicates: {list(set(duplicates))}"
            )

        # Process automata
        self.data = config_data
        self.distributions = distributions
        self.metrics_collector = metrics_collector
        self.agents = None
        self.environments = None


    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments


    def create_session(self, signal: str, event: dict) -> Optional["AutomatonSession"]:
        automaton_def = next(
            (a for a in self.data if a["automaton_name"] == signal), None
        )
        if not automaton_def:
            return None

        return AutomatonSession(
            automata=self,
            automaton_def=automaton_def,
            event=event
        )


    def _evaluate_threshold_conditions(self, X: Any, conditions: list, ctx: dict) -> bool:
        for cond in conditions:
            op = cond["operator"]
            val = resolve_value(
                {"type": "deterministic", "value": cond["value"]},
                {**ctx, "X": X},
                self.distributions,
                self.agents,
                self.environments
            )

            if op == "<" and not (X < val):
                return False
            elif op == "<=" and not (X <= val):
                return False
            elif op == ">" and not (X > val):
                return False
            elif op == ">=" and not (X >= val):
                return False
            elif op == "==" and not (X == val):
                return False
        return True


class AutomatonSession:
    def __init__(self, automata: Automata, automaton_def: dict, event: dict, initial_state=None, async_mode=True):
        self.automata = automata
        self.automaton_def = automaton_def
        self.automaton_name = automaton_def["automaton_name"]
        self.event = event

        self.current_state = initial_state if initial_state is not None else automaton_def["states"]["initial"]
        self.final_states = set(automaton_def["states"].get("final", []))
        self.async_mode = async_mode
        self.params = self._resolve_initial_params()


    def _resolve_initial_params(self) -> dict:
        params = {}
        for k, v in self.automaton_def.get("params", {}).items():
            ctx = {**self.event, **params}
            params[k] = resolve_value(
                v, ctx,
                self.automata.distributions,
                self.automata.agents,
                self.automata.environments
            )
        return params


    def step(self) -> Optional[str]:
        # Done?
        if self.current_state in self.final_states:
            self.automata.metrics_collector.record_automaton_execution(
                self.automaton_name, "success", self.current_state
            )
            return None

        # Look for transition from current state
        transition = next(
            (t for t in self.automaton_def["transitions"]
             if t["from"] == self.current_state),
            None
        )
        if not transition:
            self.automata.metrics_collector.record_automaton_execution(
                self.automaton_name, "failure", self.current_state
            )
            return None

        # Resolve _X_
        ctx = {**self.event, **self.params}
        _X_ = resolve_value(
            transition["threshold_value"], ctx,
            self.automata.distributions,
            self.automata.agents,
            self.automata.environments
        )

        # Evaluate thresholds
        chosen_case = None
        for th in transition.get("thresholds", []):
            try:
                if self.automata._evaluate_threshold_conditions(
                    _X_, th["threshold_case"], ctx
                ):
                    chosen_case = th
                    break
            except Exception:
                continue

        if not chosen_case:
            self.automata.metrics_collector.record_automaton_execution(
                self.automaton_name, "failure", self.current_state
            )
            return None

        # Apply actions
        self.current_state = chosen_case["to"]
        ctx = {**self.event, **self.params, "X": _X_}
        self._execute_actions(chosen_case.get("actions", []), ctx)

        return self.current_state


    def _execute_actions(self, actions: list, ctx: dict):
        for action_obj in actions:
            if "event_emit" in action_obj:
                obj = action_obj["event_emit"]
                resolved = {k: ctx.get(k) for k in obj.get("params", [])}

                to_agent_id = resolved.get("to_agent_id")
                if to_agent_id is not None:
                    event_data = {
                        "event_category": "dynamic",
                        "signal": obj["signal"],
                        "to_agent_id": to_agent_id,
                        **resolved
                    }
                    if self.async_mode:
                        asyncio.create_task(
                            self.automata.agents.receive_event(to_agent_id, event_data)
                        )
                    else:
                        pass

            elif "action_required" in action_obj:
                obj = action_obj["action_required"]
                args = resolve_args(obj.get("params", []), ctx)
                call_method(
                    obj["target"], obj["method"], args,
                    self.automata.agents,
                    self.automata.environments
                )

            elif "update_params" in action_obj:
                for k, v in action_obj["update_params"].items():
                    resolved_val = resolve_value(
                        v, ctx,
                        self.automata.distributions,
                        self.automata.agents,
                        self.automata.environments
                    )
                    ctx[k] = resolved_val
                    self.params[k] = resolved_val