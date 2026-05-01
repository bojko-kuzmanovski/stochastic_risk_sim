import json
from jsonschema import validate
from typing import Any
import asyncio

from sim_core.events import Events
from sim_core.utils.evaluator import resolve_args, call_method, resolve_value

class Automata:
    def __init__(self, config_data, distributions, metrics_collector):
        # Load schema file
        schema_path = "schemas/automata.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Validate against schema
        validate(instance=config_data, schema=schema)

        # Process automata
        self.data = config_data
        self.distributions = distributions
        self.metrics_collector = metrics_collector
        self.agents = None
        self.environments = None
    
    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments

    async def process_event(self, event):
        signal = event.get("signal")
        automaton = next((a for a in self.data if a["automaton_name"] == signal), None)
        if not automaton:
            return None

        automaton_name = automaton["automaton_name"]
        state = automaton["states"]["initial"]
        final_states = set(automaton["states"].get("final", []))

        # Resolve params
        params = {}
        for k, v in automaton.get("params", {}).items():
            ctx = {**event, **params}
            params[k] = resolve_value(
                v, ctx, self.distributions, self.agents, self.environments
            )
        
        # Mutiple case evaluation
        def _evaluate_threshold_conditions(X: Any, conditions: list, ctx: dict) -> bool:
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

        # Main loop / Transition Function
        while True:
            if state in final_states:
                self.metrics_collector.record_automaton_execution(automaton_name, "sucess", state)
                return None

            transition = next((t for t in automaton["transitions"] if t["from"] == state), None)
            if not transition:
                self.metrics_collector.record_automaton_execution(automaton_name, "failure", state)
                return None

            # Resolve _X_
            ctx = {**event, **params}
            _X_ = resolve_value(transition["threshold_value"], ctx, self.distributions, self.agents, self.environments)

            # Evaluate thresholds
            chosen_case = None
            for th in transition.get("thresholds", []):
                try:
                    if _evaluate_threshold_conditions(_X_, th["threshold_case"], ctx):
                        chosen_case = th
                        break
                except Exception:
                    continue

            if not chosen_case:
                self.metrics_collector.record_automaton_execution(automaton_name, "failure", state)
                return None

            # Apply transition
            state = chosen_case["to"]
            ctx = {**event, **params, "X": _X_}

            # Apply actions in order
            for action_obj in chosen_case.get("actions", []):
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
                        asyncio.create_task(self.agents.receive_event(to_agent_id, event_data))

                elif "action_required" in action_obj:
                    obj = action_obj["action_required"]
                    args = resolve_args(obj.get("params", []), ctx)
                    call_method(obj["target"], obj["method"], args, self.agents, self.environments)

                elif "update_params" in action_obj:
                    updates = {}
                    for k, v in action_obj["update_params"].items():
                        updates[k] = resolve_value(
                            v, ctx, self.distributions, self.agents, self.environments
                        )
                        ctx[k] = updates[k]
                    params.update(updates)