import json
from jsonschema import validate
import re

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
            _X_ = resolve_value(transition["rule"], ctx, self.distributions, self.agents, self.environments)

            # Evaluate thresholds
            chosen_case = None
            for th in transition.get("thresholds", []):
                try:
                    if eval(th["threshold"], {"__builtins__": {}}, {"X": _X_}):
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

            # Apply effects
            for action in chosen_case.get("effect_order", []):
                if action == "event_emit":
                    obj = chosen_case["event_emit"]
                    resolved = {k: ctx.get(k) for k in obj.get("params", [])}
                    events = Events([{"signal": obj["signal"], **resolved}], self.distributions)
                    await self.agents.receive_event(events[0].agent_id, events[0])

                elif action == "action_required":
                    obj = chosen_case["action_required"]
                    args = resolve_args(obj.get("params", []), ctx)
                    call_method(obj["target"], obj["method"], args, self.agents, self.environments)

                elif action == "update_params":
                    updates = {}
                    for k, v in chosen_case.get("update_params", {}).items():
                        updates[k] = resolve_value(
                            v, ctx, self.distributions, self.agents, self.environments
                        )
                        ctx[k] = updates[k]
                    params.update(updates)