import json
from jsonschema import validate

from core.events import Events

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

        # Funciones internas
        def eval_expr(expr, ctx):
            if isinstance(expr, str):
                try:
                    return eval(expr, {"__builtins__": {}}, ctx)
                except Exception:
                    return expr
            return expr

        def resolve_refs(refs, ctx):
            return {rk: eval_expr(rv, ctx) for rk, rv in refs.items()}

        def resolve_args(keys, ctx):
            return [ctx.get(key) for key in keys]

        def call_method(target_name, method_name, args):
            target_obj = self.agents if target_name == "agents" else self.environments
            try:
                return getattr(target_obj, method_name)(*args)
            except Exception:
                return None

        def resolve_value(vdef, ctx):
            t = vdef["type"]
            if t == "deterministic":
                return eval_expr(vdef["value"], ctx)
            elif t == "probabilistic":
                dist = vdef["distribution"]
                refs = resolve_refs(vdef.get("refs", {}), ctx)
                return self.distributions.sample(dist, refs) if refs else self.distributions.sample(dist)
            elif t == "logic":
                q = vdef["query"]
                args = resolve_args(q.get("params", []), ctx)
                return call_method(q["target"], q["method"], args)
            return None

        # Resolve params
        params = {}
        for k, v in automaton.get("params", {}).items():
            ctx = {**event, **params}
            params[k] = resolve_value(v, ctx)

        # Main loop / Transition Function
        while True:
            if state in final_states:
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            transition = next((t for t in automaton["transitions"] if t["from"] == state), None)
            if not transition:
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            # Resolve _X_
            ctx = {**event, **params}
            _X_ = resolve_value(transition["rule"], ctx)

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
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            # Apply effects
            state = chosen_case["to"]
            ctx = {**event, **params}

            for action in chosen_case.get("effect_order", []):
                if action == "event_emit":
                    obj = chosen_case["event_emit"]
                    resolved = {k: ctx.get(k) for k in obj.get("params", [])}
                    events = Events([{"signal": obj["signal"], **resolved}], self.distributions)
                    await self.agents.receive_event(events[0].agent_id, events[0])

                elif action == "action_required":
                    obj = chosen_case["action_required"]
                    args = resolve_args(obj.get("params", []), ctx)
                    call_method(obj["target"], obj["method"], args)

                elif action == "update_params":
                    updates = {}
                    for k, v in chosen_case.get("update_params", {}).items():
                        updates[k] = resolve_value(v, ctx)
                    params.update(updates)