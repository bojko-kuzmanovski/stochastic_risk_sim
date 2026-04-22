import json
from jsonschema import validate

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
        self.metrics = metrics_collector
        self.agents = None
        self.environments = None
    
    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments

    async def process_event(self, event):
        signal = event.get("signal")

        automaton = next(
            (a for a in self.data if a["automaton_name"] == signal),
            None
        )

        if not automaton:
            return None

        automaton_name = automaton["automaton_name"]
        state = automaton["states"]["initial"]
        final_states = set(automaton["states"].get("final", []))

        # Resolve params
        params = {}
        for k, v in automaton.get("params", {}).items():
            if v["type"] == "deterministic":
                params[k] = v["value"]

            elif v["type"] == "probabilistic":
                dist_name = v["distribution"]
                refs = v.get("refs", {})
                params[k] = self.distributions.sample(dist_name, refs) if refs else self.distributions.sample(dist_name)

        while True:

            # Si ya estamos en estado final, registramos y salimos
            if state in final_states:
                self.metrics.record_automaton(automaton_name, state)
                return None

            # Buscar transición válida desde el estado actual
            transition = next(
                (t for t in automaton["transitions"] if t["from"] == state),
                None
            )

            if not transition:
                # No hay transición posible → estado terminal implícito (fallo)
                self.metrics.record_automaton(automaton_name, state)
                return None

            # Resolver valor _X_
            if transition["type"] == "deterministic":
                _X_ = transition.get("value")

            elif transition["type"] == "probabilistic":
                dist_name = transition["distribution"]
                refs = transition.get("refs", {})
                _X_ = self.distributions.sample(dist_name, refs) if refs else self.distributions.sample(dist_name)

            else:
                _X_ = None
            
            # Contexto de evaluación (prioridad: X > event > params)
            eval_context = {}
            eval_context.update(params)
            eval_context.update(event)
            eval_context["_X_"] = _X_

            # Evaluar thresholds
            chosen_case = None

            for th in transition.get("thresholds", []):
                expr = th.get("threshold")

                try:
                    if eval(expr, {"__builtins__": {}}, eval_context):
                        chosen_case = th
                        break
                except Exception:
                    continue

            # Si no hay caso válido → transición fallida
            if not chosen_case:
                self.metrics.record_automaton(automaton_name, state)
                return None

            # Aplicar transición
            state = chosen_case.get("to")
            effect_order = chosen_case.get("effect_order", [])

            for action in effect_order:
                if action == "event_emit" or action == "action_required":
                    try:
                        expr = chosen_case.get(action, "")
                        eval(expr, {"__builtins__": {}}, eval_context)
                    except Exception:
                        pass

                elif action == "update":
                    updates = chosen_case.get("update", {})
                    resolved_updates = {}

                    for k, v in updates.items():
                        if v.get("type") == "deterministic":
                            resolved_updates[k] = v.get("value")

                        elif v.get("type") == "probabilistic":
                            dist_name = v["distribution"]
                            refs = v.get("refs", {})
                            resolved_updates[k] = (
                                self.distributions.sample(dist_name, refs)
                                if refs else self.distributions.sample(dist_name)
                            )

                    params.update(resolved_updates)