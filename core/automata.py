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
            
            elif v["type"] == "logic":
                query = v["query"]
                target_name = query["target"]
                method_name = query["method"]
                param_keys = query.get("params", [])
                
                # Resolve from event
                resolved_args = []
                for key in param_keys:
                    resolved_args.append(event.get(key))
                
                # Select target
                target_obj = self.agents if target_name == "agents" else self.environments
                
                try:
                    # Ejecute method
                    method = getattr(target_obj, method_name)
                    params[k] = method(*resolved_args)
                except Exception:
                    params[k] = None

        while True:

            # Si ya estamos en estado final, registramos y salimos
            if state in final_states:
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            # Buscar transición válida desde el estado actual
            transition = next(
                (t for t in automaton["transitions"] if t["from"] == state),
                None
            )

            if not transition:
                # No hay transición posible → estado terminal implícito (fallo)
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            # Resolver valor _X_
            if transition["rule"]["type"] == "deterministic":
                value = transition["rule"]["value"]
                # Si es una referencia a params
                if isinstance(value, str) and value.startswith("params."):
                    # Valor referencial
                    key = value.split(".", 1)[1]
                    _X_ = params.get(key, value)
                else:
                    # Valor literal
                    _X_ = value

            elif transition["rule"]["type"] == "probabilistic":
                dist_name = transition["rule"]["distribution"]
                refs = transition["rule"].get("refs", {})
                _X_ = self.distributions.sample(dist_name, refs) if refs else self.distributions.sample(dist_name)

            elif transition["rule"]["type"] == "logic":
                query = transition["rule"]["query"]
                target_name = query["target"]
                method_name = query["method"]
                param_keys = query.get("params", [])
                
                # Resolver valores con prioridad: params > event
                resolved_args = []
                for key in param_keys:
                    # Primero busca en params, luego en event
                    val = params.get(key)
                    if val is None:
                        val = event.get(key)
                    resolved_args.append(val)
                
                target_obj = self.agents if target_name == "agents" else self.environments
                
                try:
                    method = getattr(target_obj, method_name)
                    _X_ = method(*resolved_args)
                except Exception:
                    _X_ = None

            else:
                _X_ = None

            # Evaluar thresholds
            chosen_case = None

            for th in transition.get("thresholds", []):
                expr = th.get("threshold")

                try:
                    if eval(expr, {"__builtins__": {}}, {"X": _X_}):
                        chosen_case = th
                        break
                except Exception:
                    continue

            # Si no hay caso válido → transición fallida
            if not chosen_case:
                self.metrics_collector.record_automaton_execution(automaton_name, state)
                return None

            # Aplicar efectos de la transición
            state = chosen_case.get("to")
            effect_order = chosen_case.get("effect_order", [])

            for action in effect_order:
                if action == "event_emit":
                    event_emit_obj = chosen_case.get("event_emit")
                    signal = event_emit_obj["signal"]
                    param_keys = event_emit_obj.get("params", [])
                    
                    # Resolver valores con prioridad: params > event
                    resolved_params = {}
                    for key in param_keys:
                        val = params.get(key)
                        if val is None:
                            val = event.get(key)
                        resolved_params[key] = val
                    
                    # Crear objeto Events
                    events = Events([{ "signal": signal, **resolved_params }], self.distributions)
                    
                    # Enviar al agente correspondiente
                    await self.agents.receive_event(events[0].agent_id, events[0])
                    
                elif action == "action_required":
                    action_obj = chosen_case.get("action_required")
                    target_name = action_obj["target"]
                    method_name = action_obj["method"]
                    param_keys = action_obj.get("params", [])
                    
                    # Resolver valores con prioridad: params > event
                    resolved_args = []
                    for key in param_keys:
                        val = params.get(key)
                        if val is None:
                            val = event.get(key)
                        resolved_args.append(val)
                    
                    # Seleccionar el objeto target
                    target_obj = self.agents if target_name == "agents" else self.environments
                    
                    try:
                        method = getattr(target_obj, method_name)
                        method(*resolved_args)
                    except Exception:
                        pass

                elif action == "update_params":
                    updates = chosen_case.get("update_params", {})
                    resolved_updates = {}

                    for k, v in updates.items():
                        if v.get("type") == "deterministic":
                            value = v.get("value")
                            # Si es una referencia a params
                            if isinstance(value, str) and value.startswith("params."):
                                # Valor referencial
                                key = value.split(".", 1)[1]
                                resolved_updates[k] = params.get(key, value)
                            else:
                                # Valor literal
                                resolved_updates[k] = value

                        elif v.get("type") == "probabilistic":
                            dist_name = v["distribution"]
                            refs = v.get("refs", {})
                            resolved_updates[k] = self.distributions.sample(dist_name, refs) if refs else self.distributions.sample(dist_name)

                        elif v.get("type") == "logic":
                            query = v["query"]
                            target_name = query["target"]
                            method_name = query["method"]
                            param_keys = query.get("params", [])
                            
                            # Resolver valores con prioridad: params > event
                            resolved_args = []
                            for key in param_keys:
                                # Primero busca en params, luego en event
                                val = params.get(key)
                                if val is None:
                                    val = event.get(key)
                                resolved_args.append(val)
                            
                            target_obj = self.agents if target_name == "agents" else self.environments
                            
                            try:
                                method = getattr(target_obj, method_name)
                                resolved_updates[k] = method(*resolved_args)
                            except Exception:
                                resolved_updates[k] = None

                    params.update(resolved_updates)