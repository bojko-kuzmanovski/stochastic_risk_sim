import json
from jsonschema import validate

class Automata:
    def __init__(self, config_data, distributions):
        # Load schema file
        schema_path = "schemas/automata.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Validate against schema
        validate(instance=config_data, schema=schema)

        # Process automata
        self.data = []

        for automaton_entry in config_data:
            automaton_resolved = automaton_entry.copy()

            # Resolve params
            resolved_params = {}
            for param_name, param_def in automaton_entry.get("params", {}).items():
                if param_def["type"] == "deterministic":
                    resolved_params[param_name] = param_def["value"]
                elif param_def["type"] == "probabilistic":
                    resolved_params[param_name] = distributions.sample(param_def["distribution"])

            automaton_resolved["params"] = resolved_params

            # Resolve transitions (only distribution sampling when probabilistic)
            resolved_transitions = []
            for t in automaton_entry.get("transitions", []):
                t_resolved = t.copy()

                if t_resolved.get("type") == "probabilistic" and "distribution" in t_resolved:
                    # keep structure, distribution stays as definition-level parameter
                    pass

                # resolve threshold updates if present
                if "thresholds" in t_resolved:
                    new_thresholds = []
                    for th in t_resolved["thresholds"]:
                        th_resolved = th.copy()

                        if "update" in th_resolved:
                            new_update = {}
                            for k, v in th_resolved["update"].items():
                                if v.get("type") == "deterministic":
                                    new_update[k] = v["value"]
                                elif v.get("type") == "probabilistic":
                                    new_update[k] = distributions.sample(v["distribution"])
                            th_resolved["update"] = new_update

                        new_thresholds.append(th_resolved)

                    t_resolved["thresholds"] = new_thresholds

                resolved_transitions.append(t_resolved)

            automaton_resolved["transitions"] = resolved_transitions

            self.data.append(automaton_resolved)