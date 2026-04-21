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
            resolved_params = automaton_entry.get("params", {}).copy()
            automaton_resolved["params"] = resolved_params

            # Resolve internal_vars
            resolved_internal_vars = {}
            for internal_var_name, internal_var_def in automaton_entry.get("internal_vars", {}).items():
                if internal_var_def["type"] == "deterministic":
                    resolved_internal_vars[internal_var_name] = internal_var_def["value"]

                elif internal_var_def["type"] == "probabilistic":
                    dist_name = internal_var_def["distribution"]
                    resolved_internal_vars[internal_var_name] = distributions.sample(dist_name)
            
            # Assign resolved internal_vars
                automaton_entry["internal_vars"] = resolved_internal_vars

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