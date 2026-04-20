import json
import random
import numpy as np
from jsonschema import validate

class Distributions:
    def __init__(self, config_data):
        schema_path = "schemas/distributions.schema.json"

        # Load schema file
        with open(schema_path, 'r') as f:
            self._schema = json.load(f)
        
        # Validate data against schema
        validate(instance=config_data, schema=self._schema)
        
        # Store configurations indexed by name
        self._configs = {d["distribution_name"]: d for d in self._data}

    def sample(self, name, bound_params=None):
        config = self._configs.get(name)
        if not config:
            raise ValueError(f"Distribution {name} not found")

        # Extract config properties
        family = config["family"]
        output_type = config["output_type"]
        params = config.get("params", {})
        labels = config.get("labels", [])
        truncation = config.get("truncation")

        # Unify parameters: bound_params overrides default params
        p = {**params, **(bound_params or {})}
        
        max_attempts = 1000
        for _ in range(max_attempts):
            if family == "categorical":
                val = random.choices(labels, weights=p.get("probabilities"))[0]
            else:
                # Mapping from JSON names to random library functions
                dispatch = {
                    "exponential": (random.expovariate, ["rate"]),
                    "poisson": (np.random.poisson, ["lambda"]),
                    "gamma": (random.gammavariate, ["shape", "scale"]),
                    "beta": (random.betavariate, ["alpha", "beta"]),
                    "lognormal": (random.lognormvariate, ["mean", "sigma"])
                }
                func, arg_keys = dispatch[family]
                val = func(*(p[k] for k in arg_keys))

            if family != "categorical" and truncation:
                if not (truncation["min"] <= val <= truncation["max"]):
                    continue # Reintento

            # Normalize output type
            if output_type == "int":
                final_val = int(val)
            elif output_type == "float":
                final_val = round(float(val), 4)
            else:
                final_val = str(val)

            return {
                "output_type": output_type,
                "value": final_val
            }
            
        raise ValueError(
            f"Truncation too restrictive for distribution {name}"
        )