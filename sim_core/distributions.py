import json
import random
import numpy as np
from jsonschema import validate

class Distributions:
    def __init__(self, config_data, metrics_collector):
        schema_path = "schemas/distributions.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)

        # Validate unique distribution_name
        names = [d["distribution_name"] for d in config_data]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"distribution_name must be unique. Duplicates: {list(set(duplicates))}"
            )

        self.metrics_collector = metrics_collector
        self.samplers = {}

        for d in config_data:
            name = d["distribution_name"]
            family = d["family"]
            output_type = d["output_type"]
            params = d.get("params", {})
            labels = d.get("labels", [])
            truncation = d.get("truncation")

            def make_sampler(dist_name, family, output_type, params, labels, truncation):
                def sampler(bound_params=None):
                    p = {**params, **(bound_params or {})}
                    max_attempts = 1000

                    for _ in range(max_attempts):
                        if family == "categorical":
                            val = random.choices(labels, weights=p.get("probabilities"))[0]
                        elif family == "normal":
                            val = random.normalvariate(p["mean"], p["sigma"])
                        elif family == "exponential":
                            val = random.expovariate(p["rate"])
                        elif family == "poisson":
                            val = np.random.poisson(p["lambda"])
                        elif family == "gamma":
                            val = random.gammavariate(p["shape"], p["scale"])
                        elif family == "beta":
                            val = random.betavariate(p["alpha"], p["beta"])
                        elif family == "lognormal":
                            val = random.lognormvariate(p["mean"], p["sigma"])
                        else:
                            raise ValueError(f"Unknown family {family}")

                        if family != "categorical" and truncation:
                            if not (truncation["min"] <= val <= truncation["max"]):
                                continue

                        if output_type == "int":
                            val = int(val)
                        elif output_type == "float":
                            val = round(float(val), 4)
                        else:
                            val = str(val)

                        if self.metrics_collector:
                            self.metrics_collector.record_distribution_sample(dist_name, family)

                        return val

                    raise ValueError(
                        f"Truncation too restrictive for {dist_name}"
                    )

                return sampler

            self.samplers[name] = {
                'sampler': make_sampler(name, family, output_type, params, labels, truncation),
                'family': family
            }

    def sample(self, name, bound_params=None):
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")

        return entry['sampler'](bound_params)