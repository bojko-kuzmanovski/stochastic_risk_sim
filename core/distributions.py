import json
import random
import numpy as np
from jsonschema import validate
from scipy import stats

class Distributions:
    def __init__(self, config_data, metrics_collector):
        schema_path = "schemas/distributions.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)

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
            truncation = d.get("truncation")

            if family == "categorical":
                categories = params.get("categories", [])
                labels = [c["label"] for c in categories]
                probabilities = [c["probability"] for c in categories]
            else:
                labels = []
                probabilities = []

            def make_sampler(dist_name, family, output_type, params, labels, probabilities, truncation):
                def sampler():
                    p = {**params}
                    max_attempts = 1000

                    for _ in range(max_attempts):
                        if family == "categorical":
                            val = random.choices(labels, weights=probabilities)[0]
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

                    raise ValueError(f"Truncation too restrictive for {dist_name}")

                return sampler

            self.samplers[name] = {
                'sampler': make_sampler(name, family, output_type, params, labels, probabilities, truncation),
                'family': family,
                'params': params,
                'labels': labels,
                'probabilities': probabilities,
                'truncation': truncation,
                'output_type': output_type
            }

    def sample(self, name):
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")
        return entry['sampler']()

    def probability_interval(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True):
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")

        family = entry['family']
        params = entry['params']
        truncation = entry['truncation']

        if lower == float("-inf"):
            lower = -1e308
        if upper == float("inf"):
            upper = 1e308

        if truncation:
            lower = max(lower, truncation["min"])
            upper = min(upper, truncation["max"])
        if lower > upper:
            return 0.0

        if family == "categorical":
            labels = entry['labels']
            probs = entry['probabilities']
            total = 0.0
            for label, p in zip(labels, probs):
                try:
                    val = float(label)
                except (ValueError, TypeError):
                    continue
                if lower_inclusive and upper_inclusive:
                    if lower <= val <= upper:
                        total += p
                elif lower_inclusive:
                    if lower <= val < upper:
                        total += p
                elif upper_inclusive:
                    if lower < val <= upper:
                        total += p
                else:
                    if lower < val < upper:
                        total += p
            return total

        if family == "poisson":
            lam = params["lambda"]
            actual_lower = max(0, lower)
            actual_upper = min(upper, 1e6)
            k_min = int(np.ceil(actual_lower)) if lower_inclusive else int(np.floor(actual_lower)) + 1
            k_max = int(np.floor(actual_upper)) if upper_inclusive else int(np.ceil(actual_upper)) - 1
            if k_min > k_max:
                return 0.0
            if k_min <= 0:
                return stats.poisson.cdf(k_max, mu=lam)
            return stats.poisson.cdf(k_max, mu=lam) - stats.poisson.cdf(k_min - 1, mu=lam)

        try:
            if family == "normal":
                cdf_lower = stats.norm.cdf(lower, loc=params["mean"], scale=params["sigma"])
                cdf_upper = stats.norm.cdf(upper, loc=params["mean"], scale=params["sigma"])
            elif family == "exponential":
                cdf_lower = stats.expon.cdf(lower, scale=1.0 / params["rate"])
                cdf_upper = stats.expon.cdf(upper, scale=1.0 / params["rate"])
            elif family == "gamma":
                cdf_lower = stats.gamma.cdf(lower, a=params["shape"], scale=params["scale"])
                cdf_upper = stats.gamma.cdf(upper, a=params["shape"], scale=params["scale"])
            elif family == "beta":
                cdf_lower = stats.beta.cdf(lower, a=params["alpha"], b=params["beta"])
                cdf_upper = stats.beta.cdf(upper, a=params["alpha"], b=params["beta"])
            elif family == "lognormal":
                sigma = params["sigma"]
                scale = np.exp(params["mean"]) if params["mean"] > -100 else 1e-10
                cdf_lower = stats.lognorm.cdf(max(0, lower), s=sigma, scale=scale)
                cdf_upper = stats.lognorm.cdf(max(0, upper), s=sigma, scale=scale)
            else:
                raise ValueError(f"CDF not supported for {family}")
        except Exception as e:
            print(f"[WARN] probability_interval error for {family}: {e}, params={params}, lower={lower}, upper={upper}")
            return 0.0

        result = max(0.0, min(1.0, cdf_upper - cdf_lower))
        return result