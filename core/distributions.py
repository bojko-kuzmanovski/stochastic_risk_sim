import os
import sys
import math
import json
import random
import numpy as np
from jsonschema import validate
from scipy import stats

class Distributions:
    def __init__(self, config_data, metrics_collector, seed=None):
        # Generadores propios sembrados: la semilla registrada determina la trayectoria.
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        schema_path = "schemas/distributions.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)

        for d in config_data:
            name = d["distribution_name"]
            family = d["family"]
            params = d.get("params", {})
            truncation = d.get("truncation")

            if family == "categorical":
                categories = params.get("categories", [])
                total_prob = sum(c.get("probability", 0.0) for c in categories)
                if not math.isclose(total_prob, 1.0, rel_tol=1e-9):
                    print(f"[FATAL] DistributionsFactory::VALIDATION: "
                          f"no threshold_case matched for value Categorical distribution '{name}' "
                          f"probabilities sum up to {total_prob} instead of 1.0", file=sys.stderr)
                    os._exit(1)
            else:
                if truncation:
                    if truncation["min"] >= truncation["max"]:
                        print(f"[FATAL] DistributionsFactory::VALIDATION: "
                              f"no threshold_case matched for value Distribution '{name}' truncation bounds: "
                              f"min ({truncation['min']}) must be strictly less than max ({truncation['max']})", file=sys.stderr)
                        os._exit(1)

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

            # Precalculamos los denominadores de truncamiento analíticos para ahorrar CPU
            trunc_factor = 1.0
            if truncation and family != "categorical":
                t_min = truncation["min"]
                t_max = truncation["max"]
                try:
                    if family == "normal":
                        f_max = stats.norm.cdf(t_max, loc=params["mean"], scale=params["sigma"])
                        f_min = stats.norm.cdf(t_min, loc=params["mean"], scale=params["sigma"])
                    elif family == "exponential":
                        f_max = stats.expon.cdf(t_max, scale=1.0 / params["rate"])
                        f_min = stats.expon.cdf(t_min, scale=1.0 / params["rate"])
                    elif family == "gamma":
                        f_max = stats.gamma.cdf(t_max, a=params["shape"], scale=params["scale"])
                        f_min = stats.gamma.cdf(t_min, a=params["shape"], scale=params["scale"])
                    elif family == "beta":
                        f_max = stats.beta.cdf(t_max, a=params["alpha"], b=params["beta"])
                        f_min = stats.beta.cdf(t_min, a=params["alpha"], b=params["beta"])
                    elif family == "lognormal":
                        f_max = stats.lognorm.cdf(t_max, s=params["sigma"], scale=np.exp(params["mean"])) if t_max > 0 else 0.0
                        f_min = stats.lognorm.cdf(t_min, s=params["sigma"], scale=np.exp(params["mean"])) if t_min > 0 else 0.0
                    elif family == "poisson":
                        f_max = stats.poisson.cdf(int(np.floor(t_max)), mu=params["lambda"])
                        f_min = stats.poisson.cdf(int(np.ceil(t_min)) - 1, mu=params["lambda"]) if t_min > 0 else 0.0
                    else:
                        f_max, f_min = 1.0, 0.0
                    
                    trunc_factor = max(1e-12, f_max - f_min)
                except Exception:
                    trunc_factor = 1.0

            def make_sampler(dist_name, f_fam, out_t, p_maps, lbls, probs, trnc):
                def sampler():
                    p = {**p_maps}
                    max_attempts = 1000

                    for _ in range(max_attempts):
                        if f_fam == "categorical":
                            val = self.rng.choices(lbls, weights=probs)[0]
                        elif f_fam == "normal":
                            val = self.rng.normalvariate(p["mean"], p["sigma"])
                        elif f_fam == "exponential":
                            val = self.rng.expovariate(p["rate"])
                        elif f_fam == "poisson":
                            val = int(self.np_rng.poisson(p["lambda"]))
                        elif f_fam == "gamma":
                            val = self.rng.gammavariate(p["shape"], p["scale"])
                        elif f_fam == "beta":
                            val = self.rng.betavariate(p["alpha"], p["beta"])
                        elif f_fam == "lognormal":
                            val = self.rng.lognormvariate(p["mean"], p["sigma"])
                        else:
                            raise ValueError(f"Unknown family {f_fam}")

                        if f_fam != "categorical" and trnc:
                            if not (trnc["min"] <= val <= trnc["max"]):
                                continue

                        if out_t == "int":
                            val = int(val)
                        elif out_t == "float":
                            val = round(float(val), 4)
                        else:
                            val = str(val)

                        if self.metrics_collector:
                            self.metrics_collector.record_distribution_sample(dist_name, f_fam)

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
                'output_type': output_type,
                'trunc_factor': trunc_factor
            }

    def sample(self, name):
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")
        return entry['sampler']()

    def probability_interval(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True):
        # Cálculo determinista: se memoriza por (distribución, intervalo).
        key = ("p", name, float(lower), float(upper), bool(lower_inclusive), bool(upper_inclusive))
        cache = self.__dict__.setdefault("_exact_cache", {})
        if key not in cache:
            cache[key] = self._probability_interval(name, lower, upper, lower_inclusive, upper_inclusive)
        return cache[key]

    def _probability_interval(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True):
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")

        family = entry['family']
        params = entry['params']
        truncation = entry['truncation']
        trunc_factor = entry['trunc_factor']

        if lower == float("-inf"): lower = -1e308
        if upper == float("inf"): upper = 1e308

        # Forzar límites al espacio truncado real si existe
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
                
                # Evaluación estricta de límites discretos/categóricos
                if lower_inclusive and upper_inclusive:
                    if lower <= val <= upper: total += p
                elif lower_inclusive:
                    if lower <= val < upper: total += p
                elif upper_inclusive:
                    if lower < val <= upper: total += p
                else:
                    if lower < val < upper: total += p
            return total

        if family == "poisson":
            lam = params["lambda"]
            k_min = int(np.ceil(lower)) if lower_inclusive else int(np.floor(lower)) + 1
            k_max = int(np.floor(upper)) if upper_inclusive else int(np.ceil(upper)) - 1
            
            if k_min > k_max:
                return 0.0
                
            cdf_upper = stats.poisson.cdf(k_max, mu=lam)
            cdf_lower = stats.poisson.cdf(k_min - 1, mu=lam) if k_min > 0 else 0.0
            
            raw_prob = max(0.0, cdf_upper - cdf_lower)
            return max(0.0, min(1.0, raw_prob / trunc_factor))

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
                scale_val = np.exp(params["mean"])
                cdf_lower = stats.lognorm.cdf(lower, s=params["sigma"], scale=scale_val) if lower > 0 else 0.0
                cdf_upper = stats.lognorm.cdf(upper, s=params["sigma"], scale=scale_val) if upper > 0 else 0.0
            else:
                raise ValueError(f"CDF not supported for {family}")
        except Exception as e:
            print(f"[WARN] error analítico en {family}: {e}")
            return 0.0

        # Aplicación del Teorema de Probabilidad Condicional para Truncamiento Formal
        result = max(0.0, (cdf_upper - cdf_lower) / trunc_factor)
        return min(1.0, result)

    def probability_label(self, name, label):
        """Probabilidad exacta de una etiqueta en una distribución categórica."""
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")
        if entry['family'] != "categorical":
            raise ValueError(f"Distribution {name} is not categorical")
        return sum(p for l, p in zip(entry['labels'], entry['probabilities']) if l == label)

    def quantile_cells(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True, cells=8):
        """
        Divide la región [lower, upper] (con el truncamiento aplicado) en celdas de igual masa y devuelve
        [(masa, esperanza condicional en la celda)]. La suma de masas es probability_interval de la región.
        Solo para familias continuas; se memoriza por (distribución, región, número de celdas).
        """
        key = ("q", name, float(lower), float(upper), bool(lower_inclusive), bool(upper_inclusive), int(cells))
        cache = self.__dict__.setdefault("_exact_cache", {})
        if key in cache:
            return cache[key]
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")
        family, params, truncation = entry['family'], entry['params'], entry['truncation']
        dist = self._frozen(family, params)
        lo = -np.inf if lower == float("-inf") else lower
        hi = np.inf if upper == float("inf") else upper
        if truncation:
            lo = max(lo, truncation["min"])
            hi = min(hi, truncation["max"])
        total = self.probability_interval(name, lower, upper, lower_inclusive, upper_inclusive)
        if lo >= hi or total <= 0:
            cache[key] = []
            return []
        f_lo, f_hi = dist.cdf(lo), dist.cdf(hi)
        cuts = [lo] + [float(dist.ppf(f_lo + (f_hi - f_lo) * j / cells)) for j in range(1, cells)] + [hi]
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            mass = self.probability_interval(name, a, b, True, True)
            if mass > 0:
                out.append((mass, self.conditional_mean(name, a, b, True, True)))
        scale = total / sum(m for m, _ in out) if out else 1.0
        out = [(m * scale, v) for m, v in out]
        cache[key] = out
        return out

    @staticmethod
    def _frozen(family, params):
        if family == "normal":
            return stats.norm(loc=params["mean"], scale=params["sigma"])
        if family == "exponential":
            return stats.expon(scale=1.0 / params["rate"])
        if family == "gamma":
            return stats.gamma(a=params["shape"], scale=params["scale"])
        if family == "beta":
            return stats.beta(a=params["alpha"], b=params["beta"])
        if family == "lognormal":
            return stats.lognorm(s=params["sigma"], scale=np.exp(params["mean"]))
        raise ValueError(f"continuous distribution required, got {family}")

    def conditional_mean(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True):
        # Integración numérica costosa y determinista: se memoriza por (distribución, intervalo).
        key = ("m", name, float(lower), float(upper), bool(lower_inclusive), bool(upper_inclusive))
        cache = self.__dict__.setdefault("_exact_cache", {})
        if key not in cache:
            cache[key] = self._conditional_mean(name, lower, upper, lower_inclusive, upper_inclusive)
        return cache[key]

    def _conditional_mean(self, name, lower, upper, lower_inclusive=True, upper_inclusive=True):
        """
        E[X | X en el intervalo], respetando el truncamiento de la distribución.
        Es el representante determinista del valor muestreado dentro de la región
        de un caso de decisión; no consume el generador.
        """
        entry = self.samplers.get(name)
        if not entry:
            raise ValueError(f"Distribution {name} not found")
        family, params, truncation = entry['family'], entry['params'], entry['truncation']

        lo = -np.inf if lower == float("-inf") else lower
        hi = np.inf if upper == float("inf") else upper
        if truncation:
            lo = max(lo, truncation["min"])
            hi = min(hi, truncation["max"])
        if lo > hi:
            return None

        if family == "poisson":
            k_min = int(np.ceil(lo)) if (lower_inclusive or lo != lower) else int(np.floor(lo)) + 1
            k_max = int(np.floor(hi)) if (upper_inclusive or hi != upper) else int(np.ceil(hi)) - 1
            k_min = max(k_min, 0)
            if k_min > k_max:
                return None
            ks = np.arange(k_min, k_max + 1)
            w = stats.poisson.pmf(ks, mu=params["lambda"])
            return float((ks * w).sum() / w.sum()) if w.sum() > 0 else float(k_min)

        if family == "normal":
            dist = stats.norm(loc=params["mean"], scale=params["sigma"])
        elif family == "exponential":
            dist = stats.expon(scale=1.0 / params["rate"])
        elif family == "gamma":
            dist = stats.gamma(a=params["shape"], scale=params["scale"])
        elif family == "beta":
            dist = stats.beta(a=params["alpha"], b=params["beta"])
        elif family == "lognormal":
            dist = stats.lognorm(s=params["sigma"], scale=np.exp(params["mean"]))
        else:
            raise ValueError(f"Conditional mean not supported for {family}")

        mass = dist.cdf(hi) - dist.cdf(lo)
        if mass <= 0:
            return float(lo if np.isfinite(lo) else hi)
        return float(dist.expect(lambda x: x, lb=lo, ub=hi, conditional=True))