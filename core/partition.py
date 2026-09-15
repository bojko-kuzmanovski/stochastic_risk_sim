"""
Regiones efectivas de los casos de una transición probabilística bajo la regla del primer caso que coincide.

Funciones puras. Aplican a una transición cuyo valor de umbral es probabilístico y cuyos casos son
condiciones sobre esa única variable con valores literales (sin referencias "$..." de tiempo de ejecución).

* Continuas: cada caso define un intervalo con extremos abiertos o cerrados; su región efectiva es el
  intervalo menos las regiones de los casos anteriores. Los puntos aislados tienen medida cero.
* Categóricas: cada etiqueta, convertida como la entrega el muestreador, cae en el primer caso que la acepta.
* Poisson: cada valor del soporte dentro del truncamiento cae en el primer caso que lo acepta.

uncovered_mass devuelve la masa que ningún caso cubre; check_automata rechaza las transiciones que dejan
masa sin cubrir. La lógica de intervalos es la de PATLVerifier._interval y PATLVerifier._minus.
"""

import operator

import numpy as np
from scipy import stats

UNCOVERED_TOLERANCE = 1e-9

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
        "==": operator.eq, "!=": operator.ne}


class NotApplicable(Exception):
    """La transición no tiene la forma que este módulo analiza."""


def interval(tv_key, conditions):
    """Región de un caso sobre la variable continua; None si tiene medida cero."""
    name = tv_key.lstrip("$")
    low, high = float("-inf"), float("inf")
    li = ui = True
    for c in conditions:
        if not isinstance(c["variable"], str) or c["variable"].lstrip("$") != name:
            raise NotApplicable(f"the case conditions a variable other than {tv_key}")
        op, v = c["operator"], c["value"]
        if v is None or isinstance(v, (str, bool)):
            if op == "==":
                return None
            continue
        if op == ">":
            if v >= low: low, li = v, False
        elif op == ">=":
            if v > low: low, li = v, True
        elif op == "<":
            if v <= high: high, ui = v, False
        elif op == "<=":
            if v < high: high, ui = v, True
        elif op == "==":
            return None
    if low > high:
        return None
    return low, high, li, ui


def minus(a, r):
    """a menos r, con extremos abiertos o cerrados; descarta puntos aislados (medida cero)."""
    alo, ahi, ali, aui = a
    rlo, rhi, rli, rui = r
    disjoint = (rhi < alo or (rhi == alo and not (rui and ali)) or
                rlo > ahi or (rlo == ahi and not (rli and aui)))
    if disjoint:
        return [a]
    pieces = []
    if rlo > alo or (rlo == alo and ali and not rli):
        pieces.append((alo, rlo, ali, not rli))
    if rhi < ahi or (rhi == ahi and aui and not rui):
        pieces.append((rhi, ahi, not rui, aui))
    return [p for p in pieces if p[0] < p[1]]


def effective_regions(tv_key, thresholds):
    """Para cada caso, en orden, la lista de intervalos de su región efectiva."""
    covered, regions = [], []
    for case in thresholds:
        iv = interval(tv_key, case["threshold_case"])
        if iv is None:
            regions.append([])
            continue
        pieces = [iv]
        for previous in covered:
            pieces = [p for piece in pieces for p in minus(piece, previous)]
        covered.append(iv)
        regions.append(pieces)
    return regions


def case_accepts(value, conditions):
    """El valor cumple todas las condiciones del caso; una comparación entre tipos incompatibles no lo acepta."""
    for c in conditions:
        fn = _OPS.get(c["operator"])
        if fn is None:
            return False
        try:
            if not fn(value, c["value"]):
                return False
        except TypeError:
            return False
    return True


def first_case(value, thresholds):
    """Índice del primer caso que acepta el valor, o None."""
    for i, case in enumerate(thresholds):
        if case_accepts(value, case["threshold_case"]):
            return i
    return None


def categorical_support(entry):
    """Etiquetas con probabilidad positiva, convertidas como las entrega el muestreador."""
    out = []
    for label, p in zip(entry["labels"], entry["probabilities"]):
        if p <= 0:
            continue
        if entry["output_type"] == "int":
            value = int(label)
        elif entry["output_type"] == "float":
            value = float(label)
        else:
            value = str(label)
        out.append((value, p))
    return out


def poisson_support(entry):
    """Soporte de Poisson dentro del truncamiento, con masas normalizadas."""
    lam = entry["params"]["lambda"]
    trunc = entry["truncation"]
    k_min = int(np.ceil(trunc["min"])) if trunc else 0
    k_max = int(np.floor(trunc["max"])) if trunc else int(stats.poisson.ppf(1 - 1e-12, mu=lam))
    ks = np.arange(max(k_min, 0), k_max + 1)
    pmf = stats.poisson.pmf(ks, mu=lam)
    pmf = pmf / pmf.sum()
    return [(float(k), float(p)) for k, p in zip(ks, pmf)]


def applicable(transition, distributions):
    """
    (variable, nombre de distribución) si la transición tiene la forma analizable; si no, NotApplicable.
    Se omiten umbrales deterministas o dinámicos, distribuciones no declaradas, valores de caso con
    referencias "$..." y casos que condicionan otra variable.
    """
    tv = transition.get("threshold_value") or {}
    if len(tv) != 1:
        raise NotApplicable("threshold_value must have a single variable")
    tv_key, tv_def = next(iter(tv.items()))
    if not (isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic"):
        raise NotApplicable("threshold is not probabilistic")
    name = tv_def.get("distribution")
    if distributions is None or name not in getattr(distributions, "samplers", {}):
        raise NotApplicable(f"distribution '{name}' is not declared")
    var = tv_key.lstrip("$")
    for case in transition.get("thresholds", []):
        for c in case["threshold_case"]:
            if not isinstance(c["variable"], str) or c["variable"].lstrip("$") != var:
                raise NotApplicable("a case conditions another variable")
            if isinstance(c["value"], str) and "$" in c["value"]:
                raise NotApplicable("a case value references a runtime variable")
    return tv_key, name


def uncovered_mass(transition, distributions):
    """Masa de probabilidad del valor de umbral que ningún caso cubre; None si la transición no aplica."""
    try:
        tv_key, name = applicable(transition, distributions)
    except NotApplicable:
        return None
    entry = distributions.samplers[name]
    family = entry["family"]
    thresholds = transition.get("thresholds", [])

    if family in ("categorical", "poisson"):
        support = categorical_support(entry) if family == "categorical" else poisson_support(entry)
        total = sum(p for _, p in support)
        covered = sum(p for v, p in support if first_case(v, thresholds) is not None)
        return max(0.0, (total - covered) / total) if total > 0 else 0.0

    covered = 0.0
    for pieces in effective_regions(tv_key, thresholds):
        for low, high, li, ui in pieces:
            covered += distributions.probability_interval(name, low, high, li, ui)
    return max(0.0, 1.0 - covered)


def check_automata(config_data, distributions, tolerance=UNCOVERED_TOLERANCE):
    """Rechaza toda transición probabilística analizable que deja masa sin cubrir por sus casos."""
    problems = []
    for aut in config_data:
        for t in aut.get("transitions", []):
            mass = uncovered_mass(t, distributions)
            if mass is not None and mass > tolerance:
                var, tv_def = next(iter(t["threshold_value"].items()))
                problems.append(f"{aut['automaton_name']}::{t['from']} ({var} ~ {tv_def.get('distribution')}): "
                                f"uncovered probability mass {mass:.3g}")
    if problems:
        raise ValueError("probabilistic transitions whose cases do not cover the support: " + "; ".join(problems))
