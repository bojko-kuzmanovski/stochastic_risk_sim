"""
Utilidades para construir juegos pequeños, válidos contra los esquemas, cuyo valor PATL se calcula a mano.
"""
from core.automata import Automata
from core.distributions import Distributions
from metrics.metrics_collector import MetricsCollector
from patl.patl_verifier import PATLVerifier


def det(v): return {"type": "deterministic", "value": v}
def prob(d): return {"type": "probabilistic", "distribution": d}
def c(var, op, val): return {"variable": var, "operator": op, "value": val}
def upd(**kv): return {"update_params": {f"${k}": (v if isinstance(v, dict) else det(v)) for k, v in kv.items()}}
def case(conds, to, *actions):
    flat = []
    for a in actions:
        flat.extend(a if isinstance(a, list) else [a])
    return {"threshold_case": conds if isinstance(conds, list) else [conds], "to": to, "actions": flat}
def T(frm, var, tv, *cases): return {"from": frm, "threshold_value": {var: tv}, "thresholds": list(cases)}
def load(to, *actions):
    return T("LOAD", "$always_load", det(True), case(c("$always_load", "==", True), to, *actions))
def read_agent():
    return {"target": "agents", "method": "read_agent_param", "params": ["$agent_id", "$param_key"]}
def write(agent, key, value):
    return [upd(agent_id=agent, param_key=key, param_value=value),
            {"action_required": {"escritura": {"target": "agents", "method": "write_agent_param",
                                               "params": ["$agent_id", "$param_key", "$param_value"]}}}]
def automaton(name, finals, *transitions):
    return {"automaton_name": name, "states": {"initial": "LOAD", "final": finals}, "params": {}, "transitions": list(transitions)}


UNIFORM = {"distribution_name": "u", "family": "beta", "params": {"alpha": 1, "beta": 1},
           "output_type": "float", "truncation": {"min": 0.0, "max": 1.0}}
COIN = {"distribution_name": "coin", "family": "categorical", "output_type": "string",
        "params": {"categories": [{"label": "L", "probability": 0.5}, {"label": "R", "probability": 0.5}]}}


def gate(name, threshold, success="SUCCESS", failure="FAIL"):
    """Una sesión con una sola compuerta uniforme: P(success) = threshold."""
    return automaton(name, [success, failure],
                     load("GATE"),
                     T("GATE", "$x", prob("u"),
                       case(c("$x", "<", threshold), success),
                       case(c("$x", ">=", threshold), failure)))


def agent(agent_id, automata, **params):
    return {"agent_id": agent_id, "agent_type": agent_id.rsplit("_", 1)[0], "automata": list(automata), "params": params}


def snapshot(*agents, trigger="Observer_1"):
    agents = list(agents)
    if trigger == "Observer_1" and not any(a["agent_id"] == trigger for a in agents):
        agents.append(agent("Observer_1", []))
    return {"agent_id": trigger, "automaton_name": "none", "state": "none",
            "agents_data": agents, "environments_data": []}


def group(agent_id, *automata_targets):
    return [{"agent_id": [agent_id], "max_agents": 1,
             "automata": [({"automaton_name": a, "target_states": t} if t is not None else {"automaton_name": a})
                          for a, t in automata_targets]}]


def predicate(coalition, adversaries=None, ptype="reachability", bound=0.5, op=">=", depth=1, quantifier="exists"):
    p = {"predicate_id": "P", "type": ptype, "max_depth": depth, "coalition_quantifier": quantifier,
         "coalition": coalition, "probability_bound": bound, "probability_operator": op}
    if adversaries:
        p["adversaries"] = adversaries
    return p


def verifier(automata_defs, distributions=(UNIFORM,), memory=(1,)):
    dists = Distributions(list(distributions), None, seed=0)
    auts = Automata(list(automata_defs), dists, MetricsCollector(enabled=False))
    return PATLVerifier(auts, dists, configs={"agents": [], "environments": []}, default_memory=list(memory)), dists


def value(automata_defs, snap, pred, memory=1, distributions=(UNIFORM,)):
    v, _ = verifier(automata_defs, distributions, [memory])
    row = v.verify(snap, [pred])[0]
    return row
