"""
Escenario de juguete de otro dominio (mantenimiento de máquinas), definido solo como JSON.

Sirve para comprobar que el motor funciona con cualquier configuración: nada de este dominio existe
en el código fuente del motor. Con with_channel=True, una falla de Machine_1 publica en un canal y el
técnico recibe un evento dinámico con la latencia del canal.
"""
import json
from pathlib import Path

FAIL_THRESHOLD = 0.3
CHANNEL_LATENCY = 0.5


def det(v): return {"type": "deterministic", "value": v}
def prob(d): return {"type": "probabilistic", "distribution": d}


def scenario(with_channel=False):
    distributions = [
        {"distribution_name": "wear_draw", "family": "beta", "params": {"alpha": 2, "beta": 3},
         "output_type": "float", "truncation": {"min": 0.0, "max": 1.0}},
    ]
    plant = {"quantity": 1, "environment_type": "Plant", "params": {"shift_hours": det(8)},
             "members": [{"agent_id": "Machine_1", "agent_rol": {"line": det("A")}},
                         {"agent_id": "Machine_2", "agent_rol": {"line": det("B")}},
                         {"agent_id": "Technician_1", "agent_rol": {"senior": det(True)}}]}
    if with_channel:
        plant["channels"] = [{"channel_id": "maintenance", "members": ["Machine_1", "Technician_1"], "events": [],
                              "metadata": {"latency": det(CHANNEL_LATENCY)}}]
    agents = [
        {"quantity": 2, "agent_type": "Machine", "params": {"broken": det(False)}, "automata": ["wear_check"]},
        {"quantity": 1, "agent_type": "Technician", "params": {"repairs": det(0)}, "automata": ["repair"]},
    ]

    failed_actions = [
        {"update_params": {"$agent_id": det("$machine_id"), "$param_key": det("broken"), "$param_value": det(True)}},
        {"action_required": {"marcar": {"target": "agents", "method": "write_agent_param",
                                        "params": ["$agent_id", "$param_key", "$param_value"]}}},
    ]
    if with_channel:
        failed_actions += [
            {"update_params": {"$env_id": det("Plant_1"), "$channel_id": det("maintenance"), "$signal": det("repair")}},
            {"action_required": {"avisar": {"target": "environments", "method": "write_ch_event",
                                            "params": ["$env_id", "$channel_id", "$agent_id", "$signal"]}}},
        ]

    automata = [
        {"automaton_name": "wear_check", "states": {"initial": "LOAD", "final": ["FAILED", "OK"]}, "params": {},
         "transitions": [
             {"from": "LOAD", "threshold_value": {"$go": det(True)},
              "thresholds": [{"threshold_case": [{"variable": "$go", "operator": "==", "value": True}], "to": "CHECK",
                              "actions": [{"update_params": {"$machine_id": {"target": "events", "method": "event_agent_id", "params": []}}}]}]},
             {"from": "CHECK", "threshold_value": {"$wear": prob("wear_draw")},
              "thresholds": [
                  {"threshold_case": [{"variable": "$wear", "operator": "<", "value": FAIL_THRESHOLD}], "to": "FAILED",
                   "actions": failed_actions},
                  {"threshold_case": [{"variable": "$wear", "operator": ">=", "value": FAIL_THRESHOLD}], "to": "OK", "actions": []}]},
         ]},
        {"automaton_name": "repair", "states": {"initial": "LOAD", "final": ["FIXED", "DEFERRED"]}, "params": {},
         "transitions": [
             {"from": "LOAD", "threshold_value": {"$go": det(True)},
              "thresholds": [{"threshold_case": [{"variable": "$go", "operator": "==", "value": True}], "to": "FIX", "actions": []}]},
             {"from": "FIX", "threshold_value": {"$skill": prob("wear_draw")},
              "thresholds": [
                  {"threshold_case": [{"variable": "$skill", "operator": "<", "value": 0.6}], "to": "FIXED", "actions": []},
                  {"threshold_case": [{"variable": "$skill", "operator": ">=", "value": 0.6}], "to": "DEFERRED", "actions": []}]},
         ]},
    ]
    events = [
        {"event_category": "static", "signal": "wear_check", "agent_type": "Machine", "periodicity": det(1.0)},
        {"event_category": "static", "signal": "repair", "agent_type": "Technician", "periodicity": det(2.0)},
    ]
    patl = {"observations": [
        {"automaton_name": "wear_check", "trigger_state": "CHECK", "predicates": [
            {"predicate_id": "MACHINE_FAILS", "type": "reachability", "max_depth": 2, "coalition_quantifier": "exists",
             "coalition": [{"agent_type": "Machine", "max_agents": 1,
                            "automata": [{"automaton_name": "wear_check", "target_states": ["FAILED"]}]}],
             "adversaries": [{"agent_type": "Technician", "max_agents": 1, "automata": [{"automaton_name": "repair"}]}],
             "probability_bound": 0.5, "probability_operator": ">="}]},
    ]}
    return {"distributions": distributions, "environments": [plant], "automata": automata,
            "agents": agents, "events": events, "patl": patl}


def write(directory: Path, with_channel=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in scenario(with_channel).items():
        (directory / f"{name}.json").write_text(json.dumps(content, indent=2))
    return directory
