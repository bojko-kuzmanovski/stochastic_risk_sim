"""
Regresiones de la auditoría del motor DES frente al marco formal: identificadores, eliminación de agentes,
muestreo, errores explícitos, valor vacío, eventos estáticos y dinámicos, instantáneas, calendario,
relaciones dirigidas, orden de procesamiento, cobertura de casos y parámetros categóricos en el verificador.
"""
import json
import os
from pathlib import Path

import pytest
from scipy import stats

import main
import toy_scenario
from toy_scenario import det, prob
from core import partition
from core.agents import Agents
from core.automata import Automata, AutomatonSession, SessionError
from core.distributions import Distributions
from core.environments import Environments
from core.events import Events
from core.trace import tracer
from core.utils import evaluator
from core.utils.evaluator import EvaluationError
from des.discrete_event_engine import DiscreteEventSimulator
from des.event_scheduler import EventCalendar, EventScheduler
from metrics.metrics_collector import MetricsCollector
from patl.snapshot_manager import SnapshotManager

ROOT = Path(__file__).resolve().parents[1]
OFF = MetricsCollector(enabled=False)

UNIFORM = {"distribution_name": "u", "family": "beta", "params": {"alpha": 1, "beta": 1},
           "output_type": "float", "truncation": {"min": 0.0, "max": 1.0}}
COIN = {"distribution_name": "coin", "family": "categorical", "output_type": "string",
        "params": {"categories": [{"label": "L", "probability": 0.5}, {"label": "R", "probability": 0.5}]}}
ARRIVALS = {"distribution_name": "arrivals", "family": "poisson", "params": {"lambda": 1.0},
            "output_type": "float", "truncation": {"min": 0.0, "max": 3.0}}
GAP = {"distribution_name": "gap", "family": "exponential", "params": {"rate": 1.0},
       "output_type": "float", "truncation": {"min": 0.5, "max": 5.0}}


# ------------------------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------------------------
def c(var, op, val): return {"variable": var, "operator": op, "value": val}
def case(conds, to, *actions): return {"threshold_case": conds if isinstance(conds, list) else [conds], "to": to, "actions": list(actions)}
def T(frm, var, tv, *cases): return {"from": frm, "threshold_value": {var: tv}, "thresholds": list(cases)}
def go(frm, to, *actions): return T(frm, f"$go_{frm.lower()}", det(True), case(c(f"$go_{frm.lower()}", "==", True), to, *actions))
def automaton(name, finals, *transitions, params=None):
    return {"automaton_name": name, "states": {"initial": "LOAD", "final": finals}, "params": params or {},
            "transitions": list(transitions)}
def call(target, method, *params): return {"target": target, "method": method, "params": list(params)}
def pipe(initial, *ops):
    return {"target": "system", "method": "math_pipeline",
            "params": {"initial_value": det(initial), "operations": [{"operator": o, "with": det(w)} for o, w in ops]}}


class _Recorder:
    """Sustituto del gestor de instantáneas que registra cada captura."""
    def __init__(self, agents=None, envs=None):
        self.agents, self.envs, self.captures = agents, envs, []

    def capture(self, agent_id, automaton_name, state, trigger_event=None):
        members = self.envs.read_members("Plant_1") if self.envs else []
        self.captures.append({"agent": agent_id, "state": state, "event": trigger_event,
                              "present": self.agents._find(agent_id) is not None,
                              "member": agent_id in members})


def scenario(extra_automata=(), machine_automata=(), extra_distributions=(), with_channel=True):
    scn = toy_scenario.scenario(with_channel=with_channel)
    scn["automata"].extend(extra_automata)
    scn["agents"][0]["automata"].extend(machine_automata)
    scn["distributions"].extend(extra_distributions)
    return scn


def world(scn, seed=1):
    dists = Distributions(scn["distributions"], OFF, seed=seed)
    envs = Environments(scn["environments"], dists, OFF)
    auts = Automata(scn["automata"], dists, OFF)
    agents = Agents(scn["agents"], dists, OFF)
    auts.set_objects(agents, envs)
    recorder = _Recorder(agents, envs)
    agents.set_objects(auts, recorder)
    return dists, envs, auts, agents, recorder


class _Exit(Exception):
    pass


@pytest.fixture
def fatal(monkeypatch):
    """Convierte os._exit en una excepción para observar los errores fatales de la ejecución en vivo."""
    def fake(code):
        raise _Exit(code)
    monkeypatch.setattr(os, "_exit", fake)
    return _Exit


def session(auts, name, agent_id="Machine_1", live=False):
    return AutomatonSession(auts, auts.by_name[name], {"signal": name, "agent_id": agent_id}, live=live)


def run_to_end(s):
    while s.step() is not None:
        pass
    return s.current_state


# ------------------------------------------------------------------------------------------------
# 1. Identificadores no reutilizados y purga del agente eliminado
# ------------------------------------------------------------------------------------------------
def test_1_identificadores_nunca_se_reutilizan():
    _, _, _, agents, _ = world(scenario())
    assert agents.remove_agent("Machine_2")
    assert agents.add_agent("Machine") == "Machine_3"
    assert agents.remove_agent("Machine_3")
    assert agents.add_agent("Machine") == "Machine_4"
    assert agents.remove_agent("Machine_1")
    assert agents.add_agent("Machine") == "Machine_5"
    assert agents.get_all_agents("Machine") == ["Machine_4", "Machine_5"]


def test_1_eliminar_agente_lo_purga_de_entornos_relaciones_y_canales():
    _, envs, _, agents, _ = world(scenario())
    assert envs.add_rel("Plant_1", "Machine_1", "Technician_1")
    assert envs.add_rel("Plant_1", "Technician_1", "Machine_1")
    assert envs.add_rel("Plant_1", "Machine_2", "Technician_1")
    assert agents.remove_agent("Machine_1")
    plant = envs.data[0]
    assert "Machine_1" not in [m["agent_id"] for m in plant["members"]]
    assert [r["members"] for r in plant["relations"]] == [["Machine_2", "Technician_1"]]
    assert "Machine_1" not in plant["channels"][0]["members"]
    assert envs.read_member_param("Plant_1", "Machine_1", "line") is None
    assert agents.remove_agent("Machine_1") is False


def test_1_eventos_de_un_agente_eliminado_se_descartan_y_no_se_reprograman(tmp_path):
    scn = toy_scenario.scenario()
    repair = next(a for a in scn["automata"] if a["automaton_name"] == "repair")
    repair["transitions"][0]["thresholds"][0]["actions"] = [
        {"update_params": {"$agent_id": det("Machine_2")}},
        {"action_required": {"quitar": call("agents", "remove_agent", "$agent_id")}},
    ]
    config_dir = tmp_path / "removal"
    config_dir.mkdir()
    for name, content in scn.items():
        (config_dir / f"{name}.json").write_text(json.dumps(content))
    configs = main.load_configs(config_dir, "distributions.json")
    main.run_single_simulation(1, configs, 8, tmp_path, 3, 5, [1],
                               trace_components=frozenset({"des", "agents"}), trace_name="removal")
    tracer.close()
    records = [json.loads(l) for l in (tmp_path / "traces" / "removal_run1.jsonl").read_text().splitlines()]
    removed_at = next(r["T"] for r in records if r["e"] == "removed" and r["agent"] == "Machine_2")
    assert removed_at == 2.0
    discards = [r for r in records if r["e"] == "discard" and r["agent"] == "Machine_2"]
    assert [r["T"] for r in discards] == [2.0]
    later = [r for r in records if r["e"] == "instant" and r["T"] > removed_at
             for a, _, _ in r["events"] if a == "Machine_2"]
    assert later == []


# ------------------------------------------------------------------------------------------------
# 2. Muestreo sin redondeo y periodicidad positiva al cargar
# ------------------------------------------------------------------------------------------------
def test_2_el_muestreador_no_redondea():
    dists = Distributions([UNIFORM, dict(ARRIVALS, distribution_name="k", output_type="int"), COIN], OFF, seed=0)
    values = [dists.sample("u") for _ in range(50)]
    assert any(v != round(v, 4) for v in values)
    assert all(isinstance(dists.sample("k"), int) for _ in range(10))
    assert dists.sample("coin") in ("L", "R")


def test_2_periodicidad_probabilistica_exige_truncamiento_minimo_positivo():
    dists = Distributions([UNIFORM, GAP, dict(GAP, distribution_name="gap_int", output_type="int")], OFF, seed=0)
    ev = lambda p: [{"event_category": "static", "signal": "wear_check", "agent_type": "Machine", "periodicity": p}]
    with pytest.raises(ValueError, match="min > 0"):
        Events(ev(prob("u")), dists)
    with pytest.raises(ValueError, match=">= 1"):
        Events(ev(prob("gap_int")), dists)
    with pytest.raises(ValueError):
        Events(ev(det(-1.0)), dists)
    assert Events(ev(prob("gap")), dists).data[0]["periodicity"] >= 0.5


# ------------------------------------------------------------------------------------------------
# 3. Errores explícitos
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("operations", [
    [{"operator": "/", "with": 0}],
    [{"operator": "+", "with": None}],
    [{"operator": "*", "with": "abc"}],
    [{"operator": "-", "with": "$missing"}],
])
def test_3_math_pipeline_rechaza_operandos_invalidos(operations):
    resolve = lambda v: evaluator.resolve_ephemeral(v, {"$x": 2})
    with pytest.raises(EvaluationError):
        evaluator._execute_math_pipeline({"initial_value": 1, "operations": operations}, resolve)


def test_3_math_pipeline_valido_no_cambia():
    resolve = lambda v: evaluator.resolve_ephemeral(v, {"$x": 2})
    pipeline = {"initial_value": "$x", "operations": [{"operator": "*", "with": "2.5"}, {"operator": "floor"},
                                                      {"operator": "max", "with": 1}]}
    assert evaluator._execute_math_pipeline(pipeline, resolve) == 5


def unresolved_arg_automaton():
    return automaton("bad_arg", ["DONE"],
                     go("LOAD", "DONE", {"action_required": {"r": call("agents", "read_agent_param", "$agent_id", "$param_key")}}))


def test_3_argumento_sin_resolver_fuera_del_motor_es_session_error():
    _, _, auts, _, _ = world(scenario([unresolved_arg_automaton()], ["bad_arg"]))
    with pytest.raises(SessionError, match="unresolved"):
        run_to_end(session(auts, "bad_arg"))


def test_3_argumento_sin_resolver_en_vivo_es_fatal(fatal, capsys):
    _, _, auts, _, _ = world(scenario([unresolved_arg_automaton()], ["bad_arg"]))
    with pytest.raises(fatal):
        run_to_end(session(auts, "bad_arg", live=True))
    assert "unresolved" in capsys.readouterr().err


def test_3_division_entre_cero_en_una_sesion_es_session_error():
    aut = automaton("div", ["DONE"], go("LOAD", "DONE", {"update_params": {"$r": pipe(1, ("/", 0))}}))
    _, _, auts, _, _ = world(scenario([aut], ["div"]))
    with pytest.raises(SessionError, match="division by zero"):
        run_to_end(session(auts, "div"))


# ------------------------------------------------------------------------------------------------
# 4. Falla distinta del valor vacío
# ------------------------------------------------------------------------------------------------
def test_4_call_method_propaga_excepciones():
    class Boom:
        def explode(self):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        evaluator.call_method("agents", "explode", [], Boom(), None)
    with pytest.raises(EvaluationError):
        evaluator.call_method("agents", "no_such_method", [], Boom(), None)


def empty_read_automaton():
    return automaton("empty_read", ["EMPTY", "FULL"],
                     go("LOAD", "READ", {"update_params": {"$agent_id": det("Machine_1"), "$param_key": det("missing")}}),
                     T("READ", "$v", call("agents", "read_agent_param", "$agent_id", "$param_key"),
                       case(c("$v", "==", None), "EMPTY"),
                       case(c("$v", "!=", None), "FULL")))


@pytest.mark.parametrize("live", [False, True])
def test_4_none_devuelto_es_valor_vacio_y_no_error(live):
    _, _, auts, _, _ = world(scenario([empty_read_automaton()], ["empty_read"]))
    s = session(auts, "empty_read", live=live)
    assert run_to_end(s) == "EMPTY"
    assert s.ctx["$v"] is None


def test_4_excepcion_dentro_del_metodo_es_error():
    _, _, auts, _, _ = world(scenario([empty_read_automaton()], ["empty_read"]))

    class Failing:
        def read_agent_param(self, agent_id, key):
            raise RuntimeError("storage offline")
    auts.agents = Failing()
    with pytest.raises(SessionError, match="RuntimeError"):
        run_to_end(session(auts, "empty_read"))


# ------------------------------------------------------------------------------------------------
# 5. Eventos estáticos y event_emit
# ------------------------------------------------------------------------------------------------
def test_5_senal_estatica_no_asignada_al_tipo_se_rechaza():
    scn = scenario()
    dists, _, auts, agents, _ = world(scn)
    bad = scn["events"] + [{"event_category": "static", "signal": "repair", "agent_type": "Machine",
                            "periodicity": det(1.0)}]
    with pytest.raises(ValueError, match="not an automaton assigned"):
        Events(bad, dists, scn["agents"], auts.by_name)
    static = [e for e in Events(bad, dists).data if e["event_category"] == "static"]
    with pytest.raises(ValueError, match="not an automaton assigned"):
        EventScheduler(static, agents, dists)


@pytest.mark.parametrize("target, message", [("Ghost_1", "nonexistent agent"), ("Machine_2", "not an automaton implemented")])
def test_5_event_emit_invalido_es_fatal_en_vivo(fatal, capsys, target, message):
    shout = automaton("shout", ["DONE"],
                      go("LOAD", "DONE", {"update_params": {"$signal": det("repair"), "$agent_id": det(target)}},
                         {"event_emit": ["$signal", "$agent_id"]}))
    _, _, auts, _, _ = world(scenario([shout], ["shout"]))
    auts.set_event_sink(lambda e: pytest.fail("no debe entregarse"))
    with pytest.raises(fatal):
        run_to_end(session(auts, "shout", live=True))
    assert message in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------
# 6 y 8. Instantáneas y eventos de canal fechados
# ------------------------------------------------------------------------------------------------
def simulate(scn, tmp_path, seed, max_time):
    dists = Distributions(scn["distributions"], OFF, seed=seed)
    envs = Environments(scn["environments"], dists, OFF)
    auts = Automata(scn["automata"], dists, OFF)
    agents = Agents(scn["agents"], dists, OFF)
    auts.set_objects(agents, envs)
    snaps = SnapshotManager(scn["patl"], OFF, disk_dir=tmp_path / f"snaps_{seed}")
    snaps.set_objects(agents, envs)
    agents.set_objects(auts, snaps)
    events = Events(scn["events"], dists, scn["agents"], auts.by_name)
    scheduler = EventScheduler([e for e in events.data if e["event_category"] == "static"], agents, dists)
    sim = DiscreteEventSimulator(dists, envs, auts, agents, events, scheduler, snaps)
    sim.run_simulation(max_time=max_time)
    return envs, snaps


def test_6_instantaneas_con_tiempo_simulado_y_evento_disparador(tmp_path):
    _, snaps = simulate(toy_scenario.scenario(), tmp_path, 4, 5)
    snapshots = [s for batch in snaps.read_and_delete() for s in batch]
    assert snapshots
    for s in snapshots:
        assert s["sim_time"] in (1.0, 2.0, 3.0, 4.0, 5.0)
        assert s["trigger_event"]["signal"] == s["automaton_name"] == "wear_check"
        assert s["trigger_event"]["agent_id"] == s["agent_id"]
        assert s["trigger_event"]["event_category"] == "static"


def test_6_eventos_de_canal_guardan_el_tiempo_simulado(tmp_path):
    for seed in range(1, 30):
        envs, _ = simulate(toy_scenario.scenario(with_channel=True), tmp_path, seed, 6)
        history = envs.data[0]["channels"][0]["events"]
        if history:
            break
    assert history, "alguna semilla debe producir una falla"
    for ev in history:
        assert ev["timestamp"] in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_8_calendario_redondea_a_nueve_decimales():
    cal = EventCalendar()
    cal.push(0.1 + 0.2, "a")
    cal.push(0.3, "b")
    assert cal.pop_instant() == (0.3, ["a", "b"])


def test_8_snapshot_manager_limpia_el_directorio_y_no_omite_archivos_ilegibles(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "snap_0000000099.json").write_text("{}")
    manager = SnapshotManager(toy_scenario.scenario()["patl"], OFF, disk_dir=directory)
    assert manager.get_snapshot_count() == 0
    (directory / "snap_0000000000.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable snapshot"):
        list(manager.read_and_delete())


def test_8_lecturas_devuelven_copias():
    _, envs, _, agents, _ = world(scenario())
    envs.clock = lambda: 7.5
    assert envs.write_ch_event("Plant_1", "maintenance", "Machine_1", "repair")
    envs.read_ch_members("Plant_1", "maintenance").append("Intruder_1")
    envs.read_members("Plant_1").append("Intruder_1")
    agents.get_all_agents("Machine").append("Intruder_1")
    events = envs.read_ch_events("Plant_1", "maintenance")
    events[0]["signal"] = "tampered"
    events.append({"agent_id": "Intruder_1"})
    channel = envs.data[0]["channels"][0]
    assert channel["members"] == ["Machine_1", "Technician_1"]
    assert "Intruder_1" not in envs.read_members("Plant_1")
    assert agents.get_all_agents("Machine") == ["Machine_1", "Machine_2"]
    assert channel["events"] == [{"agent_id": "Machine_1", "signal": "repair", "timestamp": 7.5}]


# ------------------------------------------------------------------------------------------------
# 7. Eliminación durante la sesión del propio agente
# ------------------------------------------------------------------------------------------------
def test_7_autoeliminacion_se_difiere_al_fin_de_la_sesion():
    leave = automaton("leave", ["GONE"],
                      go("LOAD", "AFTER",
                         {"update_params": {"$agent_id": call("events", "event_agent_id")}},
                         {"action_required": {"quitar": call("agents", "remove_agent", "$agent_id")}}),
                      go("AFTER", "GONE"))
    _, envs, _, agents, recorder = world(scenario([leave], ["leave"]))
    agents._run_event(agents._find("Machine_1"), {"event_category": "dynamic", "signal": "leave", "agent_id": "Machine_1"})
    assert [(x["state"], x["present"], x["member"]) for x in recorder.captures] == [
        ("LOAD", True, True), ("AFTER", True, True), ("GONE", True, True)]
    assert all(x["event"]["signal"] == "leave" for x in recorder.captures)
    assert agents._find("Machine_1") is None
    assert "Machine_1" not in envs.read_members("Plant_1")
    assert "Machine_1" not in envs.read_ch_members("Plant_1", "maintenance")
    assert agents.add_agent("Machine") == "Machine_3"


# ------------------------------------------------------------------------------------------------
# 9. Relaciones dirigidas
# ------------------------------------------------------------------------------------------------
def test_9_relaciones_son_pares_ordenados():
    _, envs, _, _, _ = world(scenario())
    assert envs.add_rel("Plant_1", "Machine_1", "Technician_1")
    assert envs.read_rel("Plant_1", "Machine_1", "Technician_1") is True
    assert envs.read_rel("Plant_1", "Technician_1", "Machine_1") is False
    assert envs.write_rel_param("Plant_1", "Technician_1", "Machine_1", "k", 1) is False
    assert envs.add_rel("Plant_1", "Technician_1", "Machine_1")
    assert envs.write_rel_param("Plant_1", "Machine_1", "Technician_1", "k", "forward")
    assert envs.read_rel_param("Plant_1", "Technician_1", "Machine_1", "k") is None
    assert envs.remove_rel("Plant_1", "Technician_1", "Machine_1")
    assert envs.read_rel_param("Plant_1", "Machine_1", "Technician_1", "k") == "forward"


def test_9_caso_runway_usa_inversionista_hacia_startup():
    cfg = ROOT / "configs/startups/runway-risk"
    envs = json.loads((cfg / "environments.json").read_text())
    investor_rels = [r["members"] for e in envs for r in e.get("relations", [])
                     if any(m.startswith("Investor_") for m in r["members"])]
    assert investor_rels and all(m[0].startswith("Investor_") and m[1] == "Startup_1" for m in investor_rels)

    def updates(node):
        if isinstance(node, dict):
            if "update_params" in node:
                yield node["update_params"]
            for v in node.values():
                yield from updates(v)
        elif isinstance(node, list):
            for v in node:
                yield from updates(v)

    for aut in json.loads((cfg / "automata.json").read_text()):
        if aut["automaton_name"] not in ("investment_decision", "investor_exit"):
            continue
        merged = {k: v for u in updates(aut) for k, v in u.items()}
        assert merged["$agent_a_id"] == det("$investor_id")
        assert merged["$agent_b_id"] == det("Startup_1")


# ------------------------------------------------------------------------------------------------
# 10. Procesamiento en orden de calendario con cota K
# ------------------------------------------------------------------------------------------------
class _StubAgents:
    def __init__(self):
        self.queues, self.order, self.on_process = {}, [], None
    def _find(self, agent_id): return {"agent_id": agent_id, "automata": ["x"]}
    def set_on_agent_added(self, cb): pass
    def start(self): pass
    def stop(self): pass
    def receive_event(self, agent_id, event):
        self.queues.setdefault(agent_id, []).append(event)
        return True
    def pending(self, agent_id): return len(self.queues.get(agent_id, []))
    def process_queue(self, agent_id, max_events):
        n = 0
        while self.queues.get(agent_id) and n < max_events:
            event = self.queues[agent_id].pop(0)
            self.order.append(event["name"])
            n += 1
            if self.on_process:
                self.on_process(event)
        return n


class _StubScheduler:
    def __init__(self, timed): self.timed = timed
    def start(self, calendar, T):
        for t, e in self.timed:
            calendar.push(t, e)
    def reschedule(self, event, T): pass
    def schedule_agent(self, agent, T): pass
    def stop(self): pass


class _StubAutomata:
    def set_event_sink(self, sink): self.sink = sink


class _StubEnvs:
    pass


def test_10_orden_de_calendario_con_cota_k_y_rezago():
    ev = lambda name, agent: {"event_category": "dynamic", "signal": "x", "agent_id": agent, "name": name}
    timed = [(1, ev("e1", "A")), (1, ev("e2", "B")), (1, ev("e3", "A")), (1, ev("e4", "A")),
             (2, ev("e5", "A")), (2, ev("e6", "B"))]
    agents = _StubAgents()
    sim = DiscreteEventSimulator(None, _StubEnvs(), _StubAutomata(), agents, None, _StubScheduler(timed), None,
                                 queue_batch=1)
    # e2 emite e7 con latencia cero: entra al mismo instante detrás de los eventos ya presentes.
    agents.on_process = lambda e: sim._emit(ev("e7", "C")) if e["name"] == "e2" else None
    sim.run_simulation(max_time=10)
    # T=1: e1 (A), e2 (B), e3 y e4 esperan (A ya procesó K=1), e7 (C).
    # T=2: rezago de A (e3), e5 espera, e6 (B). Vaciado final: e4, e5.
    assert agents.order == ["e1", "e2", "e7", "e3", "e6", "e4", "e5"]


# ------------------------------------------------------------------------------------------------
# 11. Cobertura de los casos de una transición probabilística
# ------------------------------------------------------------------------------------------------
def gate_transition(var, dist, *cases): return T("GATE", var, prob(dist), *cases)


def test_11_masa_sin_cubrir_continua_categorica_y_poisson():
    dists = Distributions([UNIFORM, COIN, ARRIVALS], OFF, seed=0)
    full = gate_transition("$x", "u", case(c("$x", "<", 0.5), "A"), case(c("$x", ">", 0.5), "B"))
    assert partition.uncovered_mass(full, dists) == pytest.approx(0.0, abs=1e-12)
    overlap = gate_transition("$x", "u", case(c("$x", "<", 0.5), "A"), case(c("$x", "<", 0.8), "B"),
                              case(c("$x", ">=", 0.8), "C"))
    assert partition.uncovered_mass(overlap, dists) == pytest.approx(0.0, abs=1e-12)
    hole = gate_transition("$x", "u", case(c("$x", "<", 0.3), "A"), case(c("$x", ">", 0.6), "B"))
    assert partition.uncovered_mass(hole, dists) == pytest.approx(0.3)
    band = gate_transition("$x", "u", case([c("$x", ">", 0.2), c("$x", "<=", 0.7)], "A"))
    assert partition.uncovered_mass(band, dists) == pytest.approx(0.5)

    one_side = gate_transition("$s", "coin", case(c("$s", "==", "L"), "A"))
    assert partition.uncovered_mass(one_side, dists) == pytest.approx(0.5)
    both = gate_transition("$s", "coin", case(c("$s", "==", "L"), "A"), case(c("$s", "!=", "L"), "B"))
    assert partition.uncovered_mass(both, dists) == 0.0

    pmf = stats.poisson.pmf(range(4), 1.0)
    pmf = pmf / pmf.sum()
    skip_one = gate_transition("$k", "arrivals", case(c("$k", "<", 1), "A"), case(c("$k", ">", 1), "B"))
    assert partition.uncovered_mass(skip_one, dists) == pytest.approx(pmf[1])
    assert partition.uncovered_mass(gate_transition("$k", "arrivals", case(c("$k", "!=", None), "A")), dists) == 0.0


def test_11_transiciones_que_no_aplican_se_omiten():
    dists = Distributions([UNIFORM], OFF, seed=0)
    runtime = gate_transition("$x", "u", case(c("$x", "<", "$limit"), "A"))
    deterministic = T("GATE", "$x", det(0.2), case(c("$x", "<", 0.1), "A"))
    other_var = gate_transition("$x", "u", case(c("$y", "<", 0.1), "A"))
    assert partition.uncovered_mass(runtime, dists) is None
    assert partition.uncovered_mass(deterministic, dists) is None
    assert partition.uncovered_mass(other_var, dists) is None


def test_11_automata_rechaza_casos_que_no_cubren_el_soporte():
    dists = Distributions([UNIFORM], OFF, seed=0)
    broken = automaton("broken", ["A", "B"], go("LOAD", "GATE"),
                       gate_transition("$x", "u", case(c("$x", "<", 0.3), "A"), case(c("$x", ">", 0.6), "B")))
    with pytest.raises(ValueError, match=r"broken::GATE.*0\.3"):
        Automata([broken], dists, OFF)


# ------------------------------------------------------------------------------------------------
# 12. Parámetro categórico fuera del motor
# ------------------------------------------------------------------------------------------------
def test_12_parametro_categorico_en_el_verificador_es_session_error():
    aut = automaton("pick", ["DONE"], go("LOAD", "DONE"), params={"$side": prob("coin")})
    _, _, auts, _, _ = world(scenario([aut], ["pick"], [COIN]))
    with pytest.raises(SessionError, match="categorical"):
        session(auts, "pick", live=False)
