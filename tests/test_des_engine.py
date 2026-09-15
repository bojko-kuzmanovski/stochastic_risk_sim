"""Pruebas del motor DES: calendario, reproducibilidad por semilla y activaciones estáticas."""
import json
from pathlib import Path

from core.agents import Agents
from core.automata import Automata
from core.distributions import Distributions
from core.environments import Environments
from core.events import Events
from core.trace import tracer
from des.discrete_event_engine import DiscreteEventSimulator
from des.event_scheduler import EventCalendar, EventScheduler
from metrics.metrics_collector import MetricsCollector
from patl.snapshot_manager import SnapshotManager

CFG = Path("configs/startups/runway-risk")


def test_calendario_ordena_por_tiempo_y_agrupa_instantes():
    cal = EventCalendar()
    cal.push(2.0, "c")
    cal.push(1.0, "a")
    cal.push(1.0, "b")
    assert cal.pop_instant() == (1.0, ["a", "b"])
    assert cal.pop_instant() == (2.0, ["c"])
    assert len(cal) == 0


def run_des(seed, max_time, tmp_path, trace=frozenset()):
    configs = {name: json.loads((CFG / (f"distributions_1_base.json" if name == "distributions" else f"{name}.json")).read_text())
               for name in ["distributions", "environments", "automata", "agents", "events", "patl"]}
    tracer.configure(tmp_path / f"trace_{seed}.jsonl", trace, 1)
    metrics = MetricsCollector(enabled=True)
    dists = Distributions(configs["distributions"], metrics, seed=seed)
    envs = Environments(configs["environments"], dists, metrics)
    auts = Automata(configs["automata"], dists, metrics)
    agents = Agents(configs["agents"], dists, metrics)
    auts.set_objects(agents, envs)
    snaps = SnapshotManager(configs["patl"], metrics, disk_dir=tmp_path / f"snaps_{seed}")
    snaps.set_objects(agents, envs)
    agents.set_objects(auts, snaps)
    events = Events(configs["events"], dists)
    scheduler = EventScheduler([e for e in events.data if e["event_category"] == "static"], agents, dists)
    DiscreteEventSimulator(dists, envs, auts, agents, events, scheduler, snaps).run_simulation(max_time)
    tracer.close()
    snaps.clear_all_snapshots()
    return metrics.to_dict(), [dict(a["params"]) for a in agents.data]


def test_misma_semilla_misma_trayectoria(tmp_path):
    a = run_des(7, 12, tmp_path)
    b = run_des(7, 12, tmp_path)
    c = run_des(8, 12, tmp_path)
    assert a == b
    assert a != c


def test_eventos_estaticos_se_activan_en_t0_mas_multiplos_de_la_periodicidad(tmp_path):
    run_des(3, 12, tmp_path, trace=frozenset({"des"}))
    dispatched = []
    for line in (tmp_path / "trace_3.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["e"] == "instant":
            for agent_id, signal, category in rec["events"]:
                if category == "static" and signal == "runway_lifecycle":
                    dispatched.append(rec["T"])
    # Periodicidad determinista 1: primera activación en t0 = 1 y luego cada unidad hasta T_max = 12.
    assert dispatched == [float(t) for t in range(1, 13)]
