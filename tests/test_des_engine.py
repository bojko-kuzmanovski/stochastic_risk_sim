"""
Pruebas del motor con un escenario de juguete definido solo en JSON: calendario, reproducibilidad por
semilla, activaciones estáticas, verificación PATL de punta a punta y ausencia de lógica de dominio en
el código fuente del motor.
"""
import json
import re
from pathlib import Path

import pytest
from scipy import stats

import main
import toy_scenario
from core.trace import tracer
from des.event_scheduler import EventCalendar

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIRS = ["core", "des", "patl", "metrics"]


def test_calendario_ordena_por_tiempo_y_agrupa_instantes():
    cal = EventCalendar()
    cal.push(2.0, "c")
    cal.push(1.0, "a")
    cal.push(1.0, "b")
    assert cal.pop_instant() == (1.0, ["a", "b"])
    assert cal.pop_instant() == (2.0, ["c"])
    assert len(cal) == 0


def run_toy(tmp_path, seed, max_time=6, trace=frozenset()):
    config_dir = toy_scenario.write(tmp_path / "toy")
    configs = main.load_configs(config_dir, "distributions.json")
    result = main.run_single_simulation(1, configs, max_time, tmp_path, seed, 5, [1],
                                        trace_components=trace, trace_name=f"toy_{seed}")
    tracer.close()
    return result


def test_misma_semilla_misma_trayectoria(tmp_path):
    a, b, c = run_toy(tmp_path, 7), run_toy(tmp_path, 7), run_toy(tmp_path, 8)
    assert a["metrics"] == b["metrics"] and a["patl_rows"] == b["patl_rows"]
    assert a["metrics"] != c["metrics"]


def test_eventos_estaticos_en_t0_mas_multiplos_de_la_periodicidad(tmp_path):
    run_toy(tmp_path, 3, max_time=6, trace=frozenset({"des"}))
    times = {"wear_check": [], "repair": []}
    for line in (tmp_path / "traces" / "toy_3_run1.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["e"] == "instant":
            for agent_id, signal, category in rec["events"]:
                if category == "static":
                    times[signal].append((rec["T"], agent_id))
    assert times["wear_check"] == [(float(t), m) for t in range(1, 7) for m in ("Machine_1", "Machine_2")]
    assert times["repair"] == [(2.0, "Technician_1"), (4.0, "Technician_1"), (6.0, "Technician_1")]


def test_verificacion_patl_de_punta_a_punta_con_otro_dominio(tmp_path):
    result = run_toy(tmp_path, 11)
    rows = result["patl_rows"]
    assert rows, "el escenario de juguete debe producir instantáneas en CHECK"
    # La máquina está en CHECK: la ronda 1 termina su sesión y la ronda 2 inicia otra.
    p = stats.beta.cdf(toy_scenario.FAIL_THRESHOLD, 2, 3)
    expected = 1 - (1 - p) ** 2
    for row in rows:
        assert row[4] == "MACHINE_FAILS"
        assert row[9] in ("SATISFIED", "VIOLATED")
        assert float(row[5]) == pytest.approx(expected, abs=5e-5)


def test_el_motor_no_contiene_nombres_de_ninguna_configuracion():
    def load(path):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    names = set()
    for config_dir in (ROOT / "configs").rglob("automata.json"):
        folder = config_dir.parent
        for a in load(folder / "automata.json") or []:
            names.add(a.get("automaton_name", ""))
        for ag in load(folder / "agents.json") or []:
            names.add(ag.get("agent_type", ""))
        for dist_file in folder.glob("distributions*.json"):
            names.update(d.get("distribution_name", "") for d in load(dist_file) or [])
        patl = load(folder / "patl.json") or {"observations": []}
        names.update(p.get("predicate_id", "") for o in patl["observations"] for p in o.get("predicates", []))
    toy = toy_scenario.scenario()
    names.update(a["automaton_name"] for a in toy["automata"])
    names.update(a["agent_type"] for a in toy["agents"])
    # Se ignoran nombres genéricos que el motor usa como vocabulario propio.
    names -= {"LOAD", "LOAD_PARAMS", "DONE"}
    names = {n for n in names if len(n) > 3}

    offenders = []
    for folder in ENGINE_DIRS + ["main.py"]:
        paths = [ROOT / folder] if folder.endswith(".py") else (ROOT / folder).rglob("*.py")
        for path in paths:
            source = path.read_text()
            for name in names:
                if re.search(rf"\b{re.escape(name)}\b", source):
                    offenders.append((path.relative_to(ROOT).as_posix(), name))
    assert not offenders, f"lógica de dominio en el motor: {offenders}"
