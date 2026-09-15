"""
Validación del verificador PATL_b frente al marco formal: juegos pequeños con valor calculable a mano
y propiedades generales de la semántica.
"""
import math

import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats

from game_kit import (COIN, UNIFORM, agent, automaton, c, case, gate, group, load, predicate, prob,
                      read_agent, snapshot, T, upd, value, verifier, write)


def ok(row):
    assert row["result"] != "ERROR", row["reason"]
    return row["value"]


# ----------------------------------------------------------------------------------------------
# Ejemplo de la tesis: retención de un cliente
# ----------------------------------------------------------------------------------------------
def test_ejemplo_retencion_cinco_septimos():
    retention = gate("retention", 5 / 7, success="RENUEVA", failure="ABANDONA")
    snap = snapshot(agent("Client_1", ["retention"]))
    coalition = group("Client_1", ("retention", ["RENUEVA"]))

    row = value([retention], snap, predicate(coalition, bound=0.45, depth=1))
    assert ok(row) == pytest.approx(5 / 7)
    assert row["result"] == "SATISFIED"

    row = value([retention], snap, predicate(coalition, bound=0.80, depth=1))
    assert row["result"] == "VIOLATED"


# ----------------------------------------------------------------------------------------------
# Formas cerradas, dualidad y monotonía
# ----------------------------------------------------------------------------------------------
@settings(max_examples=25, deadline=None)
@given(p=st.floats(min_value=0.05, max_value=0.95), depth=st.integers(min_value=1, max_value=4))
def test_alcanzabilidad_por_rondas_forma_cerrada(p, depth):
    snap = snapshot(agent("Client_1", ["g"]))
    reach = ok(value([gate("g", p)], snap, predicate(group("Client_1", ("g", ["SUCCESS"])), depth=depth)))
    assert reach == pytest.approx(1 - (1 - p) ** depth)
    assert 0.0 <= reach <= 1.0


@settings(max_examples=25, deadline=None)
@given(p=st.floats(min_value=0.05, max_value=0.95), depth=st.integers(min_value=1, max_value=4))
def test_invarianza_y_dualidad_sin_eleccion(p, depth):
    snap = snapshot(agent("Client_1", ["g"]))
    coalition = group("Client_1", ("g", ["SUCCESS"]))
    reach = ok(value([gate("g", p)], snap, predicate(coalition, depth=depth)))
    inv = ok(value([gate("g", p)], snap, predicate(coalition, ptype="invariance", depth=depth)))
    assert inv == pytest.approx((1 - p) ** depth)
    assert inv == pytest.approx(1 - reach)


@settings(max_examples=15, deadline=None)
@given(p=st.floats(min_value=0.05, max_value=0.95))
def test_alcanzabilidad_monotona_en_la_profundidad(p):
    snap = snapshot(agent("Client_1", ["g"]))
    coalition = group("Client_1", ("g", ["SUCCESS"]))
    values = [ok(value([gate("g", p)], snap, predicate(coalition, depth=d))) for d in (1, 2, 3)]
    assert values[0] <= values[1] <= values[2]


# ----------------------------------------------------------------------------------------------
# Dirección del operador y adversario que influye
# ----------------------------------------------------------------------------------------------
def adversarial_game():
    # El adversario actúa antes que la coalición en cada ronda (orden por agent_id).
    hurt = automaton("hurt", ["DONE"], load("DONE", write("Coal_1", "flag", 1)))
    calm = automaton("calm", ["DONE"], load("DONE", write("Coal_1", "flag", 0)))
    attempt = automaton("attempt", ["WIN", "LOSE"],
                        load("CHECK", upd(agent_id="Coal_1", param_key="flag")),
                        T("CHECK", "$flag", read_agent(),
                          case(c("$flag", "==", 1), "HARD"),
                          case(c("$flag", "!=", 1), "EASY")),
                        T("HARD", "$x", prob("u"), case(c("$x", "<", 0.3), "WIN"), case(c("$x", ">=", 0.3), "LOSE")),
                        T("EASY", "$y", prob("u"), case(c("$y", "<", 0.6), "WIN"), case(c("$y", ">=", 0.6), "LOSE")))
    snap = snapshot(agent("Adv_1", ["hurt", "calm"]), agent("Coal_1", ["attempt"], flag=0))
    return [hurt, calm, attempt], snap


def test_cota_inferior_el_adversario_minimiza():
    auts, snap = adversarial_game()
    pred = predicate(group("Coal_1", ("attempt", ["WIN"])), group("Adv_1", ("hurt", None), ("calm", None)), op=">=")
    # Orden uniforme: si "hurt" actúa antes (probabilidad 1/2) vale 0.3; si actúa después, 0.6. min(0.45, 0.6).
    assert ok(value(auts, snap, pred)) == pytest.approx(0.45)


def test_cota_superior_el_adversario_maximiza():
    auts, snap = adversarial_game()
    pred = predicate(group("Coal_1", ("attempt", ["WIN"])), group("Adv_1", ("hurt", None), ("calm", None)), op="<=", bound=0.65)
    row = value(auts, snap, pred)
    assert ok(row) == pytest.approx(0.6)
    assert row["result"] == "SATISFIED"


def test_existencial_y_universal_sobre_la_coalicion():
    auts = [gate("low", 0.2), gate("high", 0.7)]
    snap = snapshot(agent("Coal_1", ["low", "high"]))
    coalition = group("Coal_1", ("low", ["SUCCESS"]), ("high", ["SUCCESS"]))
    exists = ok(value(auts, snap, predicate(coalition, quantifier="exists")))
    forall = ok(value(auts, snap, predicate(coalition, quantifier="forall")))
    assert exists == pytest.approx(0.7)
    assert forall == pytest.approx(0.2)
    assert exists >= forall


# ----------------------------------------------------------------------------------------------
# Memoria acotada
# ----------------------------------------------------------------------------------------------
def memory_game():
    prep = automaton("prep", ["PREPARED"], load("PREPARED", write("Coal_1", "ready", 1)))
    attempt = automaton("try", ["SUCCESS", "FAIL"],
                        load("CHECK", upd(agent_id="Coal_1", param_key="ready")),
                        T("CHECK", "$ready", read_agent(),
                          case(c("$ready", "==", 1), "PREPARED_GATE"),
                          case(c("$ready", "!=", 1), "COLD_GATE")),
                        T("PREPARED_GATE", "$x", prob("u"), case(c("$x", "<", 0.9), "SUCCESS"), case(c("$x", ">=", 0.9), "FAIL")),
                        T("COLD_GATE", "$y", prob("u"), case(c("$y", "<", 0.2), "SUCCESS"), case(c("$y", ">=", 0.2), "FAIL")))
    snap = snapshot(agent("Coal_1", ["prep", "try"], ready=0))
    pred = predicate(group("Coal_1", ("prep", []), ("try", ["SUCCESS"])), depth=2)
    return [prep, attempt], snap, pred


def test_memoria_k1_repite_la_misma_accion():
    auts, snap, pred = memory_game()
    # Deterministas con k = 1: (prep, prep) vale 0 y (try, try) vale 1 - 0.8^2 = 0.36.
    assert ok(value(auts, snap, pred, memory=1, mixed=False)) == pytest.approx(0.36)
    # Aleatorizada sin memoria: prep con probabilidad p en cada ronda; el óptimo p cerca de 0.2568 da 0.4088.
    row = value(auts, snap, pred, memory=1)
    assert ok(row) == pytest.approx(0.4088, abs=1e-3) and row["strategy"] == "mixed"


def test_memoria_k2_alterna_y_supera_a_k1():
    auts, snap, pred = memory_game()
    # k = 2 permite (prep, try), que vale 0.9.
    assert ok(value(auts, snap, pred, memory=2)) == pytest.approx(0.9)


# ----------------------------------------------------------------------------------------------
# Información imperfecta de la coalición y adversario sin restricción
# ----------------------------------------------------------------------------------------------
def coin_automata():
    flip = automaton("coin", ["FLIPPED"],
                     load("FLIP"),
                     T("FLIP", "$side_draw", prob("coin"),
                       case(c("$side_draw", "==", "L"), "FLIPPED", write("Coal_1", "side", "$side_draw")),
                       case(c("$side_draw", "==", "R"), "FLIPPED", write("Coal_1", "side", "$side_draw"))))

    def go(label):
        return automaton(f"go_{label}", ["SUCCESS", "FAIL"],
                         load("CHECK", upd(agent_id="Coal_1", param_key="side")),
                         T("CHECK", "$side", read_agent(),
                           case(c("$side", "==", label), "SUCCESS"),
                           case(c("$side", "!=", label), "FAIL")))
    return [flip, go("L"), go("R")]


def test_coalicion_con_informacion_imperfecta_no_reacciona_a_la_moneda():
    auts = coin_automata()
    snap = snapshot(agent("Coal_1", ["coin", "go_L", "go_R"], side="none"))
    coalition = group("Coal_1", ("coin", []), ("go_L", ["SUCCESS"]), ("go_R", ["SUCCESS"]))
    pred = predicate(coalition, depth=2)
    # Determinista sin memoria no puede lanzar la moneda y después ir: vale 0.
    assert ok(value(auts, snap, pred, memory=1, distributions=(UNIFORM, COIN), mixed=False)) == pytest.approx(0.0)
    # Aleatorizada sin memoria: con probabilidad x de lanzar la moneda y el resto repartido entre los lados, el
    # valor es x (1 - x) / 2, máximo 1/8 en x = 1/2.
    assert ok(value(auts, snap, pred, memory=1, distributions=(UNIFORM, COIN))) == pytest.approx(0.125, abs=1e-6)
    # Con memoria lanza y va a un lado fijo: 0.5. Con información perfecta valdría 1.
    for k in (2, 3):
        assert ok(value(auts, snap, pred, memory=k, distributions=(UNIFORM, COIN))) == pytest.approx(0.5)


def test_adversario_sin_restriccion_reacciona_al_estado():
    auts = coin_automata()

    def block(label):
        return automaton(f"block_{label}", ["DONE"], load("DONE", write("Coal_1", "blocked", label)))

    go_l = automaton("go_L", ["SUCCESS", "FAIL"],
                     load("CHECK", upd(agent_id="Coal_1", param_key="side")),
                     T("CHECK", "$side", read_agent(),
                       case(c("$side", "==", "L"), "CHECK_BLOCK", upd(agent_id="Coal_1", param_key="blocked")),
                       case(c("$side", "!=", "L"), "FAIL")),
                     T("CHECK_BLOCK", "$blocked", read_agent(),
                       case(c("$blocked", "==", "L"), "FAIL"),
                       case(c("$blocked", "!=", "L"), "SUCCESS")))
    snap = snapshot(agent("Coal_1", ["coin", "go_L"], side="none", blocked="none"),
                    agent("Zadv_1", ["block_L", "block_R"]))
    pred = predicate(group("Coal_1", ("coin", []), ("go_L", ["SUCCESS"])),
                     group("Zadv_1", ("block_L", None), ("block_R", None)), depth=2)
    # El adversario actúa después de la moneda y bloquea el lado que salió: vale 0.
    # Un adversario restringido a una acción fija dejaría 0.5.
    assert ok(value([auts[0], go_l, block("L"), block("R")], snap, pred, memory=2,
                    distributions=(UNIFORM, COIN))) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------------------
# Ramas exactas y ausencia de muestreo
# ----------------------------------------------------------------------------------------------
def test_soporte_poisson_truncado_exacto():
    poisson = {"distribution_name": "arrivals", "family": "poisson", "params": {"lambda": 1.0},
               "output_type": "float", "truncation": {"min": 0.0, "max": 3.0}}
    aut = automaton("arrive", ["ARRIVED", "QUIET"], load("DRAW"),
                    T("DRAW", "$k", prob("arrivals"), case(c("$k", ">", 0.5), "ARRIVED"), case(c("$k", "<=", 0.5), "QUIET")))
    snap = snapshot(agent("Shock_1", ["arrive"]))
    got = ok(value([aut], snap, predicate(group("Shock_1", ("arrive", ["ARRIVED"]))), distributions=(poisson,)))
    pmf = stats.poisson.pmf(range(4), 1.0)
    assert got == pytest.approx(1 - pmf[0] / pmf.sum())


def test_la_verificacion_no_consume_el_generador():
    auts, snap, pred = memory_game()
    v, dists = verifier(auts, memory=[2])
    before = (dists.rng.getstate(), dists.np_rng.bit_generator.state)
    first = v.verify(snap, [pred])[0]["value"]
    second = v.verify(snap, [pred])[0]["value"]
    assert first == second
    assert (dists.rng.getstate(), dists.np_rng.bit_generator.state) == before


def test_esperanza_condicional_contra_integracion():
    _, dists = verifier([gate("g", 0.5)], distributions=({"distribution_name": "b", "family": "beta",
                                                           "params": {"alpha": 2, "beta": 2}, "output_type": "float",
                                                           "truncation": {"min": 0.0, "max": 1.0}},))
    # E[X | X > 1/2] con X ~ Beta(2,2) = 0.6875
    assert dists.conditional_mean("b", 0.5, 1.0, False, True) == pytest.approx(0.6875, rel=1e-6)


# ----------------------------------------------------------------------------------------------
# Errores reportados, no valores inventados
# ----------------------------------------------------------------------------------------------
def test_casos_que_no_particionan_el_soporte():
    broken = automaton("broken", ["A", "B"], load("GATE"),
                       T("GATE", "$x", prob("u"), case(c("$x", "<", 0.3), "A"), case(c("$x", ">", 0.6), "B")))
    row = value([broken], snapshot(agent("Coal_1", ["broken"])), predicate(group("Coal_1", ("broken", ["A"]))))
    assert row["result"] == "ERROR" and "masa" in row["reason"]


def test_estado_objetivo_no_final():
    row = value([gate("g", 0.5)], snapshot(agent("Coal_1", ["g"])), predicate(group("Coal_1", ("g", ["GATE"]))))
    assert row["result"] == "ERROR" and "no son finales" in row["reason"]


def test_automata_no_asignado_al_agente():
    row = value([gate("g", 0.5), gate("h", 0.5)], snapshot(agent("Coal_1", ["g"])),
                predicate(group("Coal_1", ("h", ["SUCCESS"]))))
    assert row["result"] == "ERROR" and "no tiene asignado" in row["reason"]


# ----------------------------------------------------------------------------------------------
# Regresiones de la auditoría
# ----------------------------------------------------------------------------------------------
def test_c1_ramas_hermanas_no_comparten_escrituras():
    # Escribe p=1, bifurca 50/50; la última rama escribe p=2. Éxito si al leer p vale 1: valor 0.5.
    aut = automaton("alias", ["SUCCESS", "FAIL"],
                    load("W1", write("Coal_1", "p", 1)),
                    T("W1", "$x", prob("u"),
                      case(c("$x", "<", 0.5), "READ", upd(agent_id="Coal_1", param_key="p")),
                      case(c("$x", ">=", 0.5), "READ", write("Coal_1", "p", 2), upd(agent_id="Coal_1", param_key="p"))),
                    T("READ", "$p", read_agent(), case(c("$p", "==", 1), "SUCCESS"), case(c("$p", "!=", 1), "FAIL")))
    row = value([aut], snapshot(agent("Coal_1", ["alias"], p=0)), predicate(group("Coal_1", ("alias", ["SUCCESS"]))))
    assert ok(row) == pytest.approx(0.5)


def test_c2_falla_de_sesion_se_reporta_como_error():
    aut = automaton("broken_read", ["A", "B"],
                    load("READ", upd(agent_id="Coal_1", param_key="missing")),
                    T("READ", "$v", read_agent(), case(c("$v", "==", 1), "A"), case(c("$v", "!=", 1), "B")))
    row = value([aut], snapshot(agent("Coal_1", ["broken_read"])), predicate(group("Coal_1", ("broken_read", ["A"]))))
    assert row["result"] == "ERROR" and "Resolved Value" in row["reason"]


def test_c3_adversarios_implicitos_acotados():
    auts, _ = adversarial_game()
    many = [agent(f"Adv_{i}", ["hurt", "calm"]) for i in range(1, 13)]
    snap = snapshot(agent("Coal_1", ["attempt"], flag=0), *many)
    row = value(auts, snap, predicate(group("Coal_1", ("attempt", ["WIN"]))))
    assert row["result"] == "ERROR" and "exceden el límite" in row["reason"]


def test_c4_la_sesion_en_curso_del_disparador_se_completa():
    auts, _ = adversarial_game()
    adv = agent("Adv_1", ["hurt", "calm"])
    adv.update({"current_automaton": "hurt", "current_state": "LOAD", "session_ctx": {}})
    snap = snapshot(adv, agent("Coal_1", ["attempt"], flag=0), trigger="Adv_1")
    # Aunque el adversario solo puede elegir "calm", termina su sesión de "hurt" en la ronda 1. Con orden
    # uniforme: si la termina antes de que actúe la coalición vale 0.3, si después 0.6; en promedio 0.45.
    pred = predicate(group("Coal_1", ("attempt", ["WIN"])), group("Adv_1", ("calm", None)))
    assert ok(value(auts, snap, pred)) == pytest.approx(0.45)


def test_c5_casos_solapados_con_regla_del_primer_caso():
    aut = automaton("overlap", ["A", "B", "C"], load("GATE"),
                    T("GATE", "$x", prob("u"),
                      case(c("$x", "<", 0.5), "A"),
                      case(c("$x", "<", 0.8), "B"),
                      case(c("$x", ">=", 0.8), "C")))
    snap = snapshot(agent("Coal_1", ["overlap"]))
    assert ok(value([aut], snap, predicate(group("Coal_1", ("overlap", ["B"]))))) == pytest.approx(0.3)
    assert ok(value([aut], snap, predicate(group("Coal_1", ("overlap", ["A"]))))) == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------
# Segunda auditoría: estrategias aleatorizadas, cuantiles, U, R y X, orden uniforme, tipos de salida
# ----------------------------------------------------------------------------------------------
def test_pares_o_nones_requiere_aleatorizar():
    # La coalición gana si su lado coincide con el del adversario, que elige al inicio de la ronda sin ver
    # la realización de la mezcla. Determinista vale 0; aleatorizada 1/2 vale 1/2.
    def pick(label):
        return automaton(f"pick_{label}", ["DONE"], load("DONE", write("Coal_1", "side", label)))

    def guess(label):
        return automaton(f"guess_{label}", ["SUCCESS", "FAIL"],
                         load("CHECK", upd(agent_id="Coal_1", param_key="side")),
                         T("CHECK", "$side", read_agent(),
                           case(c("$side", "!=", label), "SUCCESS"), case(c("$side", "==", label), "FAIL")))
    auts = [pick("L"), pick("R"), guess("L"), guess("R")]
    snap = snapshot(agent("Coal_1", ["guess_L", "guess_R"], side="none"), agent("Adv_1", ["pick_L", "pick_R"]))
    pred = predicate(group("Coal_1", ("guess_L", ["SUCCESS"]), ("guess_R", ["SUCCESS"])),
                     group("Adv_1", ("pick_L", None), ("pick_R", None)))
    # guess_X gana si el lado escrito es distinto de X; con orden uniforme el adversario escribe antes con 1/2.
    det = ok(value(auts, snap, pred, mixed=False))
    mix = ok(value(auts, snap, pred))
    assert mix > det + 0.1


def test_cuantiles_propagan_el_valor_muestreado():
    aut = automaton("prop", ["SUCCESS", "FAIL"], load("DRAW"),
                    T("DRAW", "$x", prob("u"), case(c("$x", ">=", 0.0), "READ", write("Coal_1", "p", "$x"),
                                                    upd(agent_id="Coal_1", param_key="p"))),
                    T("READ", "$p", read_agent(), case(c("$p", "<", 0.25), "SUCCESS"), case(c("$p", ">=", 0.25), "FAIL")))
    snap = snapshot(agent("Coal_1", ["prop"], p=1.0))
    pred = predicate(group("Coal_1", ("prop", ["SUCCESS"])))
    assert ok(value([aut], snap, pred, quantiles=8)) == pytest.approx(0.25)
    assert ok(value([aut], snap, pred, quantiles=1)) == pytest.approx(0.0)


def three_way(p, q):
    return automaton("g3", ["SUCCESS", "FAIL", "RETRY"], load("GATE"),
                     T("GATE", "$x", prob("u"),
                       case(c("$x", "<", p), "SUCCESS"),
                       case(c("$x", "<", p + q), "FAIL"),
                       case(c("$x", ">=", p + q), "RETRY")))


def test_until_release_next_formas_cerradas():
    p, q = 0.3, 0.2
    r = 1 - p - q
    snap = snapshot(agent("Coal_1", ["g3"]))
    coalition = group("Coal_1", ("g3", ["SUCCESS"]))
    succ = {"prop": {"automaton": "g3", "states": ["SUCCESS"]}}
    fail = {"prop": {"automaton": "g3", "states": ["FAIL"]}}
    until = predicate(coalition, ptype="until", depth=3)
    until.update(left={"not": fail}, right=succ)
    assert ok(value([three_way(p, q)], snap, until)) == pytest.approx(p * (1 + r + r ** 2))
    release = predicate(coalition, ptype="release", depth=3)
    release.update(left=succ, right={"not": fail})
    assert ok(value([three_way(p, q)], snap, release)) == pytest.approx(p * (1 + r + r ** 2) + r ** 3)
    nxt = predicate(coalition, ptype="next")
    nxt.update(right=succ)
    assert ok(value([three_way(p, q)], snap, nxt)) == pytest.approx(p)


def test_el_valor_no_depende_del_nombre_de_los_agentes():
    auts, snap = adversarial_game()
    pred = predicate(group("Coal_1", ("attempt", ["WIN"])), group("Adv_1", ("hurt", None), ("calm", None)))
    renamed = snapshot(agent("Zadv_1", ["hurt", "calm"]), agent("Coal_1", ["attempt"], flag=0))
    pred_renamed = predicate(group("Coal_1", ("attempt", ["WIN"])), group("Zadv_1", ("hurt", None), ("calm", None)))
    assert ok(value(auts, snap, pred)) == pytest.approx(ok(value(auts, renamed, pred_renamed)))


def test_continua_con_salida_entera():
    normal_int = {"distribution_name": "units", "family": "normal", "params": {"mean": 2.0, "sigma": 1.0},
                  "output_type": "int", "truncation": {"min": 0.0, "max": 5.0}}
    aut = automaton("units_gate", ["LOW", "HIGH"], load("DRAW"),
                    T("DRAW", "$n", prob("units"), case(c("$n", "<", 1.5), "LOW"), case(c("$n", ">=", 1.5), "HIGH")))
    got = ok(value([aut], snapshot(agent("Coal_1", ["units_gate"])),
                   predicate(group("Coal_1", ("units_gate", ["LOW"]))), distributions=(normal_int,)))
    d = stats.norm(2, 1)
    assert got == pytest.approx((d.cdf(2) - d.cdf(0)) / (d.cdf(5) - d.cdf(0)))


def test_categorica_con_salida_numerica():
    cat = {"distribution_name": "lvl", "family": "categorical", "output_type": "int",
           "params": {"categories": [{"label": "1", "probability": 0.3}, {"label": "2", "probability": 0.7}]}}
    aut = automaton("lvl_gate", ["ONE", "OTHER"], load("DRAW"),
                    T("DRAW", "$v", prob("lvl"), case(c("$v", "==", 1), "ONE"), case(c("$v", "!=", 1), "OTHER")))
    got = ok(value([aut], snapshot(agent("Coal_1", ["lvl_gate"])), predicate(group("Coal_1", ("lvl_gate", ["ONE"]))),
                   distributions=(cat,)))
    assert got == pytest.approx(0.3)


def test_comparacion_tolera_redondeo_de_punto_flotante():
    snap = snapshot(agent("Coal_1", ["g"]))
    row = value([gate("g", 0.7)], snap, predicate(group("Coal_1", ("g", ["SUCCESS"])), depth=2, bound=0.91))
    assert row["result"] == "SATISFIED"
