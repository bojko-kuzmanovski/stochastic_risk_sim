"""
Herramienta de autoría del caso runway (no forma parte del motor y nada del motor la importa).

Genera configs/startups/runway-risk/automata.json y patl.json conformes a los esquemas y calibra los
umbrales binarios. El motor solo lee los JSON resultantes; cualquier otra configuración funciona igual.

Uso: python3 tools/build_runway_config.py .
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
CFG = ROOT / "configs/startups/runway-risk"
REGIMES = ["1_base", "2_bull", "3_bear", "4_crisis"]
FLOOR = 0.02

ENV = "StartupEcosystem_1"
S, F, TM = "Startup_1", "Founder_1", "TalentMarket_1"


# ================================================================ DSL
def det(v): return {"type": "deterministic", "value": v}
def prob(d): return {"type": "probabilistic", "distribution": d}
def _w(x): return det(x) if isinstance(x, (bool, int, float)) else x
def pipe(initial, *ops):
    return {"target": "system", "method": "math_pipeline",
            "params": {"initial_value": _w(initial), "operations": [{"operator": o, "with": _w(w)} for o, w in ops]}}
def c(var, op, val): return {"variable": var, "operator": op, "value": val}
def upd(**kv): return {"update_params": {f"${k}": (v if isinstance(v, dict) else det(v)) for k, v in kv.items()}}
def req(name, v): return {"action_required": {name: v}}
def case(conds, to, *actions):
    flat = []
    for a in actions:
        flat.extend(a if isinstance(a, list) else [a])
    return {"threshold_case": conds if isinstance(conds, list) else [conds], "to": to, "actions": flat}
def T(frm, var, tv, *cases): return {"from": frm, "threshold_value": {var: tv}, "thresholds": list(cases)}
def fixed(t):
    # Compuerta diseñada a mano (soporte solapado entre regímenes): la calibración no la mueve.
    t["__fixed__"] = True
    return t
def always(frm, to, *actions):
    v = f"$always_{frm.lower()}"
    return T(frm, v, det(True), case(c(v, "==", True), to, *actions))

# Lecturas: el esquema exige parámetros "$agent_id", "$param_key" (o "$env_id", "$param_key"); los fija
# la transición de entrada al estado que lee.
def rd_agent(agent, key): return {"__read__": ("agent", agent, key)}
def rd_env(key): return {"__read__": ("env", None, key)}
def rd_rel(): return {"__read__": ("rel", None, None)}
def event_agent(): return {"target": "events", "method": "event_agent_id", "params": []}

def w_agent(name, agent, key, value):
    return [upd(agent_id=agent, param_key=key, param_value=value),
            req(name, {"target": "agents", "method": "write_agent_param", "params": ["$agent_id", "$param_key", "$param_value"]})]
def w_env(name, key, value):
    return [upd(param_key=key, param_value=value),
            req(name, {"target": "environments", "method": "write_env_param", "params": ["$env_id", "$param_key", "$param_value"]})]
def remove_rel(name):
    return [req(name, {"target": "environments", "method": "remove_rel", "params": ["$env_id", "$agent_a_id", "$agent_b_id"]})]
def emit_to(signal, agent):
    return [upd(signal=signal, agent_id=agent), {"event_emit": ["$signal", "$agent_id"]}]


def A(name, initial, finals, *transitions):
    transitions = list(transitions)
    # Resuelve las lecturas e inyecta sus parámetros en las transiciones de entrada.
    for t in transitions:
        var, tv = next(iter(t["threshold_value"].items()))
        if not (isinstance(tv, dict) and "__read__" in tv):
            continue
        kind, agent, key = tv["__read__"]
        if kind == "agent":
            t["threshold_value"][var] = {"target": "agents", "method": "read_agent_param", "params": ["$agent_id", "$param_key"]}
            need = upd(agent_id=agent, param_key=key)
        elif kind == "env":
            t["threshold_value"][var] = {"target": "environments", "method": "read_env_param", "params": ["$env_id", "$param_key"]}
            need = upd(param_key=key)
        else:
            t["threshold_value"][var] = {"target": "environments", "method": "read_rel", "params": ["$env_id", "$agent_a_id", "$agent_b_id"]}
            need = None
        if t["from"] == initial and need is not None:
            raise ValueError(f"{name}: el estado inicial {initial} no puede leer")
        if need is not None:
            incoming = [th for u in transitions for th in u["thresholds"] if th["to"] == t["from"]]
            if not incoming:
                raise ValueError(f"{name}: {t['from']} no tiene transiciones de entrada")
            for th in incoming:
                th["actions"].append(need)
    return {"automaton_name": name, "states": {"initial": initial, "final": finals}, "params": {}, "transitions": transitions}


automata = []

# ================================================================ runway_lifecycle (Startup)
automata.append(A("runway_lifecycle", "LOAD_PARAMS", ["DEAD", "DONE_CYCLE"],
    always("LOAD_PARAMS", "STOCHASTIC_DRIFT"),
    T("STOCHASTIC_DRIFT", "$drift", prob("revenue_volatility"),
      case(c("$drift", "<", 0.85), "MACRO_DRIFT_FILTER"),
      case(c("$drift", ">=", 0.85), "EVALUATE_MARKET_SURVIVAL")),
    T("MACRO_DRIFT_FILTER", "$macro_noise", prob("market_noise"),
      case(c("$macro_noise", "<", -0.01), "RESCUE_ROUND"),
      case(c("$macro_noise", ">=", -0.01), "EVALUATE_MARKET_SURVIVAL")),
    T("EVALUATE_MARKET_SURVIVAL", "$sentiment", prob("market_sentiment"),
      case(c("$sentiment", "<=", 0.3), "RESCUE_ROUND"),
      case([c("$sentiment", ">", 0.3), c("$sentiment", "<=", 0.45)], "CASH_BUFFER_CHECK"),
      case([c("$sentiment", ">", 0.45), c("$sentiment", "<=", 0.65)], "STOCHASTIC_STRESS"),
      case(c("$sentiment", ">", 0.65), "OPTIMISTIC_BURNOUT_CHECK")),
    T("CASH_BUFFER_CHECK", "$cash_buffer", rd_agent(S, "cash"),
      case(c("$cash_buffer", ">", 150000), "STOCHASTIC_STRESS"),
      case(c("$cash_buffer", "<=", 150000), "RESCUE_ROUND")),
    fixed(T("RESCUE_ROUND", "$rescue", prob("revenue_volatility"),
      case(c("$rescue", ">=", 0.95), "DONE_CYCLE", emit_to("runway_critical", S)),
      case(c("$rescue", "<", 0.95), "DEAD", emit_to("startup_dead", S)))),
    T("OPTIMISTIC_BURNOUT_CHECK", "$burn_risk", prob("founder_optimism"),
      case(c("$burn_risk", ">=", 0.85), "STOCHASTIC_STRESS"),
      case(c("$burn_risk", "<", 0.85), "VOLATILITY_ABSORPTION")),
    T("VOLATILITY_ABSORPTION", "$beta_absorb", prob("volatility_profile"),
      case(c("$beta_absorb", ">", 0.4), "STOCHASTIC_STRESS"),
      case(c("$beta_absorb", "<=", 0.4), "DONE_CYCLE", emit_to("seeking_funding", F))),
    T("STOCHASTIC_STRESS", "$noise", prob("market_noise"),
      case(c("$noise", "<", 0.0), "RESCUE_ROUND"),
      case(c("$noise", ">=", 0.0), "DONE_CYCLE", emit_to("runway_critical", S))),
))

# ================================================================ startup_dead (Startup)
automata.append(A("startup_dead", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "SET_FLAG"),
    T("SET_FLAG", "$deaths", rd_agent(S, "deaths"),
      case(c("$deaths", "!=", None), "REFOUND",
           w_agent("marcar_quiebra", S, "alive", False),
           w_agent("contar_quiebra", S, "deaths", pipe("$deaths", ("+", 1))))),
    always("REFOUND", "DONE",
           w_agent("recapitalizar", S, "cash", 300000),
           w_agent("reiniciar_costos", S, "cost_fixed", 40000),
           w_agent("reiniciar_plantilla", S, "headcount", 8),
           w_agent("reactivar", S, "alive", True)),
))

# ================================================================ burn_rate_calculator (Startup)
automata.append(A("burn_rate_calculator", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_FIXED"),
    T("FETCH_FIXED", "$fixed_cost", rd_agent(S, "cost_fixed"), case(c("$fixed_cost", "!=", None), "FETCH_VARIABLE")),
    T("FETCH_VARIABLE", "$var_cost", rd_agent(S, "cost_variable_per_head"), case(c("$var_cost", "!=", None), "FETCH_HEADCOUNT")),
    T("FETCH_HEADCOUNT", "$headcount", rd_agent(S, "headcount"), case(c("$headcount", "!=", None), "FETCH_REVENUE")),
    T("FETCH_REVENUE", "$revenue", rd_agent(S, "revenue_current"), case(c("$revenue", "!=", None), "COMPUTE_BURN")),
    T("COMPUTE_BURN", "$inflation", prob("cost_inflation"),
      case(c("$inflation", ">", 1.05), "FETCH_CASH",
           upd(burn=pipe("$var_cost", ("*", "$headcount"), ("+", "$fixed_cost"), ("*", "$inflation"), ("-", "$revenue"))),
           w_agent("guardar_burn_inflado", S, "burn_rate", "$burn")),
      case(c("$inflation", "<=", 1.05), "FETCH_CASH",
           upd(burn=pipe("$var_cost", ("*", "$headcount"), ("+", "$fixed_cost"), ("-", "$revenue"))),
           w_agent("guardar_burn_normal", S, "burn_rate", "$burn"))),
    T("FETCH_CASH", "$cash_br", rd_agent(S, "cash"),
      case(c("$cash_br", "!=", None), "COMPUTE_RUNWAY",
           upd(new_cash=pipe("$cash_br", ("-", "$burn"))),
           w_agent("descontar_burn", S, "cash", "$new_cash"))),
    always("COMPUTE_RUNWAY", "DONE",
           upd(runway_val=pipe("$new_cash", ("/", pipe("$burn", ("max", 1))))),
           w_agent("guardar_runway", S, "runway", "$runway_val")),
))

# ================================================================ revenue_generator (RevenueStream)
automata.append(A("revenue_generator", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "STABLE"),
    T("STABLE", "$base", prob("startup_revenue"), case(c("$base", "!=", None), "FETCH_VOLATILITY")),
    T("FETCH_VOLATILITY", "$vol", prob("revenue_volatility"), case(c("$vol", "!=", None), "FETCH_GROWTH")),
    T("FETCH_GROWTH", "$growth", prob("growth_rate"), case(c("$growth", "!=", None), "CHECK_SHOCK")),
    T("CHECK_SHOCK", "$shock", prob("shock_arrival"),
      case(c("$shock", ">", 0.5), "COMPUTE_REVENUE_SHOCKED"),
      case(c("$shock", "<=", 0.5), "COMPUTE_REVENUE_NORMAL")),
    always("COMPUTE_REVENUE_NORMAL", "SAVE_STARTUP", upd(rev=pipe(1, ("+", "$growth"), ("*", "$base"), ("*", "$vol")))),
    T("COMPUTE_REVENUE_SHOCKED", "$shock_mag", rd_agent("RevenueStream_1", "shock_magnitude"),
      case(c("$shock_mag", "!=", None), "SAVE_STARTUP",
           upd(rev=pipe(1, ("+", "$growth"), ("*", "$base"), ("*", "$vol"), ("*", pipe(1, ("-", "$shock_mag"))))))),
    always("SAVE_STARTUP", "SAVE_STREAM", w_agent("guardar_en_startup", S, "revenue_current", "$rev")),
    always("SAVE_STREAM", "DONE", w_agent("guardar_en_stream", "RevenueStream_1", "revenue_current", "$rev")),
))

# ================================================================ other_investment_controller (Startup)
automata.append(A("other_investment_controller", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_OTHER_BUDGET"),
    T("FETCH_OTHER_BUDGET", "$budget_oi", rd_agent(F, "other_budget"),
      case(c("$budget_oi", ">", 0), "APPLY_INVESTMENT"),
      case(c("$budget_oi", "<=", 0), "DONE")),
    T("APPLY_INVESTMENT", "$rev_vol", prob("revenue_volatility"),
      case(c("$rev_vol", "!=", None), "UPDATE_REVENUE", upd(incremental=pipe("$rev_vol", ("*", 0.05), ("*", "$budget_oi"))))),
    T("UPDATE_REVENUE", "$rev_oi", rd_agent(S, "revenue_current"),
      case(c("$rev_oi", "!=", None), "SPEND_BUDGET", w_agent("sumar_ingreso", S, "revenue_current", pipe("$rev_oi", ("+", "$incremental"))))),
    T("SPEND_BUDGET", "$cash_oi", rd_agent(S, "cash"),
      case(c("$cash_oi", "!=", None), "DONE",
           w_agent("pagar_inversion", S, "cash", pipe("$cash_oi", ("-", "$budget_oi"))),
           w_agent("agotar_presupuesto", F, "other_budget", 0))),
))

# ================================================================ strategic_resource_allocator (Founder)
automata.append(A("strategic_resource_allocator", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_OPTIMISM"),
    T("FETCH_OPTIMISM", "$optimism", prob("founder_optimism"), case(c("$optimism", "!=", None), "FETCH_BELIEFS")),
    T("FETCH_BELIEFS", "$prod", rd_agent(F, "estimated_productivity_per_head"),
      case(c("$prod", "!=", None), "FETCH_ELASTICITY_BELIEF", upd(adjusted_productivity=pipe("$optimism", ("+", 1), ("*", "$prod"))))),
    T("FETCH_ELASTICITY_BELIEF", "$elast", rd_agent(F, "estimated_revenue_elasticity"),
      case(c("$elast", "!=", None), "FETCH_CURRENT_STATE", upd(adjusted_elasticity=pipe("$optimism", ("+", 1), ("*", "$elast"), ("*", 10000))))),
    T("FETCH_CURRENT_STATE", "$cash_sra", rd_agent(S, "cash"), case(c("$cash_sra", "!=", None), "APPLY_DECISION_RULE")),
    T("APPLY_DECISION_RULE", "$decision_opt", prob("founder_optimism"),
      case(c("$decision_opt", ">", 0.7), "SPLIT_BUDGET",
           upd(total_budget=pipe("$cash_sra", ("*", 0.2)),
               hiring_share=pipe("$adjusted_productivity", ("/", pipe("$adjusted_productivity", ("+", "$adjusted_elasticity")))))),
      case(c("$decision_opt", "<=", 0.7), "SPLIT_BUDGET", upd(total_budget=pipe("$cash_sra", ("*", 0.1)), hiring_share=0.5))),
    always("SPLIT_BUDGET", "SET_HIRING_BUDGET",
           upd(hiring_budget_val=pipe("$total_budget", ("*", "$hiring_share"))),
           upd(other_budget_val=pipe("$total_budget", ("-", "$hiring_budget_val")))),
    always("SET_HIRING_BUDGET", "SET_OTHER_BUDGET", w_agent("guardar_hiring", F, "hiring_budget", "$hiring_budget_val")),
    always("SET_OTHER_BUDGET", "MARK_DECISION_TICK", w_agent("guardar_other", F, "other_budget", "$other_budget_val")),
    T("MARK_DECISION_TICK", "$tick", rd_agent(F, "last_decision_tick"),
      case(c("$tick", "!=", None), "DONE", w_agent("guardar_tick", F, "last_decision_tick", pipe("$tick", ("+", 1))))),
))

# ================================================================ belief_updater (Founder)
automata.append(A("belief_updater", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_CURRENT_BELIEFS"),
    T("FETCH_CURRENT_BELIEFS", "$curr_prod", rd_agent(F, "estimated_productivity_per_head"), case(c("$curr_prod", "!=", None), "FETCH_CURRENT_ELASTICITY")),
    T("FETCH_CURRENT_ELASTICITY", "$curr_elast", rd_agent(F, "estimated_revenue_elasticity"), case(c("$curr_elast", "!=", None), "FETCH_TRUE_PRODUCTIVITY")),
    T("FETCH_TRUE_PRODUCTIVITY", "$true_prod", rd_agent(TM, "true_productivity_per_head"), case(c("$true_prod", "!=", None), "FETCH_TRUE_ELASTICITY")),
    T("FETCH_TRUE_ELASTICITY", "$true_elast", rd_agent(TM, "true_revenue_elasticity"), case(c("$true_elast", "!=", None), "FETCH_NOISE")),
    T("FETCH_NOISE", "$noise_val", prob("market_noise"),
      case(c("$noise_val", "!=", None), "UPDATE_BELIEFS",
           upd(prod_scaled=pipe("$noise_val", ("+", 1), ("*", "$curr_prod")),
               elast_scaled=pipe("$noise_val", ("+", 1), ("*", "$curr_elast"))))),
    always("UPDATE_BELIEFS", "SET_PRODUCTIVITY_BELIEF",
           upd(new_prod=pipe("$true_prod", ("-", "$prod_scaled"), ("*", 0.5), ("+", "$prod_scaled"), ("max", 0)),
               new_elast=pipe("$true_elast", ("-", "$elast_scaled"), ("*", 0.5), ("+", "$elast_scaled"), ("max", 0), ("min", 1)))),
    always("SET_PRODUCTIVITY_BELIEF", "APPLY_ELASTICITY_UPDATE", w_agent("actualizar_productividad", F, "estimated_productivity_per_head", "$new_prod")),
    always("APPLY_ELASTICITY_UPDATE", "DONE", w_agent("actualizar_elasticidad", F, "estimated_revenue_elasticity", "$new_elast")),
))

# ================================================================ headcount_controller (TalentMarket)
automata.append(A("headcount_controller", "LOAD_PARAMS", ["HIRING_DONE", "LAYOFFS_DONE", "FREEZE_SET", "NO_ACTION"],
    always("LOAD_PARAMS", "IDLE"),
    T("IDLE", "$cash_hc", rd_agent(S, "cash"),
      case(c("$cash_hc", "<=", 200000), "PANIC_GATE"),
      case(c("$cash_hc", ">", 200000), "SHOCK_GATE")),
    fixed(T("PANIC_GATE", "$panic", prob("market_noise"),
      case(c("$panic", "<", -0.02), "DO_LAYOFFS"),
      case(c("$panic", ">=", -0.02), "DO_FREEZE"))),
    T("SHOCK_GATE", "$shock_hc", prob("shock_arrival"),
      case(c("$shock_hc", ">", 0.5), "SEVERITY_GATE"),
      case(c("$shock_hc", "<=", 0.5), "DO_HIRING")),
    fixed(T("SEVERITY_GATE", "$severity_gate", prob("shock_impact"),
      case(c("$severity_gate", ">", 0.35), "DO_LAYOFFS"),
      case(c("$severity_gate", "<=", 0.35), "DO_FREEZE"))),
    T("DO_HIRING", "$budget_h", rd_agent(F, "hiring_budget"),
      case(c("$budget_h", ">", 0), "CHECK_HIRING_SUCCESS"),
      case(c("$budget_h", "<=", 0), "NO_ACTION")),
    T("CHECK_HIRING_SUCCESS", "$hiring_success", prob("hiring_success"),
      case(c("$hiring_success", ">", 0.4), "APPLY_HIRING", upd(success_factor=1.0)),
      case(c("$hiring_success", "<=", 0.4), "APPLY_HIRING", upd(success_factor=0.5))),
    T("APPLY_HIRING", "$cost_per_head", rd_agent(S, "cost_variable_per_head"),
      case(c("$cost_per_head", "!=", None), "CALC_NEW_HEADCOUNT",
           upd(new_heads=pipe("$budget_h", ("*", "$success_factor"), ("/", "$cost_per_head"), ("floor", 0))))),
    T("CALC_NEW_HEADCOUNT", "$current_hc", rd_agent(S, "headcount"),
      case(c("$current_hc", "!=", None), "HIRING_DONE",
           w_agent("ejecutar_contratacion", S, "headcount", pipe("$current_hc", ("+", "$new_heads"))),
           w_agent("gastar_presupuesto", F, "hiring_budget", 0))),
    always("DO_FREEZE", "FREEZE_SET", w_agent("congelar_vacantes", S, "cost_rigidity", 0.6)),
    T("DO_LAYOFFS", "$hc_layoffs", rd_agent(S, "headcount"), case(c("$hc_layoffs", "!=", None), "APPLY_LAYOFFS")),
    T("APPLY_LAYOFFS", "$severity", prob("layoff_severity"),
      case(c("$severity", ">", 0.6), "LAYOFFS_DONE",
           w_agent("despido_severo", S, "headcount", pipe(1, ("-", "$severity"), ("*", "$hc_layoffs"), ("floor", 0), ("max", 1)))),
      case(c("$severity", "<=", 0.6), "LAYOFFS_DONE",
           w_agent("despido_leve", S, "headcount", pipe("$hc_layoffs", ("-", 1), ("max", 1))))),
))

# ================================================================ runway_critical (Startup)
automata.append(A("runway_critical", "ASSESS_SURVIVAL_CHANCE", ["LAYOFFS_EXECUTED", "COSTS_FROZEN"],
    T("ASSESS_SURVIVAL_CHANCE", "$panic_index", prob("layoff_severity"),
      case(c("$panic_index", ">", 0.6), "STOCHASTIC_INVESTOR_INTERVENTION"),
      case(c("$panic_index", "<=", 0.6), "STOCHASTIC_RIGIDITY_RELAX")),
    fixed(T("STOCHASTIC_INVESTOR_INTERVENTION", "$appetite", prob("investor_appetite"),
      case(c("$appetite", ">=", 0.42), "BOARD_DECISION_SPLIT"),
      case(c("$appetite", "<", 0.42), "FORCE_EMERGENCY_LAYOFFS"))),
    fixed(T("BOARD_DECISION_SPLIT", "$board_traction", prob("revenue_volatility"),
      case(c("$board_traction", ">=", 1.0), "CRITICAL_FREEZE"),
      case([c("$board_traction", ">=", 0.9), c("$board_traction", "<", 1.0)], "RESERVES_CHECK"),
      case(c("$board_traction", "<", 0.9), "FORCE_EMERGENCY_LAYOFFS"))),
    T("RESERVES_CHECK", "$cash_rc", rd_agent(S, "cash"),
      case(c("$cash_rc", ">", 250000), "CRITICAL_FREEZE"),
      case(c("$cash_rc", "<=", 250000), "FORCE_EMERGENCY_LAYOFFS")),
    T("STOCHASTIC_RIGIDITY_RELAX", "$elasticity", prob("true_elasticity"),
      case(c("$elasticity", "<", 0.5), "LABOR_RIGIDITY_TEST"),
      case(c("$elasticity", ">=", 0.5), "CRITICAL_FREEZE")),
    T("LABOR_RIGIDITY_TEST", "$noise_burn", prob("market_noise"),
      case(c("$noise_burn", "<", 0.0), "FORCE_EMERGENCY_LAYOFFS"),
      case(c("$noise_burn", ">=", 0.0), "CRITICAL_FREEZE")),
    always("FORCE_EMERGENCY_LAYOFFS", "LAYOFFS_EXECUTED",
           w_agent("vaciar_presupuesto", F, "hiring_budget", 0), emit_to("headcount_controller", TM)),
    always("CRITICAL_FREEZE", "COSTS_FROZEN", w_agent("congelar_estructura", S, "cost_rigidity", 0.95)),
))

# ================================================================ investment_decision (Investor)
automata.append(A("investment_decision", "LOAD_PARAMS", ["INVESTED", "PASSED"],
    always("LOAD_PARAMS", "WATCHING",
           upd(investor_id=event_agent(), env_id=ENV, agent_b_id=S), upd(agent_a_id="$investor_id")),
    T("WATCHING", "$engaged", rd_agent("$investor_id", "is_engaged"),
      case(c("$engaged", "==", True), "STOCHASTIC_SENTIMENT_GATE"),
      case(c("$engaged", "!=", True), "COLD_INTEREST")),
    fixed(T("COLD_INTEREST", "$cold", prob("market_noise"),
      case(c("$cold", ">=", -0.03), "STOCHASTIC_SENTIMENT_GATE"),
      case(c("$cold", "<", -0.03), "CHECK_RELATION"))),
    fixed(T("STOCHASTIC_SENTIMENT_GATE", "$market_sentiment", prob("market_sentiment"),
      case(c("$market_sentiment", ">", 0.4), "APPETITE_GATE"),
      case(c("$market_sentiment", "<=", 0.4), "TRACTION_CHECK"))),
    fixed(T("TRACTION_CHECK", "$traction", prob("revenue_volatility"),
      case(c("$traction", ">=", 1.0), "APPETITE_GATE"),
      case(c("$traction", "<", 1.0), "CHECK_RELATION"))),
    fixed(T("APPETITE_GATE", "$appetite", prob("investor_appetite"),
      case(c("$appetite", ">", 0.7), "FETCH_AMOUNT"),
      case([c("$appetite", ">", 0.4), c("$appetite", "<=", 0.7)], "RUNWAY_CHECK"),
      case(c("$appetite", "<=", 0.4), "SECOND_LOOK"))),
    T("RUNWAY_CHECK", "$runway_id", rd_agent(S, "runway"),
      case(c("$runway_id", ">=", 6), "FETCH_AMOUNT"),
      case(c("$runway_id", "<", 6), "SECOND_LOOK")),
    fixed(T("SECOND_LOOK", "$second_look", prob("market_noise"),
      case(c("$second_look", ">=", -0.02), "FETCH_AMOUNT"),
      case(c("$second_look", "<", -0.02), "CHECK_RELATION"))),
    T("FETCH_AMOUNT", "$amount", prob("investment_size"), case(c("$amount", "!=", None), "CHECK_CAPACITY")),
    T("CHECK_CAPACITY", "$capacity", rd_agent("$investor_id", "investment_capacity"),
      case(c("$capacity", "!=", None), "COMPARE_CAPACITY")),
    T("COMPARE_CAPACITY", "$capacity_margin", pipe("$capacity", ("-", "$amount")),
      case(c("$capacity_margin", ">=", 0), "STOCHASTIC_OPTIMISM_FILTER"),
      case(c("$capacity_margin", "<", 0), "CHECK_RELATION")),
    fixed(T("STOCHASTIC_OPTIMISM_FILTER", "$bias", prob("founder_optimism"),
      case(c("$bias", ">=", 0.45), "DUE_DILIGENCE"),
      case(c("$bias", "<", 0.45), "TERM_NEGOTIATION"))),
    fixed(T("DUE_DILIGENCE", "$diligence", prob("cost_inflation"),
      case(c("$diligence", "<=", 1.05), "MAKE_OFFER"),
      case(c("$diligence", ">", 1.05), "CHECK_RELATION"))),
    fixed(T("TERM_NEGOTIATION", "$terms", prob("cost_inflation"),
      case(c("$terms", "<=", 1.08), "MAKE_OFFER"),
      case(c("$terms", ">", 1.08), "CHECK_RELATION"))),
    T("MAKE_OFFER", "$cash_id", rd_agent(S, "cash"),
      case(c("$cash_id", "!=", None), "PREPARE_INVESTMENT_EMIT", w_agent("inyectar_capital", S, "cash", pipe("$cash_id", ("+", "$amount"))))),
    always("PREPARE_INVESTMENT_EMIT", "INVESTED",
           w_agent("marcar_inversion", "$investor_id", "has_invested", True), emit_to("investment_offer", S)),
    T("CHECK_RELATION", "$rel", rd_rel(),
      case(c("$rel", "==", True), "RELATION_STRAIN"),
      case(c("$rel", "!=", True), "PASSED")),
    T("RELATION_STRAIN", "$strain", prob("market_sentiment"),
      case(c("$strain", "<=", 0.3), "PASSED", remove_rel("romper_relacion")),
      case(c("$strain", ">", 0.3), "PASSED")),
))

# ================================================================ investor_exit (Investor)
automata.append(A("investor_exit", "LOAD_PARAMS", ["EXITED", "REMAINED"],
    always("LOAD_PARAMS", "CHECK_PATIENCE",
           upd(investor_id=event_agent(), env_id=ENV, agent_b_id=S), upd(agent_a_id="$investor_id")),
    fixed(T("CHECK_PATIENCE", "$patience_raw", prob("market_noise"),
      case(c("$patience_raw", "<", -0.01), "CHECK_INVESTMENT_STATUS"),
      case(c("$patience_raw", ">=", -0.01), "REMAINED"))),
    T("CHECK_INVESTMENT_STATUS", "$invested", rd_agent("$investor_id", "has_invested"),
      case(c("$invested", "==", True), "DETERMINE_EXIT_REASON"),
      case(c("$invested", "!=", True), "DETERMINE_EXIT_REASON")),
    T("DETERMINE_EXIT_REASON", "$exit_reason", prob("investor_exit_reason"),
      case(c("$exit_reason", "==", "patience_exhausted"), "BREAK_RELATION"),
      case(c("$exit_reason", "==", "regulatory_risk"), "REGULATORY_REVIEW"),
      case(c("$exit_reason", "==", "market_downturn"), "MARKET_REVIEW")),
    T("REGULATORY_REVIEW", "$reg_inflation", prob("cost_inflation"),
      case(c("$reg_inflation", ">", 1.05), "BREAK_RELATION"),
      case(c("$reg_inflation", "<=", 1.05), "REMAINED")),
    T("MARKET_REVIEW", "$downturn", prob("market_noise"),
      case(c("$downturn", "<", -0.02), "BREAK_RELATION"),
      case([c("$downturn", ">=", -0.02), c("$downturn", "<", 0.0)], "RUNWAY_REVIEW"),
      case(c("$downturn", ">=", 0.0), "REMAINED")),
    T("RUNWAY_REVIEW", "$runway_ie", rd_agent(S, "runway"),
      case(c("$runway_ie", "<", 6), "BREAK_RELATION"),
      case(c("$runway_ie", ">=", 6), "REMAINED")),
    T("BREAK_RELATION", "$rel_exists", rd_rel(),
      case(c("$rel_exists", "==", True), "REMOVE_INVESTOR_RELATION"),
      case(c("$rel_exists", "!=", True), "EXITED", w_agent("desvincular", "$investor_id", "is_engaged", False))),
    always("REMOVE_INVESTOR_RELATION", "EXITED",
           remove_rel("remover_relacion"), w_agent("desvincular_tras_ruptura", "$investor_id", "is_engaged", False)),
))

# ================================================================ seeking_funding (Founder)
automata.append(A("seeking_funding", "ACTIVATE", ["DONE"],
    always("ACTIVATE", "DONE", *[w_agent(f"activar_inversor_{i}", f"Investor_{i}", "is_engaged", True) for i in range(1, 6)]),
))

# ================================================================ investment_offer (Startup)
automata.append(A("investment_offer", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "PROCESS_OFFER"),
    always("PROCESS_OFFER", "DONE", emit_to("runway_lifecycle", S)),
))

# ================================================================ exogenous_shock (Startup, ShockEvent)
automata.append(A("exogenous_shock", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_ENV_PRESSURE", upd(env_id=ENV)),
    T("FETCH_ENV_PRESSURE", "$pressure", rd_env("regulatory_pressure"),
      case(c("$pressure", ">", 0.05), "STOCHASTIC_MARKET_NOISE", upd(raw_impact="$pressure")),
      case(c("$pressure", "<=", 0.05), "STOCHASTIC_MARKET_NOISE", upd(raw_impact=0.05))),
    T("STOCHASTIC_MARKET_NOISE", "$m_noise", prob("market_noise"),
      case(c("$m_noise", "<", 0.01), "FETCH_STARTUP_CASH"),
      case(c("$m_noise", ">=", 0.01), "DONE")),
    T("FETCH_STARTUP_CASH", "$cash_val", rd_agent(S, "cash"), case(c("$cash_val", "!=", None), "CALCULATE_STOCHASTIC_FACTOR")),
    T("CALCULATE_STOCHASTIC_FACTOR", "$stochastic_noise", prob("revenue_volatility"), case(c("$stochastic_noise", "!=", None), "STOCHASTIC_SHOCK_MULTIPLIER")),
    T("STOCHASTIC_SHOCK_MULTIPLIER", "$shock_roll", prob("shock_impact"),
      case(c("$shock_roll", ">", 0.3), "APPLY_CASH_DRAIN"),
      case(c("$shock_roll", "<=", 0.3), "DONE")),
    always("APPLY_CASH_DRAIN", "DONE",
           upd(drain_factor=pipe(1, ("-", pipe("$raw_impact", ("*", "$stochastic_noise"), ("*", 0.4))))),
           w_agent("drenar_caja", S, "cash", pipe("$cash_val", ("*", "$drain_factor")))),
))

# ================================================================ regulatory_shock (Startup, ShockEvent)
automata.append(A("regulatory_shock", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_ENV_PRESSURE", upd(env_id=ENV)),
    T("FETCH_ENV_PRESSURE", "$pressure", rd_env("regulatory_pressure"),
      case(c("$pressure", ">", 0.05), "STOCHASTIC_INFLATION_CHECK", upd(raw_impact="$pressure")),
      case(c("$pressure", "<=", 0.05), "STOCHASTIC_INFLATION_CHECK", upd(raw_impact=0.05))),
    T("STOCHASTIC_INFLATION_CHECK", "$macro_inflation", prob("cost_inflation"),
      case(c("$macro_inflation", ">", 1.03), "FETCH_STARTUP_COSTS"),
      case(c("$macro_inflation", "<=", 1.03), "DONE")),
    T("FETCH_STARTUP_COSTS", "$cost_val", rd_agent(S, "cost_fixed"), case(c("$cost_val", "!=", None), "APPLY_NEW_COSTS")),
    always("APPLY_NEW_COSTS", "DONE",
           w_agent("aumentar_costo_fijo", S, "cost_fixed",
                   pipe("$raw_impact", ("*", "$macro_inflation"), ("*", 0.1), ("+", 1), ("*", "$cost_val")))),
))

# ================================================================ shock_lifecycle (ShockEvent)
CASCADES = ["MARKET_CASCADE", "REGULATORY_CASCADE", "COMPOUND_CASCADE"]
automata.append(A("shock_lifecycle", "LOAD_PARAMS", ["QUIET", "ABSORBED"] + CASCADES,
    always("LOAD_PARAMS", "DORMANT", upd(env_id=ENV)),
    T("DORMANT", "$arrival", prob("shock_arrival"),
      case(c("$arrival", ">", 0.5), "FETCH_IMPACT"),
      case(c("$arrival", "<=", 0.5), "QUIET")),
    T("FETCH_IMPACT", "$impact", prob("shock_impact"),
      case(c("$impact", ">", 0.25), "CLASSIFY_TYPE",
           w_env("guardar_presion", "regulatory_pressure", "$impact"), w_env("guardar_condicion", "market_condition", "$impact")),
      case([c("$impact", ">", 0.1), c("$impact", "<=", 0.25)], "PRESSURE_AMPLIFIER"),
      case(c("$impact", "<=", 0.1), "ABSORBED")),
    T("PRESSURE_AMPLIFIER", "$pressure_sl", rd_env("regulatory_pressure"),
      case(c("$pressure_sl", ">", 0.4), "CLASSIFY_TYPE",
           w_env("amplificar_presion", "regulatory_pressure", "$impact"), w_env("amplificar_condicion", "market_condition", "$impact")),
      case(c("$pressure_sl", "<=", 0.4), "ABSORBED")),
    T("CLASSIFY_TYPE", "$shock_type", prob("shock_type"),
      case(c("$shock_type", "==", "market"), "MARKET_CASCADE", emit_to("exogenous_shock", S)),
      case(c("$shock_type", "==", "regulatory"), "REGULATORY_CASCADE", emit_to("regulatory_shock", S)),
      case(c("$shock_type", "==", "compound"), "COMPOUND_CASCADE", emit_to("exogenous_shock", S), emit_to("regulatory_shock", S))),
))

# ================================================================ compliance_monitor (Founder)
automata.append(A("compliance_monitor", "LOAD_PARAMS", ["NO_PRESSURE", "RELIEVED", "PENALIZED"],
    always("LOAD_PARAMS", "READ_REGULATOR_CHANNEL", upd(env_id=ENV)),
    T("READ_REGULATOR_CHANNEL", "$channel_pressure", rd_env("regulatory_pressure"),
      case(c("$channel_pressure", ">", 0.15), "TAX_BURDEN_VALUATION"),
      case(c("$channel_pressure", "<=", 0.15), "LOW_PRESSURE_AUDIT")),
    fixed(T("LOW_PRESSURE_AUDIT", "$audit_mood", prob("cost_inflation"),
      case(c("$audit_mood", "<=", 1.05), "NO_PRESSURE"),
      case(c("$audit_mood", ">", 1.05), "TAX_BURDEN_VALUATION"))),
    T("TAX_BURDEN_VALUATION", "$tax_variance", prob("revenue_volatility"),
      case(c("$tax_variance", ">", 1.05), "PENALIZE_OPERATIONS"),
      case(c("$tax_variance", "<=", 1.05), "STOCHASTIC_INFLATION_STRESS")),
    T("STOCHASTIC_INFLATION_STRESS", "$inflation_roll", prob("cost_inflation"),
      case(c("$inflation_roll", ">", 1.02), "PENALIZE_OPERATIONS"),
      case(c("$inflation_roll", "<=", 1.02), "EVALUATE_BURDEN_FATIGUE")),
    T("EVALUATE_BURDEN_FATIGUE", "$bureaucracy_shock", prob("market_noise"),
      case(c("$bureaucracy_shock", ">", 0.005), "PENALIZE_OPERATIONS"),
      case(c("$bureaucracy_shock", "<=", 0.005), "RELIEVED",
           w_env("aliviar_presion", "regulatory_pressure", pipe("$channel_pressure", ("*", 0.5))))),
    T("PENALIZE_OPERATIONS", "$current_budget", rd_agent(F, "other_budget"),
      case(c("$current_budget", ">", 0), "STOCHASTIC_AUDIT_ESCAPE",
           w_agent("reducir_eficiencia", F, "other_budget", pipe("$current_budget", ("*", 0.9)))),
      case(c("$current_budget", "<=", 0), "STOCHASTIC_AUDIT_ESCAPE")),
    T("STOCHASTIC_AUDIT_ESCAPE", "$shock_luck", prob("shock_impact"),
      case(c("$shock_luck", ">", 0.25), "AUDIT_LUCK_MITIGATION"),
      case(c("$shock_luck", "<=", 0.25), "PENALIZED")),
    T("AUDIT_LUCK_MITIGATION", "$sentiment_defense", prob("market_sentiment"),
      case(c("$sentiment_defense", ">", 0.66), "NO_PRESSURE"),
      case(c("$sentiment_defense", "<=", 0.66), "PENALIZED")),
))

# ================================================================ client_retention_cycle (KeyClient)
automata.append(A("client_retention_cycle", "LOAD_PARAMS", ["CHURNED", "RENEWED", "INACTIVE"],
    always("LOAD_PARAMS", "CHECK_ACTIVE", upd(client_id=event_agent())),
    T("CHECK_ACTIVE", "$is_active", rd_agent("$client_id", "is_active"),
      case(c("$is_active", "==", True), "ACTIVE"),
      case(c("$is_active", "!=", True), "INACTIVE")),
    T("ACTIVE", "$months", rd_agent("$client_id", "contract_months_remaining"),
      case(c("$months", ">", 1), "DECREMENT_CONTRACT"),
      case(c("$months", "<=", 1), "STOCHASTIC_RETENTION_FILTER")),
    always("DECREMENT_CONTRACT", "RENEWED", w_agent("descontar_mes", "$client_id", "contract_months_remaining", pipe("$months", ("-", 1)))),
    T("STOCHASTIC_RETENTION_FILTER", "$profile_check", prob("volatility_profile"),
      case(c("$profile_check", "<=", 0.65), "EVALUATE_STOCHASTIC_RETENTION"),
      case(c("$profile_check", ">", 0.65), "WIN_BACK")),
    T("EVALUATE_STOCHASTIC_RETENTION", "$retention_roll", prob("client_retention"),
      case(c("$retention_roll", ">=", 0.5), "EXECUTE_RENEWAL"),
      case([c("$retention_roll", ">=", 0.35), c("$retention_roll", "<", 0.5)], "CHECK_SUCCESS_BUDGET"),
      case(c("$retention_roll", "<", 0.35), "WIN_BACK")),
    T("CHECK_SUCCESS_BUDGET", "$success_budget", rd_agent(F, "other_budget"),
      case(c("$success_budget", ">", 0), "EXECUTE_RENEWAL"),
      case(c("$success_budget", "<=", 0), "WIN_BACK")),
    fixed(T("WIN_BACK", "$win_back", prob("market_noise"),
      case(c("$win_back", ">=", 0.01), "EXECUTE_RENEWAL"),
      case(c("$win_back", "<", 0.01), "PREPARE_STARTUP_READ"))),
    always("EXECUTE_RENEWAL", "RENEWED", w_agent("renovar_contrato", "$client_id", "contract_months_remaining", 6)),
    T("PREPARE_STARTUP_READ", "$old_rev", rd_agent(S, "revenue_current"), case(c("$old_rev", "!=", None), "CALC_PUNISHMENT")),
    T("CALC_PUNISHMENT", "$stochastic_loss", prob("revenue_volatility"),
      case(c("$stochastic_loss", "!=", None), "APPLY_PUNISHMENT",
           upd(new_rev=pipe("$stochastic_loss", ("*", -0.15), ("+", 1), ("*", "$old_rev"))))),
    always("APPLY_PUNISHMENT", "CHURN_UPDATE_FLAG", w_agent("castigar_startup", S, "revenue_current", "$new_rev")),
    always("CHURN_UPDATE_FLAG", "PREPARE_CHURN_EMIT", w_agent("apagar_cliente", "$client_id", "is_active", False)),
    always("PREPARE_CHURN_EMIT", "CHURNED", emit_to("client_replacement", F)),
))

# ================================================================ client_replacement (Founder)
automata.append(A("client_replacement", "LOAD_PARAMS", ["REPLACED", "FRICTION_ABANDONMENT", "FAILED_MARKET_SENTIMENT", "FAILED_CLIENT_CREATION"],
    always("LOAD_PARAMS", "CHECK_FOUNDER_BUDGET", upd(env_id=ENV)),
    T("CHECK_FOUNDER_BUDGET", "$acq_budget", rd_agent(F, "other_budget"),
      case(c("$acq_budget", ">", 0), "PAID_CHANNEL", w_agent("gastar_campana", F, "other_budget", 0)),
      case(c("$acq_budget", "<=", 0), "ORGANIC_CHANNEL")),
    fixed(T("PAID_CHANNEL", "$conversion", prob("revenue_volatility"),
      case(c("$conversion", ">=", 0.9), "CREATE_NEW_CLIENT"),
      case(c("$conversion", "<", 0.9), "FRICTION_ABANDONMENT"))),
    T("ORGANIC_CHANNEL", "$channel_raw", prob("client_acquisition_channel"),
      case(c("$channel_raw", "==", "referral"), "CREATE_NEW_CLIENT"),
      case(c("$channel_raw", "==", "organic"), "FRICTION_ABANDONMENT"),
      case(c("$channel_raw", "==", "paid"), "FRICTION_ABANDONMENT")),
    fixed(T("CREATE_NEW_CLIENT", "$sentiment_cr", prob("market_sentiment"),
      case(c("$sentiment_cr", ">", 0.6), "LEGAL_COMPLIANCE_BARRIER"),
      case([c("$sentiment_cr", ">", 0.4), c("$sentiment_cr", "<=", 0.6)], "CHECK_MARKET_CONDITION"),
      case(c("$sentiment_cr", "<=", 0.4), "SECOND_CHANCE"))),
    fixed(T("SECOND_CHANCE", "$second_chance", prob("cost_inflation"),
      case(c("$second_chance", "<=", 1.03), "CHECK_MARKET_CONDITION"),
      case(c("$second_chance", ">", 1.03), "FAILED_MARKET_SENTIMENT"))),
    T("CHECK_MARKET_CONDITION", "$market_cr", rd_env("market_condition"),
      case(c("$market_cr", "<=", 0.28), "LEGAL_COMPLIANCE_BARRIER"),
      case(c("$market_cr", ">", 0.28), "STOCHASTIC_VOLATILITY_GATE")),
    fixed(T("STOCHASTIC_VOLATILITY_GATE", "$profile", prob("market_noise"),
      case(c("$profile", ">=", -0.03), "LEGAL_COMPLIANCE_BARRIER"),
      case(c("$profile", "<", -0.03), "FAILED_MARKET_SENTIMENT"))),
    fixed(T("LEGAL_COMPLIANCE_BARRIER", "$inflation_gate", prob("cost_inflation"),
      case(c("$inflation_gate", ">", 1.08), "FAILED_CLIENT_CREATION"),
      case(c("$inflation_gate", "<=", 1.08), "ADD_TO_ECOSYSTEM"))),
    always("ADD_TO_ECOSYSTEM", "REGISTER_IN_ENV",
           upd(agent_type="KeyClient"), req("new_client", {"target": "agents", "method": "add_agent", "params": ["$agent_type"]})),
    always("REGISTER_IN_ENV", "REPLACED",
           upd(agent_id="$new_client"), req("registrar_miembro", {"target": "environments", "method": "write_member", "params": ["$env_id", "$agent_id"]})),
))

# ================================================================ regulatory_intervention (Regulator)
automata.append(A("regulatory_intervention", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "MONITOR", upd(env_id=ENV)),
    T("MONITOR", "$shock_ri", prob("shock_arrival"),
      case(c("$shock_ri", ">", 0.5), "ASSESS_MARKET"),
      case(c("$shock_ri", "<=", 0.5), "DONE")),
    T("ASSESS_MARKET", "$market_val", rd_env("market_condition"), case(c("$market_val", "!=", None), "DETERMINE_PRESSURE")),
    T("DETERMINE_PRESSURE", "$noise_ri", prob("market_noise"),
      case(c("$noise_ri", ">", 0.0), "APPLY_REGULATION",
           upd(new_pressure=pipe("$market_val", ("+", "$noise_ri"), ("+", 0.1)), new_compliance=pipe(1.0, ("+", "$noise_ri"), ("+", 0.1)), new_tax=0.2)),
      case(c("$noise_ri", "<=", 0.0), "APPLY_REGULATION",
           upd(new_pressure="$market_val", new_compliance=1.0, new_tax=0.15))),
    always("APPLY_REGULATION", "WRITE_ENV_PARAM_2", w_env("registrar_presion", "regulatory_pressure", "$new_pressure")),
    always("WRITE_ENV_PARAM_2", "WRITE_ENV_PARAM_3", w_env("registrar_compliance", "compliance_cost_index", "$new_compliance")),
    always("WRITE_ENV_PARAM_3", "DONE", w_env("registrar_tax", "tax_rate", "$new_tax")),
))

# ================================================================ compliance_burden (Startup)
automata.append(A("compliance_burden", "LOAD_PARAMS", ["DONE"],
    always("LOAD_PARAMS", "FETCH_COMPLIANCE_COST", upd(env_id=ENV)),
    T("FETCH_COMPLIANCE_COST", "$compliance_idx", rd_env("compliance_cost_index"), case(c("$compliance_idx", "!=", None), "FETCH_REGULATORY_PRESSURE")),
    T("FETCH_REGULATORY_PRESSURE", "$pressure_cb", rd_env("regulatory_pressure"), case(c("$pressure_cb", "!=", None), "FETCH_CASH")),
    T("FETCH_CASH", "$cash_cb", rd_agent(S, "cash"),
      case(c("$cash_cb", "!=", None), "DONE",
           w_agent("cobrar_cumplimiento", S, "cash",
                   pipe("$cash_cb", ("-", pipe("$pressure_cb", ("+", 1), ("*", "$compliance_idx"), ("*", 10000))))))),
))

# ================================================================ runway_warning (Startup)
automata.append(A("runway_warning", "EVALUATE_POISSON_ARRIVAL", ["DONE", "FUNDING_FAILED"],
    T("EVALUATE_POISSON_ARRIVAL", "$poisson_click", prob("shock_arrival"),
      case(c("$poisson_click", "<=", 0.5), "TRIGGER_FUNDING_EFFORT"),
      case(c("$poisson_click", ">", 0.5), "FUNDING_FAILED")),
    always("TRIGGER_FUNDING_EFFORT", "DONE", emit_to("seeking_funding", F)),
))

assert len(automata) == 22, len(automata)


# ================================================================ calibración de umbrales binarios
def calibrate(auts):
    from core.distributions import Distributions
    dists = [Distributions(json.loads((CFG / f"distributions_{r}.json").read_text()), None, seed=0) for r in REGIMES]
    import numpy as np
    changes = []
    for a in auts:
        for t in a["transitions"]:
            var, tv = next(iter(t["threshold_value"].items()))
            if not (isinstance(tv, dict) and tv.get("type") == "probabilistic") or t.get("__fixed__"):
                continue
            name = tv["distribution"]
            family = dists[0].samplers[name]["family"]
            ths = t["thresholds"]
            if family in ("categorical", "poisson") or len(ths) != 2:
                continue
            if any(len(th["threshold_case"]) != 1 for th in ths):
                continue
            c0, c1 = ths[0]["threshold_case"][0], ths[1]["threshold_case"][0]
            if c0["value"] != c1["value"] or not isinstance(c0["value"], (int, float)) or isinstance(c0["value"], bool):
                continue
            t0 = c0["value"]

            def score(x):
                # (regímenes con ambas ramas >= FLOOR, masa mínima entre esos regímenes)
                sides = []
                for d in dists:
                    m = d.probability_interval(name, float("-inf"), x)
                    sides.append(min(m, 1 - m))
                good = [s for s in sides if s >= FLOOR]
                return (len(good), round(min(good), 4) if good else 0.0)

            original = score(t0)
            if original[0] == len(dists):
                continue
            lows = [d.samplers[name]["truncation"]["min"] for d in dists if d.samplers[name]["truncation"]]
            highs = [d.samplers[name]["truncation"]["max"] for d in dists if d.samplers[name]["truncation"]]
            grid = np.linspace(min(lows), max(highs), 1401)[1:-1]
            best = max(grid, key=score)
            new = float(round(best, 3))
            if score(new) <= original:
                continue
            for th in ths:
                th["threshold_case"][0]["value"] = new
            changes.append((a["automaton_name"], t["from"], name, t0, new, original, score(new)))
    return changes


def band_report(auts):
    from core.distributions import Distributions
    from patl.patl_verifier import PATLVerifier
    dists = {r: Distributions(json.loads((CFG / f"distributions_{r}.json").read_text()), None, seed=0) for r in REGIMES}
    rows = []
    for a in auts:
        for t in a["transitions"]:
            var, tv = next(iter(t["threshold_value"].items()))
            if not (isinstance(tv, dict) and tv.get("type") == "probabilistic") or len(t["thresholds"]) < 2:
                continue
            name = tv["distribution"]
            per = {}
            for r, d in dists.items():
                e = d.samplers[name]
                masses = []
                for th in t["thresholds"]:
                    if e["family"] == "categorical":
                        masses.append(sum(p for l, p in zip(e["labels"], e["probabilities"]) if th["threshold_case"][0]["value"] == l))
                    elif e["family"] == "poisson":
                        iv = PATLVerifier._interval(var, th["threshold_case"], {})
                        sup = PATLVerifier._poisson_support(e)
                        masses.append(sum(p for v, p in sup if iv and (iv[0] < v or (iv[2] and iv[0] == v)) and (v < iv[1] or (iv[3] and v == iv[1]))))
                    else:
                        iv = PATLVerifier._interval(var, th["threshold_case"], {})
                        masses.append(d.probability_interval(name, *iv) if iv else 0.0)
                per[r] = masses
            rows.append((a["automaton_name"], t["from"], name, {r: [round(m, 4) for m in ms] for r, ms in per.items()},
                         min(min(ms) for ms in per.values())))
    return rows


changes = calibrate(automata)

# ================================================================ PATL
def grp(agent_type, automata_targets, max_agents=1):
    return [{"agent_type": agent_type, "max_agents": max_agents,
             "automata": [({"automaton_name": a, "target_states": t} if t is not None else {"automaton_name": a}) for a, t in automata_targets]}]
def P(pid, ptype, bound, op, coalition, adversaries, depth=2, extra=None):
    d = {"predicate_id": pid, "type": ptype, "max_depth": depth, "coalition_quantifier": "exists",
         "coalition": coalition, "adversaries": adversaries, "probability_bound": bound, "probability_operator": op}
    d.update(extra or {})
    return d

observations = [
    ("shock_lifecycle", "DORMANT", [
        P("PR_STOCHASTIC_MORTALITY_VIA_DRIFT", "reachability", 0.5, "<=",
          grp("Startup", [("runway_lifecycle", ["DEAD"])]),
          grp("ShockEvent", [("shock_lifecycle", None), ("exogenous_shock", None)]), depth=1),
        P("PR_STOCHASTIC_SHOCK_CASCADE_VIA_ARRIVAL", "reachability", 0.1, ">=",
          grp("ShockEvent", [("shock_lifecycle", CASCADES)]),
          grp("Founder", [("compliance_monitor", None)])),
    ]),
    ("runway_lifecycle", "EVALUATE_MARKET_SURVIVAL", [
        P("PR_STOCHASTIC_DEAD_VIA_SENTIMENT", "invariance", 0.5, ">=",
          grp("Startup", [("runway_lifecycle", ["DEAD"])]),
          grp("ShockEvent", [("exogenous_shock", None), ("regulatory_shock", None)]), depth=1),
    ]),
    ("runway_critical", "ASSESS_SURVIVAL_CHANCE", [
        P("PR_STOCHASTIC_LAYOFFS_VIA_PANIC", "reachability", 0.8, ">=",
          grp("Startup", [("runway_critical", ["LAYOFFS_EXECUTED"])]),
          grp("Investor", [("investment_decision", None)]), depth=1),
        P("PR_STOCHASTIC_CAPITAL_CRUNCH_VIA_APPETITE", "reachability", 0.75, ">=",
          grp("Investor", [("investment_decision", ["PASSED"])]),
          grp("Startup", [("burn_rate_calculator", None), ("runway_lifecycle", None)]), depth=1),
    ]),
    ("compliance_monitor", "READ_REGULATOR_CHANNEL", [
        P("PR_STOCHASTIC_IMMUNITY_VIA_AUDIT", "reachability", 0.65, ">=",
          grp("Founder", [("compliance_monitor", ["NO_PRESSURE"])]),
          grp("ShockEvent", [("shock_lifecycle", None)])),
        P("PR_STOCHASTIC_VULNERABILITY_VIA_BURDEN", "reachability", 0.65, "<=",
          grp("Founder", [("compliance_monitor", ["PENALIZED"])]),
          grp("ShockEvent", [("shock_lifecycle", None), ("regulatory_shock", None)])),
    ]),
    ("client_retention_cycle", "PREPARE_CHURN_EMIT", [
        P("PR_STOCHASTIC_DEMAND_STABILITY_VIA_PROFILE", "reachability", 0.25, ">=",
          grp("Founder", [("strategic_resource_allocator", []), ("client_replacement", ["REPLACED"])]),
          grp("ShockEvent", [("shock_lifecycle", None)])),
    ]),
    ("client_retention_cycle", "STOCHASTIC_RETENTION_FILTER", [
        P("PR_STOCHASTIC_HYPER_CHURN_VIA_PROFILE", "reachability", 0.45, ">=",
          grp("KeyClient", [("client_retention_cycle", ["CHURNED"])]),
          grp("Founder", [("strategic_resource_allocator", None)])),
    ]),
    ("investment_decision", "STOCHASTIC_SENTIMENT_GATE", [
        P("PR_STOCHASTIC_INVESTMENT_VIA_SENTIMENT", "reachability", 0.5, ">=",
          grp("Investor", [("investment_decision", ["INVESTED"])]),
          grp("Startup", [("burn_rate_calculator", None), ("runway_lifecycle", None)])),
    ]),
    ("headcount_controller", "IDLE", [
        P("PR_STOCHASTIC_LAYOFFS_VIA_SHOCK", "reachability", 0.4, ">=",
          grp("TalentMarket", [("headcount_controller", ["LAYOFFS_DONE"])]),
          grp("Investor", [("investment_decision", None)])),
    ]),
    ("investor_exit", "DETERMINE_EXIT_REASON", [
        P("PR_STOCHASTIC_EXIT_VIA_PATIENCE", "reachability", 0.3, ">=",
          grp("Investor", [("investor_exit", ["EXITED"])]),
          grp("Startup", [("burn_rate_calculator", None), ("runway_lifecycle", None)])),
    ]),
]


# ================================================================ escritura
def dump_automata(auts):
    out = ["["]
    for i, a in enumerate(auts):
        out.append("  {")
        out.append(f'    "automaton_name": {json.dumps(a["automaton_name"])},')
        out.append(f'    "states": {json.dumps(a["states"], ensure_ascii=False)},')
        out.append('    "params": {},')
        out.append('    "transitions": [')
        for j, t in enumerate(a["transitions"]):
            clean = {k: v for k, v in t.items() if k != "__fixed__"}
            out.append("      " + json.dumps(clean, ensure_ascii=False) + ("," if j < len(a["transitions"]) - 1 else ""))
        out.append("    ]")
        out.append("  }" + ("," if i < len(auts) - 1 else ""))
    out.append("]")
    return "\n".join(out) + "\n"


def dump_patl(obs):
    out = ['{', '  "observations": [']
    for i, (aut, state, preds) in enumerate(obs):
        out.append("    {")
        out.append(f'      "automaton_name": "{aut}", "trigger_state": "{state}", "predicates": [')
        for j, p in enumerate(preds):
            out.append("        " + json.dumps(p) + ("," if j < len(preds) - 1 else ""))
        out.append("      ]")
        out.append("    }" + ("," if i < len(obs) - 1 else ""))
    out.append("  ]")
    out.append("}")
    return "\n".join(out) + "\n"


(CFG / "automata.json").write_text(dump_automata(automata))
(CFG / "patl.json").write_text(dump_patl(observations))

agents = json.loads((CFG / "agents.json").read_text())
for a in agents:
    if a["agent_type"] == "Startup":
        a["params"].setdefault("deaths", det(0))
(CFG / "agents.json").write_text(json.dumps(agents, indent=2, ensure_ascii=False) + "\n")

print("22 autómatas, 9 observaciones, 12 predicados")
print("\nUmbrales recalibrados (autómata, estado, distribución, antes, después, masa mínima antes, masa mínima después):")
for ch in changes:
    print("  ", ch)
print(f"\nCompuertas con alguna rama < {FLOOR:.0%} en algún régimen (después de calibrar):")
for name, state, dist, per, mn in band_report(automata):
    if mn < FLOOR:
        print(f"   {name}::{state} [{dist}] mínimo={mn:.4f}  {per}")
