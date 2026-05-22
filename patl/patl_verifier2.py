import random
from copy import deepcopy
from itertools import product

from core.agents import Agents
from core.environments import Environments
from core.utils.evaluator import call_method


class PATLVerifier:
    def __init__(self, automata, distributions):
        self.automata = automata
        self.distributions = distributions

    def verify(self, snapshot, predicates):
        return [self._verify_predicate(snapshot, p) for p in predicates]

    def _verify_predicate(self, snapshot, pred):
        predicate_id = pred["predicate_id"]
        pred_type = pred["type"]
        print(f"\n" + "="*80)
        print(f"[PATL START] Evaluando Predicado: {predicate_id} ({pred_type})")
        print(f"="*80)
        
        targets = set()
        for agent_spec in pred.get("coalition", []):
            for aut_spec in agent_spec.get("automata", []):
                targets.update(aut_spec.get("target_states", []))
        print(f"[PATL TARGETS] Objetivos buscados para {predicate_id}: {targets}")
        
        agents = Agents([], self.distributions, None, worker_mode=False)
        agents.automata = self.automata
        agents.load_snapshot(snapshot.get("agents_data", {}))
        
        environments = Environments([], self.distributions, None)
        environments.load_snapshot(snapshot.get("environments_data", {}))
        
        coalition = self._resolve_agents(agents, pred["coalition"])
        adversaries = self._resolve_agents(agents, pred.get("adversaries", []))
        
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")

        if not coalition:
            print(f"[PATL ERROR] Coalition vacía o no resuelta para {predicate_id}")
            return {"predicate_id": predicate_id, "result": "ERROR", "p_value": 0.0}

        c_strats = list(self._strategies(coalition))
        a_strats = list(self._strategies(adversaries)) if adversaries else [{}]
        
        max_depth = pred.get("max_depth", 20)
        quantifier = pred.get("coalition_quantifier", "exists")

        p_game = 0.0 if quantifier == "exists" else 1.0
        
        print(f"[PATL STRATS] {len(c_strats)} estrategias de coalición × {len(a_strats)} de adversarios.")
        
        for idx_c, c_strat in enumerate(c_strats):
            worst = 1.0
            for idx_a, a_strat in enumerate(a_strats):
                full = {**c_strat, **a_strat}
                
                print(f"\n  [STRAT COMBINATION #{idx_c}-{idx_a}] Matriz completa: {full}")
                
                clean_agents_data = deepcopy(agents.data)
                clean_env_data = deepcopy(environments.data)
                
                p = self._reach_from_snapshot(
                    clean_agents_data, 
                    clean_env_data,
                    full, 
                    targets,  # Pasamos explícitamente los targets de ESTE predicado
                    max_depth, 
                    pred_type,
                    agents,
                    environments,
                    predicate_id
                )
                if p < worst:
                    worst = p
                    
            if quantifier == "exists":
                if worst > p_game:
                    p_game = worst
            else:
                if worst < p_game:
                    p_game = worst

        satisfied = self._compare(p_game, bound, operator)
        print(f"\n" + "-"*80)
        print(f"[PATL RESULT] {predicate_id} -> P={p_game:.4f} Final: {'SATISFIED (SAT)' if satisfied else 'VIOLATED (VIOL)'}")
        print(f"-"*80)

        return {
            "predicate_id": predicate_id,
            "result": "SATISFIED" if satisfied else "VIOLATED",
            "p_value": round(p_game, 6),
            "bound": bound, 
            "operator": operator
        }

    def _reach_from_snapshot(self, agents_data, env_data, strat, targets, depth, pred_type, agents_obj, envs_obj, predicate_id):
        # El mapa de memoización debe aislarse por predicado y por targets para evitar falsos positivos inter-bucle
        memo = {}
        
        def get_state_key(a_data, d):
            items = []
            for a in a_data:
                a_id = a.get("agent_id", a.get("id", str(id(a))))
                c_state = a.get("current_state", "UNKNOWN")
                items.append((a_id, c_state))
            # Incluimos los targets del predicado en la clave hash para evitar colisiones de caché cruzada
            return (tuple(sorted(items)), tuple(sorted(list(targets))), d)
        
        def resolve_threshold_value(tv_def, ctx, indent):
            if not tv_def:
                return None
            tv_key = next(iter(tv_def))
            tv_value_def = tv_def[tv_key]
            
            if isinstance(tv_value_def, dict) and "target" in tv_value_def:
                params = tv_value_def.get("params", [])
                resolved_params = []
                for p in params:
                    if isinstance(p, str) and p.startswith("$"):
                        resolved_params.append(ctx.get(p[1:], p))
                    else:
                        resolved_params.append(p)
                
                # ==================================================================
                # INTERCEPCIÓN / PARCHE DIRECTO PARA LA LECTURA DEL ENTORNO
                # ==================================================================
                if tv_value_def.get("target") == "environments" and tv_value_def.get("method") == "read_env_param":
                    # Forzar extracción segura desde la lista interna de diccionarios
                    try:
                        target_env_id = resolved_params[0]
                        target_param_key = resolved_params[1]
                        
                        res = None
                        for e in envs_obj.data:
                            if e.get("env_id") == target_env_id:
                                res = e.get("params", {}).get(target_param_key, None)
                                break
                        print(f"{indent}[PARCHE INTERCEPTOR] Extracción directa exitosa -> {target_env_id}::params::{target_param_key} = {res}")
                    except Exception as e:
                        print(f"{indent}[PARCHE INTERCEPTOR ERROR] Falló extracción directa: {e}")
                        res = call_method(tv_value_def["target"], tv_value_def["method"], resolved_params, agents_obj, envs_obj)
                else:
                    res = call_method(tv_value_def["target"], tv_value_def["method"], resolved_params, agents_obj, envs_obj)
                # ==================================================================
                
                print(f"{indent}[DEB TV] call_method -> {tv_value_def['target']}::{tv_value_def['method']}({resolved_params}) = {res}")
                return {tv_key: res}
            
            if isinstance(tv_value_def, dict) and tv_value_def.get("type") == "probabilistic":
                return {tv_key: 0.5}
            
            if isinstance(tv_value_def, str) and tv_value_def.startswith("$"):
                return {tv_key: ctx.get(tv_value_def[1:], None)}
            
            return {tv_key: tv_value_def}

        def check_case(conditions, ctx, indent):
            for cond in conditions:
                var_name = cond["variable"]
                op = cond["operator"]
                expected = cond["value"]
                
                if isinstance(expected, str) and expected.startswith("$"):
                    expected = ctx.get(expected[1:], None)
                
                real_var = var_name[1:] if var_name.startswith("$") else var_name
                actual = ctx.get(real_var, None)
                
                if actual is None:
                    print(f"{indent}[DEB CASE FAIL] Variable '{real_var}' es None. Condición rechazada.")
                    return False
                
                try:
                    if op == "<" and not (actual < expected): return False
                    elif op == "<=" and not (actual <= expected): return False
                    elif op == ">" and not (actual > expected): return False
                    elif op == ">=" and not (actual >= expected): return False
                    elif op == "==" and not (actual == expected): return False
                    elif op == "!=" and not (actual != expected): return False
                except TypeError as e:
                    print(f"{indent}[DEB CASE ERROR] Error de tipo comparando {actual} {op} {expected}: {e}")
                    return False
            return True
        
        def apply_actions(actions, ctx, indent):
            new_ctx = ctx.copy()
            for action in actions:
                if "update_params" in action:
                    for k, vdef in action["update_params"].items():
                        if isinstance(vdef, dict) and "target" in vdef:
                            params = vdef.get("params", [])
                            resolved_params = [ctx.get(p[1:], p) if (isinstance(p, str) and p.startswith("$")) else p for p in params]
                            
                            # Aplicamos la misma regla de interceptación para acciones si requiere leer entornos
                            if vdef.get("target") == "environments" and vdef.get("method") == "read_env_param":
                                try:
                                    t_env_id = resolved_params[0]
                                    t_param_key = resolved_params[1]
                                    val = None
                                    for e in envs_obj.data:
                                        if e.get("env_id") == t_env_id:
                                            val = e.get("params", {}).get(t_param_key, None)
                                            break
                                    new_ctx[k] = val
                                except Exception:
                                    new_ctx[k] = call_method(vdef["target"], vdef["method"], resolved_params, agents_obj, envs_obj)
                            else:
                                new_ctx[k] = call_method(vdef["target"], vdef["method"], resolved_params, agents_obj, envs_obj)
                                
                            print(f"{indent}[DEB ACT] update_params {k} de llamada externa = {new_ctx[k]}")
                        elif isinstance(vdef, str) and vdef.startswith("$"):
                            new_ctx[k] = ctx.get(vdef[1:], None)
                        else:
                            new_ctx[k] = vdef
                elif "action_required" in action:
                    for k, vdef in action["action_required"].items():
                        if isinstance(vdef, dict) and "target" in vdef:
                            params = vdef.get("params", [])
                            resolved_params = [ctx.get(p[1:], p) if (isinstance(p, str) and p.startswith("$")) else p for p in params]
                            new_ctx[k] = call_method(vdef["target"], vdef["method"], resolved_params, agents_obj, envs_obj)
                            print(f"{indent}[DEB ACT] action_required ejecutada en {k}")
            return new_ctx
        
        def dp(current_agents, current_env, d):
            indent = "    " * (depth - d + 1)
            
            if d == 0:
                return 0.0
                
            for a in current_agents:
                st = a.get("current_state", "")
                if st in targets:
                    print(f"{indent}🎉 [TARGET ALCANZADO] Agente {a.get('agent_id')} en estado objetivo '{st}'")
                    return 1.0
            
            agents_obj.data = current_agents
            envs_obj.data = current_env
            
            print(f"{indent}[DEBUG PARAMS] Datos del Entorno Actual:")
            print(f"{indent}  -> {envs_obj.data}")
            print(f"{indent}[DEBUG PARAMS] Parámetros de los Agentes Actuales:")
            for a in current_agents:
                print(f"{indent}  -> Agente: {a.get('agent_id')} | Params: {a.get('params', {})}")
            
            memo_key = get_state_key(current_agents, d)
            if memo_key in memo:
                return memo[memo_key]
            
            total = 0.0
            
            for agent in current_agents:
                agent_id = agent.get("agent_id", agent.get("id"))
                aut_name = strat.get(agent_id)
                if not aut_name:
                    continue
                
                state = agent.get("current_state", "")
                print(f"{indent}[EXPLORANDO] Agente: {agent_id} | Estado Actual: {state} | Profundidad Restante: {d}")
                
                aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
                if not aut_def:
                    print(f"{indent}  [WARN] No se encontró definición para autómata {aut_name}")
                    continue
                
                final_states = aut_def.get("states", {}).get("final", [])
                if state in final_states:
                    print(f"{indent}  [STOP] Agente está en estado final/sumidero '{state}' del autómata.")
                    continue
                
                trans_list = [t for t in aut_def.get("transitions", []) if t.get("from") == state]
                if not trans_list:
                    print(f"{indent}  [DEADLOCK] Sin transiciones salientes desde '{state}'")
                    continue
                
                ctx = agent.get("params", {}).copy()
                
                if state == "READ_REGULATOR_CHANNEL":
                    ctx.setdefault("env_id", "StartupEcosystem_1")
                    ctx.setdefault("param_key", "regulatory_pressure")
                    ctx.setdefault("penalty_factor", 0.9)
                    ctx.setdefault("agent_id", "Startup_1")
                
                trans = trans_list[0] 
                tv = trans.get("threshold_value", {})
                if tv:
                    resolved = resolve_threshold_value(tv, ctx, indent)
                    if resolved:
                        ctx.update(resolved)
                
                thresholds = trans.get("thresholds", [])
                valid_cases_found = 0
                
                for th in thresholds:
                    if not check_case(th.get("threshold_case", []), ctx, indent):
                        continue
                    
                    valid_cases_found += 1
                    prob = th.get("probability")
                    if prob is None:
                        prob = 1.0 / len(thresholds) if thresholds else 1.0
                    prob = float(prob)
                    
                    next_state = th.get("to", th.get("next", th.get("target", state)))
                    print(f"{indent}  [TRANSICIÓN SIMPLE] '{state}' -> '{next_state}' (p={prob})")
                    
                    branch_agents = deepcopy(current_agents)
                    branch_env = deepcopy(current_env)
                    
                    agents_obj.data = branch_agents
                    envs_obj.data = branch_env
                    
                    new_ctx = apply_actions(th.get("actions", []), ctx, indent)
                    
                    for a in branch_agents:
                        if a.get("agent_id", a.get("id")) == agent_id:
                            a["current_state"] = next_state
                            new_params = a.get("params", {}).copy()
                            for k, v in new_ctx.items():
                                if not k.startswith("$"):
                                    new_params[k] = v
                            a["params"] = new_params
                    
                    total += prob * dp(branch_agents, branch_env, d - 1)
                
                if valid_cases_found == 0:
                    print(f"{indent}  [BLOQUEO] Ningún threshold_case se cumplió en el estado '{state}' con contexto: { {k:v for k,v in ctx.items() if not k.startswith('$')} }")
            
            memo[memo_key] = total
            return total
        
        result = dp(agents_data, env_data, depth)
        result = max(0.0, min(1.0, result))
        
        if pred_type == "invariance":
            result = 1.0 - result
            
        return result

    def _resolve_agents(self, agents, spec):
        r = []
        if not isinstance(spec, list): return r
        for entry in spec:
            agent_type = entry.get("agent_type")
            if not agent_type: continue
            max_agents = entry.get("max_agents", float("inf"))
            opts = [a["automaton_name"] for a in entry.get("automata", [])]
            tmap = {a["automaton_name"]: set(a["target_states"]) for a in entry.get("automata", [])}
            matching_agents = [ag for ag in agents.data if ag.get("agent_type") == agent_type]
            if len(matching_agents) > max_agents:
                random.seed(42)
                matching_agents = random.sample(matching_agents, max_agents)
            for ag in matching_agents:
                r.append({"id": ag.get("agent_id", ag.get("id")), "opts": opts, "targets": tmap})
        return r

    def _strategies(self, agents):
        if not agents:
            yield {}
            return
        ids = [a["id"] for a in agents]
        for combo in product(*[a["opts"] for a in agents]):
            yield dict(zip(ids, combo))

    def _target_set(self, coalition, strat):
        t = set()
        for a in coalition:
            chosen = strat.get(a["id"])
            if chosen and chosen in a["targets"]:
                t |= a["targets"][chosen]
        return t

    def _compare(self, value, bound, operator):
        eps = 1e-9
        if operator == ">=": return value >= bound - eps
        if operator == ">":  return value > bound + eps
        if operator == "<=": return value <= bound + eps
        if operator == "<":  return value < bound - eps
        return False