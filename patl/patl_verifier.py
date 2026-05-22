"""
PATL Semantic Predicates Verifier.
"""

from copy import deepcopy
from itertools import product
import sys
import time
import threading
import math

from core.agents import Agents
from core.environments import Environments


class PATLVerifier:
    def __init__(self, automata, distributions):
        self.automata = automata
        self.distributions = distributions
        self._thread_id = threading.get_ident()
        print(f"[DEBUG] PATLVerifier creado en thread {self._thread_id}", flush=True)

    def verify(self, snapshot, predicates):
        print(f"[DEBUG][T{self._thread_id}] verify() INICIO - {len(predicates)} predicados", flush=True)
        t0 = time.time()
        results = []
        for i, p in enumerate(predicates):
            print(f"[DEBUG][T{self._thread_id}]   Verificando predicado {i+1}/{len(predicates)}: {p['predicate_id']}", flush=True)
            t1 = time.time()
            r = self._verify_predicate(snapshot, p)
            print(f"[DEBUG][T{self._thread_id}]   Predicado {p['predicate_id']} terminó en {time.time()-t1:.3f}s: {r['result']}", flush=True)
            results.append(r)
        print(f"[DEBUG][T{self._thread_id}] verify() FIN - {time.time()-t0:.3f}s", flush=True)
        return results

    def _verify_predicate(self, snapshot, pred):
        print(f"[DEBUG][T{self._thread_id}]   _verify_predicate INICIO: {pred['predicate_id']}", flush=True)
        t0 = time.time()
        
        agents, environments = self._load_snapshot(snapshot)
        print(f"[DEBUG][T{self._thread_id}]   _load_snapshot: {len(agents.data)} agents, {len(environments.data)} envs ({time.time()-t0:.3f}s)", flush=True)
        
        t1 = time.time()
        coalition = self._resolve_agents(agents, pred["coalition"])
        print(f"[DEBUG][T{self._thread_id}]   coalition: {len(coalition)} agents ({time.time()-t1:.3f}s)", flush=True)
        
        t1 = time.time()
        adversaries = self._resolve_agents(agents, pred.get("adversaries", {}))
        print(f"[DEBUG][T{self._thread_id}]   adversaries: {len(adversaries)} agents ({time.time()-t1:.3f}s)", flush=True)
        
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")
        predicate_id = pred["predicate_id"]

        if not coalition:
            print(f"[DEBUG][T{self._thread_id}]   ERROR: No coalition agents", flush=True)
            return {"predicate_id": predicate_id, "result": "ERROR",
                    "p_value": 0.0, "bound": bound, "operator": operator,
                    "reason": "No coalition agents"}

        t1 = time.time()
        c_strats = list(self._strategies(coalition))
        a_strats = list(self._strategies(adversaries)) if adversaries else [{}]
        print(f"[DEBUG][T{self._thread_id}]   estrategias: {len(c_strats)} coal, {len(a_strats)} adv ({time.time()-t1:.3f}s)", flush=True)
        
        max_depth = pred.get("max_depth", 20)
        pred_type = pred["type"]
        quantifier = pred.get("coalition_quantifier", "exists")

        p_game = 0.0
        total_iters = len(c_strats) * len(a_strats)
        iter_count = 0
        
        if quantifier == "exists":
            for c_idx, c_strat in enumerate(c_strats):
                worst = 1.0
                for a_idx, a_strat in enumerate(a_strats):
                    iter_count += 1
                    t_copy = time.time()
                    agents_copy = deepcopy(agents.data)
                    envs_copy = deepcopy(environments.data)
                    print(f"[DEBUG][T{self._thread_id}]   deepcopy #{iter_count}: {time.time()-t_copy:.3f}s", flush=True)
                    
                    full = {**c_strat, **a_strat}
                    t_reach = time.time()
                    print(f"[DEBUG][T{self._thread_id}]   _reach #{iter_count}/{total_iters} depth={max_depth}...", flush=True)
                    
                    p = self._reach(agents_copy, envs_copy,
                                    full, self._target_set(coalition, c_strat),
                                    max_depth, pred_type)
                    
                    print(f"[DEBUG][T{self._thread_id}]   _reach #{iter_count} terminó: p={p:.4f} ({time.time()-t_reach:.3f}s)", flush=True)
                    if p < worst:
                        worst = p
                if worst > p_game:
                    p_game = worst
        else:
            p_game = 1.0
            for c_idx, c_strat in enumerate(c_strats):
                worst = 1.0
                for a_idx, a_strat in enumerate(a_strats):
                    iter_count += 1
                    t_copy = time.time()
                    agents_copy = deepcopy(agents.data)
                    envs_copy = deepcopy(environments.data)
                    print(f"[DEBUG][T{self._thread_id}]   deepcopy #{iter_count}: {time.time()-t_copy:.3f}s", flush=True)
                    
                    full = {**c_strat, **a_strat}
                    t_reach = time.time()
                    print(f"[DEBUG][T{self._thread_id}]   _reach #{iter_count}/{total_iters} depth={max_depth}...", flush=True)
                    
                    p = self._reach(agents_copy, envs_copy,
                                    full, self._target_set(coalition, c_strat),
                                    max_depth, pred_type)
                    
                    print(f"[DEBUG][T{self._thread_id}]   _reach #{iter_count} terminó: p={p:.4f} ({time.time()-t_reach:.3f}s)", flush=True)
                    if p < worst:
                        worst = p
                if worst < p_game:
                    p_game = worst

        satisfied = self._compare(p_game, bound, operator)
        print(f"[DEBUG][T{self._thread_id}]   _verify_predicate FIN: p_game={p_game:.4f}, {operator} {bound} = {satisfied} ({time.time()-t0:.3f}s)", flush=True)

        return {"predicate_id": predicate_id,
                "result": "SATISFIED" if satisfied else "VIOLATED",
                "p_value": round(p_game, 6), "bound": bound, "operator": operator}

    def _load_snapshot(self, snap):
        a = Agents([], self.distributions, None, worker_mode=False)
        a.load_snapshot(snap["agents_data"])
        e = Environments([], self.distributions, None)
        e.load_snapshot(snap.get("environments_data"))
        return a, e

    def _resolve_agents(self, agents, spec):
        r = []
        for entry in spec:
            opts = [a["automaton_name"] for a in entry["automata"]]
            tmap = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            
            if "agent_type" in entry:
                for ag in agents.data:
                    if ag.get("agent_type") == entry["agent_type"]:
                        r.append({"id": ag["agent_id"], "opts": opts, "targets": tmap})
            
            elif "agent_id" in entry:
                for agent_id in entry["agent_id"]:
                    if any(a["agent_id"] == agent_id for a in agents.data):
                        r.append({"id": agent_id, "opts": opts, "targets": tmap})
        
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

    def _reach(self, agents_data, env_data, strat, targets, depth, pred_type):
        print(f"[DEBUG][T{self._thread_id}]     _reach INICIO: depth={depth}, targets={targets}, strat={strat}", flush=True)
        t0 = time.time()
        memo = {}
        dp_calls = [0]
        expand_calls = [0]
        last_print = [time.time()]

        def dp(agents_data, env_data, d, visited_states=None):
            if visited_states is None:
                visited_states = set()
                
            dp_calls[0] += 1
            
            # Print progress cada 2 segundos
            if time.time() - last_print[0] > 2:
                print(f"[DEBUG][T{self._thread_id}]     dp calls: {dp_calls[0]}, expand calls: {expand_calls[0]}, memo size: {len(memo)}, depth={d}", flush=True)
                last_print[0] = time.time()
            
            if d == 0:
                return 0.0
            
            # Detección de ciclo
            state_key = self._hash(agents_data, env_data)
            if state_key in visited_states:
                print(f"[DEBUG][T{self._thread_id}]     CICLO DETECTADO a depth={d}!", flush=True)
                return 0.0
                
            if any(a.get("current_state") in targets for a in agents_data):
                return 1.0
                
            if (state_key, d) in memo:
                return memo[(state_key, d)]
                
            if all(self._is_final(a) for a in agents_data if a.get("current_state")):
                memo[(state_key, d)] = 0.0
                return 0.0

            total = 0.0
            new_visited = visited_states | {state_key}
            
            branches = list(self._expand(agents_data, env_data, strat))
            expand_calls[0] += 1
            
            if dp_calls[0] <= 3:
                print(f"[DEBUG][T{self._thread_id}]     expand generó {len(branches)} ramas a depth={d}", flush=True)
            
            for prob, updates, new_env in branches:
                saved = {}
                for aid in updates:
                    for a in agents_data:
                        if a["agent_id"] == aid:
                            saved[aid] = a.get("current_state")
                            break
                
                for aid, ns in updates.items():
                    for a in agents_data:
                        if a["agent_id"] == aid:
                            a["current_state"] = ns
                            break
                            
                total += prob * dp(agents_data, new_env, d - 1, new_visited)
                
                for aid, old in saved.items():
                    for a in agents_data:
                        if a["agent_id"] == aid:
                            a["current_state"] = old
                            break

            memo[(state_key, d)] = total
            return total

        result = dp(agents_data, env_data, depth)
        print(f"[DEBUG][T{self._thread_id}]     _reach FIN: result={result:.4f}, dp_calls={dp_calls[0]}, expand_calls={expand_calls[0]}, memo_size={len(memo)} ({time.time()-t0:.3f}s)", flush=True)
        return 1.0 - result if pred_type == "invariance" else result

    def _expand(self, agents_data, env_data, strat):
        branches = [(1.0, {}, deepcopy(env_data))]
        agents_obj = self._make_agents(agents_data)
        envs_obj = self._make_envs(env_data)
        
        self.automata.agents = agents_obj
        self.automata.environments = envs_obj

        for agent_idx, agent in enumerate(agents_data):
            state = agent.get("current_state")
            aut_name = strat.get(agent["agent_id"], (agent.get("automata") or [None])[0])
            
            if not state or not aut_name:
                continue
                
            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
            if not aut_def or state in aut_def["states"].get("final", []):
                continue
                
            trans = next((t for t in aut_def["transitions"] if t["from"] == state), None)
            if not trans:
                continue

            tv = trans.get("threshold_value", {})
            # El type está dentro del valor de la variable, no en tv directamente
            # tv tiene la forma: { "$var": { "type": "...", ... } }
            threshold_type = "deterministic"  # Default seguro
            if isinstance(tv, dict):
                for var_name, var_config in tv.items():
                    if isinstance(var_config, dict) and "type" in var_config:
                        threshold_type = var_config["type"]
                        break
            
            new_branches = []
            for base_prob, base_upd, base_env in branches:
                remaining_low, remaining_high = float("-inf"), float("inf")
                for th in trans.get("thresholds", []):
                    conds = th["threshold_case"]
                    low, high, li, ui = self._interval(conds)
                    if low is None or high is None:
                        continue
                    low, high = max(low, remaining_low), min(high, remaining_high)
                    if low > high:
                        continue
                    
                    # Solo calcular probabilidad si es probabilístico
                    if threshold_type == "probabilistic":
                        # Buscar la distribución en el threshold_value
                        dist_name = None
                        for var_name, var_config in tv.items():
                            if isinstance(var_config, dict) and "distribution" in var_config:
                                dist_name = var_config["distribution"]
                                break
                        if dist_name:
                            p = self.distributions.probability_interval(dist_name, low, high, li, ui)
                        else:
                            p = 1.0
                    else:
                        p = 1.0
                    
                    if p > 0:
                        if threshold_type == "probabilistic":
                            # Calcular mid para forzar el step estocástico
                            if math.isinf(low) and math.isinf(high) and low < 0 and high > 0:
                                mid = 0.0
                            elif math.isinf(low) and math.isinf(high):
                                remaining_low = high
                                continue
                            elif low == float("-inf"):
                                mid = high - 1.0 if not math.isinf(high) else 0.0
                            elif high == float("inf"):
                                mid = low + 1.0 if not math.isinf(low) else 0.0
                            else:
                                mid = (low + high) / 2.0
                            
                            event_data = {"signal": aut_name, "agent_id": agent["agent_id"]}
                            session = self.automata.create_session(aut_name, event_data, async_mode=False)
                            session.current_state = state
                            session.params = agent.get("params", {}).copy()
                            ns = session.step(forced_X=mid)
                        else:
                            # Determinístico: tomar la transición directamente
                            ns = th.get("to")
                        
                        if ns:
                            upd = dict(base_upd)
                            upd[agent["agent_id"]] = ns
                            new_branches.append((base_prob * p, upd, deepcopy(envs_obj.data)))
                    remaining_low = high
            branches = new_branches or branches
            
        return branches

    def _make_agents(self, data):
        a = Agents([], self.distributions, None, worker_mode=False)
        a.data = data
        return a

    def _make_envs(self, data):
        e = Environments([], self.distributions, None)
        e.data = data
        return e

    def _interval(self, conditions):
        low, high = float("-inf"), float("inf")
        li = ui = True
        for c in conditions:
            op, v = c["operator"], c["value"]
            if isinstance(v, str):
                return None, None, False, False
            if v is None:
                continue
            if op == ">":
                low, li = max(low, v), False
            elif op == ">=":
                low, li = max(low, v), True
            elif op == "<":
                high, ui = min(high, v), False
            elif op == "<=":
                high, ui = min(high, v), True
            elif op == "==":
                low = high = v
                li = ui = True
            elif op == "!=":
                continue
        
        # Validación: si low > high, intervalo vacío
        if low is None or high is None:
            return None, None, False, False
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if low > high:
                return None, None, False, False
        
        # CORRECCIÓN: si low es +inf, no hay valores posibles
        if low == float("inf"):
            return None, None, False, False
        # Si high es -inf, no hay valores posibles  
        if high == float("-inf"):
            return None, None, False, False
            
        return low, high, li, ui

    def _hash(self, agents_data, env_data):
        items = tuple((a["agent_id"], a.get("current_state"), tuple(sorted(a.get("params", {}).items())))
                      for a in sorted(agents_data, key=lambda x: x["agent_id"]))
        env_items = tuple((e.get("env_id"), tuple(sorted(e.get("params", {}).items()))) for e in env_data)
        return (items, env_items)

    def _is_final(self, agent):
        aut_name = (agent.get("automata") or [None])[0]
        if not aut_name:
            return True
        aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
        return aut_def and agent.get("current_state") in aut_def["states"].get("final", []) if aut_def else True

    def _compare(self, value, bound, operator):
        if operator == ">=": return value >= bound
        if operator == ">":  return value > bound
        if operator == "<=": return value <= bound
        if operator == "<":  return value < bound
        return False