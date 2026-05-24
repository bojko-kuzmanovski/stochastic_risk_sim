"""
PATL Semantic Predicates Verifier - Optimized High-Performance Edition.
"""

from itertools import product
import sys
import time
import threading
import math
import random

from core.agents import Agents
from core.environments import Environments


class PATLVerifier:
    def __init__(self, automata, distributions, configs=None, epsilon=1e-6):
        self.automata = automata
        self.distributions = distributions
        self.configs = configs or {}
        self._thread_id = threading.get_ident()
        self.epsilon = epsilon

        self.agent_meta = {a["agent_type"]: a["params"] for a in self.configs.get("agents", [])}
        self.env_meta = {e["environment_type"]: e["params"] for e in self.configs.get("environments", [])}

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
        t0 = time.time()
        agents, environments = self._load_snapshot(snapshot)
        
        # Indexación rápida para evitar búsquedas O(N) lineales posteriores
        agents_map = {ag["agent_id"]: ag for ag in agents.data}
        
        coalition = self._resolve_agents(agents_map, pred["coalition"])
        adversaries = self._resolve_agents(agents_map, pred.get("adversaries", {}))
        
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")
        predicate_id = pred["predicate_id"]

        if not coalition:
            return {"predicate_id": predicate_id, "result": "ERROR",
                    "p_value": 0.0, "bound": bound, "operator": operator,
                    "reason": "No coalition agents"}

        c_strats = list(self._strategies(coalition))
        a_strats = list(self._strategies(adversaries)) if adversaries else [{}]
        
        max_depth = pred.get("max_depth", 20)
        pred_type = pred["type"]
        quantifier = pred.get("coalition_quantifier", "exists")

        p_game = 0.0 if quantifier == "exists" else 1.0
        total_iters = len(c_strats) * len(a_strats)
        iter_count = 0
        
        # Reutilizamos las estructuras base evitando deepcopies masivos
        for c_strat in c_strats:
            worst = 1.0
            for a_strat in a_strats:
                iter_count += 1
                full_strategy = {**c_strat, **a_strat}
                targets = self._target_set(coalition, c_strat)
                
                # Pasamos diccionarios planos que mutaremos de forma controlada (backtracking)
                coalition_ids = [c["id"] for c in coalition]
                p = self._reach(agents_map, environments.data, full_strategy, targets, max_depth, pred_type, coalition_ids)
                
                if p < worst:
                    worst = p
                    
            if quantifier == "exists":
                if worst > p_game:
                    p_game = worst
            else:
                if worst < p_game:
                    p_game = worst

        satisfied = self._compare(p_game, bound, operator)
        return {"predicate_id": predicate_id,
                "result": "SATISFIED" if satisfied else "VIOLATED",
                "p_value": round(p_game, 6), "bound": bound, "operator": operator}

    def _load_snapshot(self, snap):
        a = Agents([], self.distributions, None, worker_mode=False)
        a.load_snapshot(snap["agents_data"])
        e = Environments([], self.distributions, None)
        e.load_snapshot(snap.get("environments_data"))
        return a, e

    def _resolve_agents(self, agents_map, spec):
        r = []
        for entry in spec:
            opts = [a["automaton_name"] for a in entry["automata"]]
            tmap = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            max_agents = entry.get("max_agents", 10)
            
            matching = []
            if "agent_type" in entry:
                matching = [ag for ag in agents_map.values() if ag.get("agent_type") == entry["agent_type"]]
            elif "agent_id" in entry:
                matching = [agents_map[aid] for aid in entry["agent_id"] if aid in agents_map]
            
            # Muestreo estocástico uniforme balanceado para preservar el espacio de estados
            if len(matching) > max_agents:
                # 1. Aseguramos un orden base estricto por ID para que la semilla actúe sobre una estructura idéntica
                matching = sorted(matching, key=lambda x: x["agent_id"])
                
                # 2. Creamos una semilla local reproducible combinando los IDs estables disponibles
                seed_string = "".join(ag["agent_id"] for ag in matching)
                local_rng = random.Random(seed_string)
                
                # 3. Extraemos la muestra uniforme sin alterar el estado global de random
                matching = local_rng.sample(matching, max_agents)
                
            for ag in matching:
                r.append({"id": ag["agent_id"], "opts": opts, "targets": tmap})
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

    def _reach(self, agents_map, env_data, strat, targets, depth, pred_type, coalition_ids=None):
        print(f"[DEBUG][T{self._thread_id}]     _reach INICIO: depth={depth}, targets={targets}", flush=True)
        print(f"[DEBUG][T{self._thread_id}]     strat={strat}", flush=True)
        print(f"[DEBUG][T{self._thread_id}]     agentes en strat: {[(aid, agents_map[aid].get('current_state')) for aid in strat if aid in agents_map]}", flush=True)
        # DEBUG: Mostrar regulatory_pressure del entorno
        for env in env_data:
            reg_pressure = env.get("params", {}).get("regulatory_pressure", "NO_ENCONTRADO")
            print(f"[DEBUG][T{self._thread_id}]     ENV {env.get('env_id', '?')}: regulatory_pressure={reg_pressure}", flush=True)

        memo = {}
        
        def dp(d, visited_states):
            if d == 0:
                print(f"[DEBUG][T{self._thread_id}]     dp: d=0 -> 0.0", flush=True)
                return 0.0
                
            state_key = self._hash(agents_map, env_data)
            if state_key in visited_states:
                print(f"[DEBUG][T{self._thread_id}]     dp: CICLO detectado -> 0.0", flush=True)
                return 0.0
                
            if (state_key, d) in memo:
                return memo[(state_key, d)]
            
            # Verificar target SOLO en coalición
            if coalition_ids:
                for aid in coalition_ids:
                    if aid in agents_map and agents_map[aid].get("current_state") in targets:
                        print(f"[DEBUG][T{self._thread_id}]     dp: TARGET alcanzado! {aid} en {agents_map[aid].get('current_state')} -> 1.0", flush=True)
                        return 1.0
            
            # VERIFICAR FINALES ANTES DE EXPANDIR
            non_final_agents = [aid for aid in strat if aid in agents_map and not self._is_final(agents_map[aid])]
            if not non_final_agents:
                print(f"[DEBUG][T{self._thread_id}]     dp: Todos finales -> 0.0", flush=True)
                memo[(state_key, d)] = 0.0
                return 0.0

            total = 0.0
            new_visited = visited_states | {state_key}
            branches = self._expand_inplace(agents_map, env_data, strat)
            print(f"[DEBUG][T{self._thread_id}]     dp: d={d}, branches={len(branches)}", flush=True)
            
            if not branches:
                print(f"[DEBUG][T{self._thread_id}]     dp: SIN RAMAS -> 0.0", flush=True)
                memo[(state_key, d)] = 0.0
                return 0.0
            
            for prob, updates in branches:
                print(f"[DEBUG][T{self._thread_id}]     dp: rama prob={prob:.4f}, updates={updates}", flush=True)
                if prob < self.epsilon:
                    continue
                    
                saved_states = {}
                for aid, ns in updates.items():
                    if aid in agents_map:
                        saved_states[aid] = agents_map[aid]["current_state"]
                        agents_map[aid]["current_state"] = ns
                
                total += prob * dp(d - 1, new_visited)
                
                for aid, old_state in saved_states.items():
                    agents_map[aid]["current_state"] = old_state

            memo[(state_key, d)] = total
            return total

        result = dp(depth, frozenset())
        print(f"[DEBUG][T{self._thread_id}]     _reach FINAL: result={result:.4f}", flush=True)
        return 1.0 - result if pred_type == "invariance" else result

    def _expand_inplace(self, agents_map, env_data, strat):
        """Genera combinaciones de transiciones viables sin clonar entornos."""
        print(f"[DEBUG][T{self._thread_id}]       _expand_inplace: strat_agent_ids={set(strat.keys())}", flush=True)
        branches = [(1.0, {})]
        strat_agent_ids = set(strat.keys())
        
        for agent_id in strat_agent_ids:
            if agent_id not in agents_map:
                print(f"[DEBUG][T{self._thread_id}]       {agent_id} NO EN agents_map!", flush=True)
                continue
            agent = agents_map[agent_id]
            state = agent.get("current_state")
            aut_name = strat[agent_id]
            print(f"[DEBUG][T{self._thread_id}]       agente={agent_id}, state={state}, aut={aut_name}", flush=True)
            
            if not state or not aut_name:
                continue
                
            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
            if not aut_def:
                continue

            if state not in aut_def["states"].get("final", []) and \
                state not in aut_def["states"].get("initial", "") and \
                state not in [t["from"] for t in aut_def.get("transitions", [])]:
                    state = aut_def["states"]["initial"]
                    agent["current_state"] = state
                    print(f"[DEBUG][T{self._thread_id}]       {agent_id}: estado reset a inicial={state}", flush=True)
                
            if state in aut_def["states"].get("final", []):
                continue
                
            trans = next((t for t in aut_def["transitions"] if t["from"] == state), None)
            if not trans:
                continue

            tv = trans.get("threshold_value", {})
            tv_key = next(iter(tv))
            tv_def = tv[tv_key]
            
            threshold_type = "deterministic"
            dist_name = None
            truncation_limits = None
            
            if isinstance(tv_def, dict):
                if "type" in tv_def:
                    threshold_type = tv_def.get("type", "deterministic")
                    dist_name = tv_def.get("distribution")
                    if dist_name:
                        dist_entry = self.distributions.samplers.get(dist_name)
                        if dist_entry and dist_entry.get("truncation"):
                            truncation_limits = (dist_entry["truncation"]["min"], dist_entry["truncation"]["max"])
                elif "target" in tv_def:
                    threshold_type = "dynamic"

            new_branches = []
            for base_prob, base_upd in branches:
                for th in trans.get("thresholds", []):
                    conds = th["threshold_case"]
                    low, high, li, ui = self._interval(conds, truncation_limits)
                    if low is None or high is None:
                        continue
                    if low > high:
                        continue

                    ns = None
                    p = 0.0

                    if threshold_type == "probabilistic" and dist_name:
                        p = self.distributions.probability_interval(dist_name, low, high, li, ui)
                        if p > 0:
                            saved_agents = self.automata.agents
                            saved_envs = self.automata.environments
                            
                            agents_obj = self._make_agents_from_map(agents_map)
                            envs_obj = self._make_envs(env_data)
                            self.automata.agents = agents_obj
                            self.automata.environments = envs_obj
                            
                            event_data = {"signal": aut_name, "agent_id": agent_id}
                            session = self.automata.create_session(aut_name, event_data, async_mode=False)
                            session.current_state = state
                            session.params = agent.get("params", {}).copy()
                            
                            if low == float("-inf") and high == float("inf"):
                                mid = 0.0
                            elif low == float("-inf"):
                                mid = high - 1.0
                            elif high == float("inf"):
                                mid = low + 1.0
                            else:
                                mid = (low + high) / 2.0
                            
                            ns = session.step(forced_X=mid)
                            print(f"[DEBUG][T{self._thread_id}]       PROB step: state={state}, dist={dist_name}, interval=[{low:.4f}, {high:.4f}], p={p:.4f}, mid={mid:.4f}, ns={ns}", flush=True)
                            
                            self.automata.agents = saved_agents
                            self.automata.environments = saved_envs
                    
                    elif threshold_type == "dynamic":
                        saved_agents = self.automata.agents
                        saved_envs = self.automata.environments
                        
                        agents_obj = self._make_agents_from_map(agents_map)
                        envs_obj = self._make_envs(env_data)
                        self.automata.agents = agents_obj
                        self.automata.environments = envs_obj
                        
                        event_data = {"signal": aut_name, "agent_id": agent_id}
                        session = self.automata.create_session(aut_name, event_data, async_mode=False)
                        session.current_state = state
                        session.params = agent.get("params", {}).copy()
                        
                        if low == float("-inf") and high == float("inf"):
                            mid = 0.0
                        elif low == float("-inf"):
                            mid = high - 1.0
                        elif high == float("inf"):
                            mid = low + 1.0
                        else:
                            mid = (low + high) / 2.0
                        
                        ns = session.step(forced_X=mid)
                        print(f"[DEBUG][T{self._thread_id}]       DYNAMIC step: state={state}, interval=[{low:.4f}, {high:.4f}], mid={mid:.4f}, ns={ns}", flush=True)
                        
                        self.automata.agents = saved_agents
                        self.automata.environments = saved_envs
                        p = 1.0 if ns else 0.0
                    
                    else:
                        saved_agents = self.automata.agents
                        saved_envs = self.automata.environments
                        
                        agents_obj = self._make_agents_from_map(agents_map)
                        envs_obj = self._make_envs(env_data)
                        self.automata.agents = agents_obj
                        self.automata.environments = envs_obj
                        
                        event_data = {"signal": aut_name, "agent_id": agent_id}
                        session = self.automata.create_session(aut_name, event_data, async_mode=False)
                        session.current_state = state
                        session.params = agent.get("params", {}).copy()
                        
                        if low == float("-inf") and high == float("inf"):
                            mid = 0.0
                        elif low == float("-inf"):
                            mid = high - 1.0
                        elif high == float("inf"):
                            mid = low + 1.0
                        else:
                            mid = (low + high) / 2.0
                        
                        ns = session.step(forced_X=mid)
                        
                        self.automata.agents = saved_agents
                        self.automata.environments = saved_envs
                        p = 1.0 if ns else 0.0
                    
                    if p > 0 and ns:
                        upd = dict(base_upd)
                        upd[agent_id] = ns
                        new_branches.append((base_prob * p, upd))
            
            if new_branches:
                branches = new_branches
            
        if len(branches) == 1 and branches[0][1] == {}:
            return []
                
        return branches

    def _make_agents_from_map(self, agents_map):
        a = Agents([], self.distributions, None, worker_mode=False)
        a.data = list(agents_map.values())
        return a
    
    def _make_envs(self, env_data):
        e = Environments([], self.distributions, None)
        e.data = env_data if isinstance(env_data, list) else [env_data]
        return e
    
    def _compare_deterministic_case(self, actual_value, conditions):
        """Evalúa un valor escalar determinista frente a un set de condiciones de intervalo."""
        if actual_value is None:
            return False
        for c in conditions:
            op, expected = c["operator"], c["value"]
            if isinstance(expected, str): 
                return False
            try:
                if op == "<" and not (actual_value < expected): return False
                elif op == "<=" and not (actual_value <= expected): return False
                elif op == ">" and not (actual_value > expected): return False
                elif op == ">=" and not (actual_value >= expected): return False
                elif op == "==" and not (actual_value == expected): return False
                elif op == "!=" and not (actual_value != expected): return False
            except TypeError:
                return False
        return True

    def _interval(self, conditions, truncation_limits=None):
        low, high = float("-inf"), float("inf")
        li = ui = True
        for c in conditions:
            op, v = c["operator"], c["value"]
            if isinstance(v, str):
                return None, None, False, False
            if v is None:
                continue
            if op == ">": low, li = max(low, v), False
            elif op == ">=": low, li = max(low, v), True
            elif op == "<": high, ui = min(high, v), False
            elif op == "<=": high, ui = min(high, v), True
            elif op == "==": low = high = v; li = ui = True
        
        if truncation_limits:
            t_min, t_max = truncation_limits
            if low == float("-inf"):
                low = t_min
            if high == float("inf"):
                high = t_max
            
        if low > high or low == float("inf") or high == float("-inf"):
            return None, None, False, False
            
        return low, high, li, ui
    
    def _hash(self, agents_map, env_data):
        items = tuple(
            (aid, ag.get("current_state"), tuple(sorted(ag.get("params", {}).items())))
            for aid, ag in sorted(agents_map.items())
        )
        env_items = tuple(
            (e.get("env_id"), tuple(sorted(e.get("params", {}).items()))) 
            for e in sorted(env_data, key=lambda x: x.get("env_id", ""))
        )
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