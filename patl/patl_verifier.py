from copy import deepcopy
from itertools import product
import sys

from core.agents import Agents
from core.environments import Environments


class PATLVerifier:
    def __init__(self, automata, distributions):
        self.automata = automata
        self.distributions = distributions

    def verify(self, snapshot, predicates):
        return [self._verify_predicate(snapshot, p) for p in predicates]

    def _verify_predicate(self, snapshot, pred):
        agents, environments = self._load_snapshot(snapshot)
        coalition = self._resolve_agents(agents, pred["coalition"], is_coalition=True)
        adversaries = self._resolve_agents(agents, pred.get("adversaries", {}), is_coalition=False)
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

        original_agents = self.automata.agents
        original_envs = self.automata.environments

        p_game = 0.0
        if quantifier == "exists":
            for c_strat in c_strats:
                worst = 1.0
                for a_strat in a_strats:
                    full = {**c_strat, **a_strat}
                    # Forzamos un deepcopy inicial limpio de los datos del snapshot antes de iniciar la búsqueda de la estrategia
                    p = self._reach(deepcopy(agents.data), deepcopy(environments.data),
                                    full, self._target_set(coalition, c_strat),
                                    max_depth, pred_type, coalition)
                    if p < worst:
                        worst = p
                if worst > p_game:
                    p_game = worst
        else:
            p_game = 1.0
            for c_strat in c_strats:
                worst = 1.0
                for a_strat in a_strats:
                    full = {**c_strat, **a_strat}
                    p = self._reach(deepcopy(agents.data), deepcopy(environments.data),
                                    full, self._target_set(coalition, c_strat),
                                    max_depth, pred_type, coalition)
                    if p < worst:
                        worst = p
                if worst < p_game:
                    p_game = worst

        satisfied = self._compare(p_game, bound, operator)

        self.automata.agents = original_agents
        self.automata.environments = original_envs

        return {"predicate_id": predicate_id,
                "result": "SATISFIED" if satisfied else "VIOLATED",
                "p_value": round(p_game, 6), "bound": bound, "operator": operator}

    def _load_snapshot(self, snap):
        a = Agents([], self.distributions, None, worker_mode=False)
        a.automata = self.automata
        a.load_snapshot(snap["agents_data"])
        e = Environments([], self.distributions, None)
        e.load_snapshot(snap.get("environments_data"))
        return a, e

    def _resolve_agents(self, agents, spec, is_coalition=False):
        r = []
        for entry in spec:
            opts = [a["automaton_name"] for a in entry["automata"]]
            tmap = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            for ag in agents.data:
                if ag.get("agent_type") == entry["agent_type"]:
                    initial_states = {}
                    if is_coalition:
                        for aut_name in opts:
                            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
                            if aut_def:
                                initial_states[aut_name] = aut_def["states"].get("initial", "LOAD_PARAMS")
                    
                    r.append({
                        "id": ag["agent_id"], 
                        "opts": opts, 
                        "targets": tmap,
                        "initial_states": initial_states,
                        "is_coalition": is_coalition
                    })
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

    def _reach(self, agents_data, env_data, strat, targets, depth, pred_type, coalition_spec):
        memo = {}
        coalition_ids = {c["id"]: c for c in coalition_spec}

        def dp(agents_data, env_data, d):
            if d == 0:
                return 0.0
            if any(a.get("current_state") in targets for a in agents_data):
                return 1.0
            key = self._hash(agents_data, env_data)
            if (key, d) in memo:
                return memo[(key, d)]
            if all(self._is_final(a) for a in agents_data if a.get("current_state")):
                memo[(key, d)] = 0.0
                return 0.0

            total = 0.0
            for prob, updates, new_env in self._expand(agents_data, env_data, strat, coalition_ids, is_initial_step=(d == depth)):
                saved = {aid: next(a for a in agents_data if a["agent_id"] == aid).get("current_state")
                         for aid in updates}
                for aid, ns in updates.items():
                    next(a for a in agents_data if a["agent_id"] == aid)["current_state"] = ns
                total += prob * dp(agents_data, new_env, d - 1)
                for aid, old in saved.items():
                    next(a for a in agents_data if a["agent_id"] == aid)["current_state"] = old

            memo[(key, d)] = total
            return total

        result = dp(agents_data, env_data, depth)
        return 1.0 - result if pred_type == "invariance" else result

    def _expand(self, agents_data, env_data, strat, coalition_ids, is_initial_step=False):
        branches = [(1.0, {}, deepcopy(env_data))]
        agents_obj = self._make_agents(agents_data)
        envs_obj = self._make_envs(env_data)
        self.automata.agents = agents_obj
        self.automata.environments = envs_obj

        agents_to_expand = [a for a in agents_data if a["agent_id"] in strat]
        
        for idx, agent in enumerate(agents_to_expand):
            aut_name = strat.get(agent["agent_id"])
            if not aut_name:
                aut_name = (agent.get("automata") or [None])[0]
                
            if not aut_name:
                continue

            # RESET LÓGICO: Si es el agente de la coalición y estamos en el paso 0 de la verificación PATL, 
            # ignoramos el estado del snapshot y forzamos el estado inicial del autómata
            if is_initial_step and agent["agent_id"] in coalition_ids:
                state = coalition_ids[agent["agent_id"]]["initial_states"].get(aut_name, "LOAD_PARAMS")
            else:
                state = agent.get("current_state")
                
            if not state:
                continue
                
            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
            if not aut_def or state in aut_def["states"].get("final", []):
                continue
            trans = next((t for t in aut_def["transitions"] if t["from"] == state), None)
            if not trans:
                continue

            tv = trans["threshold_value"]
            tv_key = next(iter(tv))
            tv_def = tv[tv_key]

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
                    
                    is_probabilistic = isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic"
                    
                    if is_probabilistic:
                        p = self.distributions.probability_interval(tv_def["distribution"], low, high, li, ui)
                    else:
                        p = 1.0
                        
                    if p > 0:
                        mid = (low + high) / 2 if low != float("-inf") and high != float("inf") else (low if low != float("-inf") else high)
                        event_data = {"signal": aut_name, "agent_id": agent["agent_id"]}
                        
                        session = self.automata.create_session(aut_name, event_data, async_mode=False)
                        session.current_state = state
                        session.ctx = agent.get("params", {}).copy()
                        
                        if is_probabilistic:
                            ns = session.step(forced_X=mid)
                        else:
                            ns = session.step()
                            
                        if ns:
                            upd = dict(base_upd)
                            upd[agent["agent_id"]] = ns
                            new_branches.append((base_prob * p, upd, deepcopy(envs_obj.data)))
                    remaining_low = high
            branches = new_branches or branches

        return branches

    def _is_final(self, agent):
        aut_name = (agent.get("automata") or [None])[0]
        if not aut_name:
            return True
        aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
        return aut_def and agent.get("current_state") in aut_def["states"].get("final", []) if aut_def else True
    
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
        if low is None or high is None:
            return None, None, False, False
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if low > high:
                return None, None, False, False
        return low, high, li, ui

    def _hash(self, agents_data, env_data):
        items = tuple((a["agent_id"], a.get("current_state"), tuple(sorted(a.get("params", {}).items())))
                      for a in sorted(agents_data, key=lambda x: x["agent_id"]))
        env_items = tuple((e.get("env_id"), tuple(sorted(e.get("params", {}).items()))) for e in env_data)
        return (items, env_items)

    def _compare(self, value, bound, operator):
        if operator == ">=": return value >= bound
        if operator == ">":  return value > bound
        if operator == "<=": return value <= bound
        if operator == "<":  return value < bound
        return False