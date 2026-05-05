"""
PATL Semantic Predicates Verifier.
Evalúa predicados PATL sobre snapshots usando Agents, Environments y Automata reales.
"""

from copy import deepcopy
from itertools import product

from sim_core.agents import Agents
from sim_core.environments import Environments


class PATLVerifier:
    """Verificador de predicados PATL sobre snapshots."""

    def __init__(self, automata, distributions, metrics_collector):
        self.automata = automata
        self.distributions = distributions
        self.metrics_collector = metrics_collector

    def verify(self, snapshot, predicates):
        results = []
        for pred in predicates:
            result = self._verify_predicate(snapshot, pred)
            results.append(result)
        return results

    # ── Predicado individual ─────────────────────────

    def _verify_predicate(self, snapshot, pred):
        pred_id = pred["predicate_id"]
        pred_type = pred["type"]
        max_depth = pred.get("max_depth", 20)
        coalition_spec = pred["coalition"]
        adversary_spec = pred.get("adversaries", {})
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")
        quantifier = pred.get("coalition_quantifier", "exists")

        # Cargar snapshot en objetos reales
        agents, environments = self._load_snapshot(snapshot)
        original_agents = self.automata.agents
        original_envs = self.automata.environments
        self.automata.agents = agents
        self.automata.environments = environments

        try:
            coalition_agents = self._resolve_agents(agents, coalition_spec)
            adversary_agents = self._resolve_agents(agents, adversary_spec)

            if not coalition_agents:
                return {
                    "predicate_id": pred_id,
                    "result": "ERROR",
                    "p_value": 0.0,
                    "reason": "No coalition agents in snapshot"
                }

            coalition_strategies = list(self._enumerate_strategies(coalition_agents))
            adversary_strategies = (
                list(self._enumerate_strategies(adversary_agents))
                if adversary_agents else [{}]
            )

            p_game = 0.0
            for c_strat in coalition_strategies:
                worst_p = 1.0
                for a_strat in adversary_strategies:
                    p = self._compute_reachability(
                        agents, environments,
                        coalition_agents, c_strat, a_strat,
                        target_states=self._target_set(coalition_agents, c_strat),
                        max_depth=max_depth,
                        pred_type=pred_type
                    )
                    if p < worst_p:
                        worst_p = p
                if worst_p > p_game:
                    p_game = worst_p

            satisfied = self._evaluate(p_game, bound, operator)
            if quantifier == "forall":
                satisfied = not satisfied

            return {
                "predicate_id": pred_id,
                "result": "SATISFIED" if satisfied else "VIOLATED",
                "p_value": round(p_game, 6),
                "bound": bound,
                "operator": operator
            }
        finally:
            self.automata.agents = original_agents
            self.automata.environments = original_envs

    # ── Carga de snapshot ────────────────────────────

    def _load_snapshot(self, snapshot):
        agents = Agents([], self.distributions, self.metrics_collector, worker_mode=False)
        agents.load_snapshot(snapshot["agents_data"])
        environments = Environments([], self.distributions, self.metrics_collector)
        environments.load_snapshot(snapshot.get("environments_data"))
        return agents, environments

    # ── Resolución de agentes ────────────────────────

    def _resolve_agents(self, agents, spec):
        resolved = []
        if not spec:
            return resolved
        for entry in spec.get("by_type", []):
            agent_type = entry["agent_type"]
            automata_opts = [a["automaton_name"] for a in entry["automata"]]
            targets_map = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            for agent in agents.data:
                if agent.get("agent_type") == agent_type:
                    resolved.append({
                        "agent_id": agent["agent_id"],
                        "automata_options": automata_opts,
                        "target_states_map": targets_map
                    })
        for entry in spec.get("by_id", []):
            agent_id = entry["agent_id"]
            automata_opts = [a["automaton_name"] for a in entry["automata"]]
            targets_map = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            if any(a["agent_id"] == agent_id for a in agents.data):
                resolved.append({
                    "agent_id": agent_id,
                    "automata_options": automata_opts,
                    "target_states_map": targets_map
                })
        return resolved

    # ── Estrategias ──────────────────────────────────

    def _enumerate_strategies(self, agent_list):
        if not agent_list:
            yield {}
            return
        ids = [a["agent_id"] for a in agent_list]
        opts = [a["automata_options"] for a in agent_list]
        for combo in product(*opts):
            yield dict(zip(ids, combo))

    def _target_set(self, coalition_agents, c_strat):
        targets = set()
        for ca in coalition_agents:
            chosen = c_strat.get(ca["agent_id"])
            if chosen and chosen in ca["target_states_map"]:
                targets |= ca["target_states_map"][chosen]
        return targets

    # ── Reachability (DP) ────────────────────────────

    def _compute_reachability(self, agents, environments, coalition_agents,
                              c_strat, a_strat, target_states, max_depth, pred_type):
        # Clonar estado
        agents_copy = Agents([], self.distributions, self.metrics_collector, worker_mode=False)
        agents_copy.load_snapshot(deepcopy(agents.data))
        envs_copy = Environments([], self.distributions, self.metrics_collector)
        envs_copy.load_snapshot(deepcopy(environments.data))

        # Aplicar estrategias
        full_strat = {**c_strat, **a_strat}
        for agent in agents_copy.data:
            if agent["agent_id"] in full_strat:
                agent["automata"] = [full_strat[agent["agent_id"]]]

        if not target_states:
            return 0.0

        self.automata.agents = agents_copy
        self.automata.environments = envs_copy
        try:
            return self._dp(agents_copy, target_states, max_depth, pred_type)
        finally:
            pass  # restore en el caller

    def _dp(self, agents, target_states, depth, pred_type):
        memo = {}

        def recurse(d):
            if d == 0:
                return 0.0
            key = self._hash_state(agents.data)
            if (key, d) in memo:
                return memo[(key, d)]

            if self._target_reached(agents.data, target_states):
                return 1.0
            if self._all_final(agents.data):
                memo[(key, d)] = 0.0
                return 0.0

            total = 0.0
            branches = self._expand_one_step(agents)
            for prob, state_updates in branches:
                if prob == 0.0:
                    continue
                saved = {}
                saved_env = deepcopy(self.automata.environments.data)
                for agent_id, new_state in state_updates.items():
                    agent = next(a for a in agents.data if a["agent_id"] == agent_id)
                    saved[agent_id] = agent.get("current_state")
                    agent["current_state"] = new_state
                total += prob * recurse(d - 1)
                for agent_id, old_state in saved.items():
                    agent = next(a for a in agents.data if a["agent_id"] == agent_id)
                    agent["current_state"] = old_state
                self.automata.environments.data = saved_env

            memo[(key, d)] = total
            return total

        result = recurse(depth)
        return 1.0 - result if pred_type == "invariance" else result

    # ── Expansión de un paso ─────────────────────────

    def _expand_one_step(self, agents):
        branches = [(1.0, {})]

        for agent in agents.data:
            state = agent.get("current_state")
            if state is None:
                continue
            aut_name = (agent.get("automata") or [None])[0]
            if not aut_name:
                continue

            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
            if not aut_def:
                continue
            if state in aut_def["states"].get("final", []):
                continue

            transition = next((t for t in aut_def["transitions"] if t["from"] == state), None)
            if not transition:
                continue

            # Evaluar _X_
            ctx = {"event": {}, "params": agent.get("params", {})}
            _X_ = self._resolve_threshold_value(transition["threshold_value"], ctx)

            # Ramas según thresholds
            new_branches = []
            for base_prob, base_updates in branches:
                remaining = 1.0
                for th in transition.get("thresholds", []):
                    conds = th["threshold_case"]
                    p = self._probability_of_case(_X_, transition["threshold_value"], conds, remaining)
                    if p > 0:
                        session = self.automata.create_session(aut_name, {"signal": aut_name})
                        if session:
                            session.current_state = state
                            session.params = agent.get("params", {}).copy()
                            new_state = session.step()
                            if new_state:
                                updates = dict(base_updates)
                                updates[agent["agent_id"]] = new_state
                                new_branches.append((base_prob * p, updates))
                    remaining -= p
                    if remaining <= 0:
                        break
            branches = new_branches or [(1.0, {})]

        return branches

    def _resolve_threshold_value(self, tv_def, ctx):
        from sim_core.utils.evaluator import resolve_value
        return resolve_value(tv_def, ctx, self.distributions, self.automata.agents, self.automata.environments)

    def _probability_of_case(self, X_val, tv_def, conditions, remaining):
        if tv_def["type"] == "deterministic" or tv_def["type"] == "logic":
            return 1.0 if self._check_conditions(X_val, conditions) else 0.0
        if tv_def["type"] == "probabilistic":
            lower, upper, li, ui = self._conditions_to_interval(conditions)
            if lower is None:
                return 0.0
            dist_name = tv_def["distribution"]
            return self.distributions.probability_interval(dist_name, lower, upper, li, ui)
        return 0.0

    def _check_conditions(self, X_val, conditions):
        for cond in conditions:
            op = cond["operator"]
            val = cond["value"]
            if op == "<" and not (X_val < val):
                return False
            elif op == "<=" and not (X_val <= val):
                return False
            elif op == ">" and not (X_val > val):
                return False
            elif op == ">=" and not (X_val >= val):
                return False
            elif op == "==" and not (X_val == val):
                return False
        return True

    def _conditions_to_interval(self, conditions):
        lower = float("-inf")
        upper = float("inf")
        li = True
        ui = True
        for cond in conditions:
            op = cond["operator"]
            val = cond["value"]
            if op == ">":
                lower = max(lower, val)
                li = False
            elif op == ">=":
                lower = max(lower, val)
                li = True
            elif op == "<":
                upper = min(upper, val)
                ui = False
            elif op == "<=":
                upper = min(upper, val)
                ui = True
            elif op == "==":
                lower = max(lower, val)
                upper = min(upper, val)
                li = True
                ui = True
        if lower > upper:
            return None, None, False, False
        return lower, upper, li, ui

    # ── Helpers ──────────────────────────────────────

    def _hash_state(self, agents_data):
        items = tuple(
            (a["agent_id"], a.get("current_state"), tuple(sorted(a.get("params", {}).items())))
            for a in sorted(agents_data, key=lambda x: x["agent_id"])
        )
        env_items = tuple(
            (e.get("env_id"), tuple(sorted(e.get("params", {}).items())))
            for e in self.automata.environments.data
        )
        return (items, env_items)

    def _target_reached(self, agents_data, target_states):
        return any(a.get("current_state") in target_states for a in agents_data)

    def _all_final(self, agents_data):
        for agent in agents_data:
            state = agent.get("current_state")
            if state is None:
                continue
            aut_name = (agent.get("automata") or [None])[0]
            if not aut_name:
                continue
            aut_def = next((a for a in self.automata.data if a["automaton_name"] == aut_name), None)
            if aut_def and state not in aut_def["states"].get("final", []):
                return False
        return True

    def _evaluate(self, value, bound, operator):
        if operator == ">=":
            return value >= bound
        elif operator == ">":
            return value > bound
        elif operator == "<=":
            return value <= bound
        elif operator == "<":
            return value < bound
        return False