"""
Verificador de predicados PATL_b sobre instantáneas del motor DES.

Semántica implementada (sección de Verificación con PATL de la tesis):

    G, pi |= <<C>>_k^{op d} psi   sii   existe sigma_C en prod_{a en C} Str_{a,k}
                                        tal que para toda sigma_A:  P(psi | sigma_C, sigma_A) op d

* psi es alcanzabilidad acotada  <>^{<=delta} Target_C  o invarianza acotada  []^{<=delta} not Target_C.
* Target_C contiene únicamente pares (autómata, estado) de la coalición.
* Las estrategias de la coalición son deterministas, basadas en observación y con memoria k:
  sigma_a = (Q_a, act_a, Delta_a, start_a) con |Q_a| = k. Como AP son las proposiciones de
  pertenencia a Target_C y todo estado que satisface Target_C es absorbente para el valor de psi,
  la observación en los estados donde todavía se decide es siempre el conjunto vacío; por eso
  act_a y Delta_a se tabulan sobre los modos de memoria.
* Los adversarios no tienen restricción de memoria ni de información: fijada sigma_C, su mejor
  respuesta en horizonte acotado se obtiene por inducción hacia atrás sobre (estado, modos, profundidad).
* La dirección del operador depende de op: para >= y > la coalición maximiza y los adversarios
  minimizan; para <= y < la coalición minimiza y los adversarios maximizan. La invarianza se
  evalúa directamente sobre psi (valor 0 al tocar Target_C), no por complemento del valor de alcanzabilidad.
* coalition_quantifier = "forall" (extensión fuera de la gramática de la tesis) cuantifica también
  universalmente sobre sigma_C.

Modelo de transición del juego:

* El estado global es la instantánea: agentes (con la sesión activa del agente disparador) y entornos.
* Participan la coalición y los adversarios; los demás agentes permanecen congelados.
* En cada paso cada participante con sesión activa aplica una transición de su autómata; un
  participante sin sesión elige (según su estrategia) cuál de sus autómatas iniciar y aplica su primera
  transición. Al llegar a un estado final la sesión termina y el agente queda inactivo en el horizonte.
  Los participantes se aplican en orden de agent_id y la probabilidad conjunta es el producto.
* Umbral probabilístico: una rama por caso con probabilidad exacta tomada de la CDF (continuas),
  de la masa de cada valor del soporte (Poisson) o de cada etiqueta (categóricas). En continuas el valor que
  se propaga a las acciones es la esperanza condicional dentro de la región del caso. No se muestrea.
* Umbral determinista o dinámico: se evalúa sobre el estado del nodo y produce una sola rama.
* Los eventos emitidos durante la verificación no se propagan.

Complejidad por predicado: O(|Str_{C,k}| * delta * |S_delta| * A_A * b).
"""

import json
import random
from copy import deepcopy
from itertools import product

import numpy as np
from scipy import stats

from core.agents import Agents
from core.environments import Environments
from core.automata import AutomatonSession
from metrics.metrics_collector import MetricsCollector

MAX_STRATEGIES = 100_000
MAX_NODES = 200_000
MAX_BRANCHES = 100_000
PARTITION_TOLERANCE = 1e-6


class VerificationError(Exception):
    pass


# ----------------------------------------------------------------------
# Estado del juego con copia en escritura
# ----------------------------------------------------------------------
class _Node:
    """Estado global del juego: modificaciones sobre la instantánea base y configuración de sesiones."""

    __slots__ = ("amods", "emods", "cfg", "_key")

    def __init__(self, amods, emods, cfg):
        self.amods = amods      # agent_id -> dict del agente (copia) o None si fue eliminado
        self.emods = emods      # env_id -> dict del entorno (copia)
        self.cfg = cfg          # agent_id -> (automata, estado, ctx, terminado)
        self._key = None

    def child(self):
        return _Node(dict(self.amods), dict(self.emods), dict(self.cfg))

    def key(self):
        if self._key is None:
            dump = lambda o: json.dumps(o, sort_keys=True, default=str)
            self._key = (
                tuple((pid, a, s, dump(ctx), done) for pid, (a, s, ctx, done) in sorted(self.cfg.items())),
                tuple((i, dump(v)) for i, v in sorted(self.amods.items())),
                tuple((i, dump(v)) for i, v in sorted(self.emods.items())),
            )
        return self._key


class _View:
    """
    Expone la API de Agents o Environments sobre un nodo. Toda escritura copia primero la
    entidad afectada, de modo que el nodo padre y la instantánea base no se modifican.
    """

    def __init__(self, cls, base_list, id_key, mods, distributions, config_data):
        self._cls = cls
        self._base_list = base_list
        self._base_ids = {e[id_key] for e in base_list}
        self._base_by_id = {e[id_key]: e for e in base_list}
        self._id_key = id_key
        self._mods = mods
        self._owned = set()
        obj = cls.__new__(cls)
        obj.metrics_collector = MetricsCollector(enabled=False)
        obj.distributions = distributions
        obj.config_data = config_data
        obj.worker_mode = False
        obj._running = False
        obj._on_agent_added = None
        obj.automata = None
        obj.snapshot_manager = None
        self._obj = obj

    def _materialize(self):
        mods = self._mods
        data = [mods.get(e[self._id_key], e) for e in self._base_list]
        data = [e for e in data if e is not None]
        data.extend(v for k, v in mods.items() if k not in self._base_ids and v is not None)
        return data

    def _own(self, entity_id):
        if entity_id in self._owned:
            return
        src = self._mods.get(entity_id, self._base_by_id.get(entity_id))
        if src is None:
            return
        self._mods[entity_id] = deepcopy(src)
        self._owned.add(entity_id)

    def __getattr__(self, name):
        fn = getattr(self._cls, name)
        if not callable(fn):
            raise AttributeError(name)

        def call(*args):
            structural_add = name.startswith("add_") and name in ("add_agent", "add_env")
            structural_remove = name in ("remove_agent", "remove_env")
            if not structural_add and (name.startswith(("write_", "add_", "remove_"))) and args:
                self._own(args[0])
            self._obj.data = self._materialize()
            before = {e[self._id_key] for e in self._obj.data}
            result = fn(self._obj, *args)
            if structural_add:
                for e in self._obj.data:
                    if e[self._id_key] not in before:
                        self._mods[e[self._id_key]] = e
                        self._owned.add(e[self._id_key])
            elif structural_remove and result:
                self._mods[args[0]] = None
            return result

        return call


# ----------------------------------------------------------------------
# Verificador
# ----------------------------------------------------------------------
class PATLVerifier:
    def __init__(self, automata, distributions, configs=None, default_memory=1):
        self.automata = automata
        self.distributions = distributions
        self.configs = configs or {}
        self.default_memory = default_memory

    def verify(self, snapshot, predicates):
        return [self._verify_predicate(snapshot, p) for p in predicates]

    # --------------------------------------------------------------
    def _verify_predicate(self, snapshot, pred):
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")
        memory = pred.get("max_memory", self.default_memory)
        base = {"predicate_id": pred["predicate_id"], "bound": bound, "operator": operator,
                "memory_k": memory}
        try:
            value = self._value(snapshot, pred, memory)
        except VerificationError as e:
            return {**base, "result": "ERROR", "value": "", "reason": str(e)}
        return {**base, "result": "SATISFIED" if self._compare(value, bound, operator) else "VIOLATED",
                "value": round(value, 6), "reason": ""}

    def _value(self, snapshot, pred, memory):
        agents_data = snapshot["agents_data"] or []
        envs_data = snapshot.get("environments_data") or []
        agents_by_id = {a["agent_id"]: a for a in agents_data}
        trigger_id = snapshot.get("agent_id")

        coalition = self._resolve_group(agents_by_id, pred["coalition"], trigger_id)
        if not coalition:
            raise VerificationError("coalición vacía")
        coalition_ids = {m["id"] for m in coalition}
        if pred.get("adversaries"):
            adversaries = self._resolve_group(agents_by_id, pred["adversaries"], trigger_id)
        else:
            # Sin adversarios explícitos se consideran todos los agentes fuera de C.
            adversaries = [{"id": a["agent_id"], "opts": list(a.get("automata", [])), "targets": set()}
                           for a in agents_data if a["agent_id"] not in coalition_ids]
        if coalition_ids & {m["id"] for m in adversaries}:
            raise VerificationError("un agente aparece en la coalición y en los adversarios")

        participants = {m["id"]: m for m in coalition + adversaries}
        order = sorted(participants)
        self._base_agents = agents_data
        self._base_envs = envs_data
        self._agents_config = self.configs.get("agents", [])
        self._envs_config = self.configs.get("environments", [])

        root_cfg = {}
        for pid in order:
            ag = agents_by_id[pid]
            active = ag.get("current_automaton")
            if active and active in participants[pid]["opts"] and ag.get("current_state") is not None:
                aut_def = self.automata.by_name[active]
                state = ag["current_state"]
                root_cfg[pid] = (active, state, dict(ag.get("session_ctx") or {}),
                                 self._is_done(aut_def, state))
            else:
                root_cfg[pid] = (None, None, None, False)
        root = _Node({}, {}, root_cfg)

        pred_type = pred["type"]
        depth = pred.get("max_depth", 20)
        operator = pred.get("probability_operator", ">=")
        maximize = operator in (">=", ">")
        adversary_ext = min if maximize else max
        forall = pred.get("coalition_quantifier", "exists") == "forall"
        coalition_ext = adversary_ext if forall else (max if maximize else min)

        strategies = self._coalition_strategies(coalition, memory)
        targets = {m["id"]: m["targets"] for m in coalition}
        self._nodes = 0

        best = None
        for sigma in strategies:
            memo = {}
            modes0 = tuple(0 for _ in coalition)
            v = self._V(root, modes0, depth, sigma, coalition, adversaries, order, targets,
                        pred_type, adversary_ext, memo)
            best = v if best is None else coalition_ext(best, v)
        return best

    # --------------------------------------------------------------
    def _resolve_group(self, agents_by_id, spec, trigger_id):
        group = []
        for entry in spec:
            opts = [a["automaton_name"] for a in entry["automata"]]
            tmap = {a["automaton_name"]: set(a["target_states"]) for a in entry["automata"]}
            max_agents = entry.get("max_agents", 10)

            if "agent_type" in entry:
                matching = sorted((a for a in agents_by_id.values() if a.get("agent_type") == entry["agent_type"]),
                                  key=lambda x: x["agent_id"])
            else:
                matching = [agents_by_id[i] for i in entry.get("agent_id", []) if i in agents_by_id]

            # El agente disparador, si pertenece al grupo, siempre se incluye.
            chosen = [a for a in matching if a["agent_id"] == trigger_id][:1]
            rest = [a for a in matching if a["agent_id"] != trigger_id]
            slots = max_agents - len(chosen)
            if len(rest) > slots:
                rng = random.Random("".join(a["agent_id"] for a in rest))
                rest = sorted(rng.sample(rest, slots), key=lambda x: x["agent_id"])
            chosen.extend(rest)

            for ag in chosen:
                allowed = [o for o in opts if o in ag.get("automata", [])]
                if not allowed:
                    raise VerificationError(
                        f"{ag['agent_id']} no tiene asignado ninguno de los autómatas {opts}")
                group.append({
                    "id": ag["agent_id"],
                    "opts": allowed,
                    "targets": {(a, s) for a in allowed for s in tmap[a]},
                })
        return group

    def _coalition_strategies(self, coalition, memory):
        """Estrategias deterministas con k modos: act: Q -> Act y Delta: Q -> Q, con start = 0."""
        per_agent = []
        for m in coalition:
            if len(m["opts"]) == 1:
                # Todas las tablas prescriben la misma acción: una sola estrategia efectiva.
                per_agent.append([((m["opts"][0],) * memory, tuple(range(memory)))])
            else:
                acts = list(product(m["opts"], repeat=memory))
                deltas = list(product(range(memory), repeat=memory))
                per_agent.append([(a, d) for a in acts for d in deltas])
        total = 1
        for s in per_agent:
            total *= len(s)
        if total > MAX_STRATEGIES:
            raise VerificationError(f"{total} estrategias de coalición exceden el límite {MAX_STRATEGIES}")
        return list(product(*per_agent))

    # --------------------------------------------------------------
    def _V(self, node, modes, n, sigma, coalition, adversaries, order, targets, pred_type, adversary_ext, memo):
        key = (node.key(), modes, n)
        if key in memo:
            return memo[key]

        if self._in_target(node, targets):
            val = 1.0 if pred_type == "reachability" else 0.0
        elif n == 0:
            val = 0.0 if pred_type == "reachability" else 1.0
        else:
            self._nodes += 1
            if self._nodes > MAX_NODES:
                raise VerificationError(f"el espacio de estados excede {MAX_NODES} nodos")

            actions = {}
            for i, m in enumerate(coalition):
                act_table, _ = sigma[i]
                actions[m["id"]] = act_table[modes[i]]
            next_modes = tuple(sigma[i][1][modes[i]] for i in range(len(coalition)))

            idle_adv = [m for m in adversaries if node.cfg[m["id"]][0] is None and not node.cfg[m["id"]][3]]
            choices = list(product(*[m["opts"] for m in idle_adv])) if idle_adv else [()]

            absorbing = 0.0 if pred_type == "reachability" else 1.0
            values = []
            for choice in choices:
                acts = dict(actions)
                for m, c in zip(idle_adv, choice):
                    acts[m["id"]] = c
                children = self._joint_step(node, order, acts)
                if children is None:
                    values.append(absorbing)
                    continue
                values.append(sum(p * self._V(child, next_modes, n - 1, sigma, coalition, adversaries,
                                              order, targets, pred_type, adversary_ext, memo)
                                  for p, child in children))
            val = adversary_ext(values)

        memo[key] = val
        return val

    @staticmethod
    def _in_target(node, targets):
        for pid, tset in targets.items():
            aut, state, _, _ = node.cfg[pid]
            if aut is not None and (aut, state) in tset:
                return True
        return False

    def _joint_step(self, node, order, actions):
        current = [(1.0, node)]
        moved = False
        for pid in order:
            nxt = []
            for p, n in current:
                branches = self._agent_branches(n, pid, actions.get(pid))
                if branches is None:
                    nxt.append((p, n))
                    continue
                moved = True
                nxt.extend((p * q, c) for q, c in branches)
            if len(nxt) > MAX_BRANCHES:
                raise VerificationError(f"el paso conjunto excede {MAX_BRANCHES} ramas")
            current = nxt
        return current if moved else None

    # --------------------------------------------------------------
    def _is_done(self, aut_def, state):
        if state in aut_def["states"].get("final", []):
            return True
        return not any(t["from"] == state for t in aut_def["transitions"])

    def _bind(self, node):
        self.automata.agents = _View(Agents, self._base_agents, "agent_id", node.amods,
                                     self.distributions, self._agents_config)
        self.automata.environments = _View(Environments, self._base_envs, "env_id", node.emods,
                                           self.distributions, self._envs_config)

    def _session(self, node, pid, aut, state, ctx):
        aut_def = self.automata.by_name[aut]
        return AutomatonSession(self.automata, aut_def, {"signal": aut, "agent_id": pid},
                                live=False, ctx=ctx, state=state)

    def _agent_branches(self, node, pid, action):
        aut, state, ctx, done = node.cfg[pid]
        if done:
            return None
        if aut is None:
            if action is None:
                return None
            aut, state, ctx = action, self.automata.by_name[action]["states"]["initial"], None
            if self._is_done(self.automata.by_name[aut], state):
                return None

        probe = node.child()
        self._bind(probe)
        session = self._session(probe, pid, aut, state, ctx)
        transition = session.transition()
        tv_key, tv_def = session.threshold_value(transition)
        aut_def = self.automata.by_name[aut]

        branches = []
        if isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic":
            base_ctx = session.ctx
            for prob, value, case in self._probabilistic_cases(session, transition, tv_def["distribution"]):
                child = node.child()
                self._bind(child)
                s = self._session(child, pid, aut, state, base_ctx)
                s.apply_case(transition, case, value)
                child.cfg[pid] = (aut, s.current_state, s.ctx, self._is_done(aut_def, s.current_state))
                branches.append((prob, child))
        else:
            X = session.resolve_threshold(transition)
            case = session.select_case(transition, X)
            if case is None:
                raise VerificationError(f"{aut}::{state} ningún caso coincide con el valor {X}")
            session.apply_case(transition, case, X)
            probe.cfg[pid] = (aut, session.current_state, session.ctx, self._is_done(aut_def, session.current_state))
            branches.append((1.0, probe))
        return branches

    def _probabilistic_cases(self, session, transition, dist_name):
        """Ramas (probabilidad, valor representativo, caso) de un umbral probabilístico, sin muestrear."""
        entry = self.distributions.samplers[dist_name]
        family = entry["family"]
        where = f"{session.automaton_name}::{session.current_state}"
        out = []

        if family in ("categorical", "poisson"):
            if family == "categorical":
                support = list(zip(entry["labels"], entry["probabilities"]))
            else:
                support = self._poisson_support(entry)
            merged = {}
            for value, mass in support:
                if mass <= 0:
                    continue
                case = session.select_case(transition, value)
                if case is None:
                    raise VerificationError(f"{where}: ningún caso coincide con el valor {value}")
                out.append((mass, value, case))
            total = sum(m for m, _, _ in out)
        else:
            tv_key, _ = session.threshold_value(transition)
            total = 0.0
            for case in transition.get("thresholds", []):
                interval = self._interval(tv_key, case["threshold_case"])
                if interval is None:
                    continue
                low, high, li, ui = interval
                mass = self.distributions.probability_interval(dist_name, low, high, li, ui)
                if mass <= 0:
                    continue
                value = self.distributions.conditional_mean(dist_name, low, high, li, ui)
                if value is None or session.select_case(transition, value) is not case:
                    raise VerificationError(f"{where}: los casos no forman una partición del soporte")
                out.append((mass, value, case))
                total += mass

        if abs(total - 1.0) > PARTITION_TOLERANCE:
            raise VerificationError(f"{where}: la masa de los casos suma {total:.6f}, no 1")
        return out

    @staticmethod
    def _poisson_support(entry):
        lam = entry["params"]["lambda"]
        trunc = entry["truncation"]
        k_min = int(np.ceil(trunc["min"])) if trunc else 0
        k_max = int(np.floor(trunc["max"])) if trunc else int(stats.poisson.ppf(1 - 1e-12, mu=lam))
        ks = np.arange(max(k_min, 0), k_max + 1)
        pmf = stats.poisson.pmf(ks, mu=lam)
        pmf = pmf / pmf.sum()
        return [(float(k), float(p)) for k, p in zip(ks, pmf)]

    @staticmethod
    def _interval(tv_key, conditions):
        """Región de un caso sobre la variable continua; None si tiene medida cero."""
        name = tv_key.lstrip("$")
        low, high = float("-inf"), float("inf")
        li = ui = True
        for c in conditions:
            if c["variable"].lstrip("$") != name:
                raise VerificationError(f"el caso condiciona una variable distinta de {tv_key}")
            op, v = c["operator"], c["value"]
            if v is None:
                if op == "==":
                    return None
                continue
            if isinstance(v, (str, bool)):
                if op == "==":
                    return None
                continue
            if op == ">":
                if v >= low: low, li = v, False
            elif op == ">=":
                if v > low: low, li = v, True
            elif op == "<":
                if v <= high: high, ui = v, False
            elif op == "<=":
                if v < high: high, ui = v, True
            elif op == "==":
                return None
        if low > high:
            return None
        return low, high, li, ui

    @staticmethod
    def _compare(value, bound, operator):
        if operator == ">=": return value >= bound
        if operator == ">":  return value > bound
        if operator == "<=": return value <= bound
        if operator == "<":  return value < bound
        return False
