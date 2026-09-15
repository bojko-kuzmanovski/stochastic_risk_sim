"""
Verificador de predicados PATL_b sobre instantáneas del motor DES.

Semántica implementada (sección de Verificación con PATL de la tesis):

    G, pi |= <<C>>_k^{op d} psi   sii   existe sigma_C en prod_{a en C} Str_{a,k}
                                        tal que para toda sigma_A:  P(psi | sigma_C, sigma_A) op d

* psi es alcanzabilidad acotada  <>^{<=delta} Target_C  o invarianza acotada  []^{<=delta} not Target_C.
* Target_C contiene únicamente pares (autómata, estado final) de la coalición.
* Las estrategias de la coalición son deterministas, basadas en observación y con memoria k:
  sigma_a = (Q_a, act_a, Delta_a, start_a) con |Q_a| = k. Las proposiciones atómicas son las de
  pertenencia a Target_C y todo estado que satisface Target_C cierra el valor de psi, así que la
  observación en los estados donde se decide es siempre el conjunto vacío. Una estrategia queda
  determinada entonces por la secuencia de acciones que su autómata de memoria produce en delta
  rondas; se enumeran las secuencias distintas generadas por las tablas (act_a, Delta_a) con k modos.
* Los adversarios no tienen restricción de memoria ni de información: fijada sigma_C, su mejor
  respuesta en horizonte acotado se obtiene por inducción hacia atrás sobre (estado, ronda).
* La dirección del operador depende de op: para >= y > la coalición maximiza y los adversarios
  minimizan; para <= y < la coalición minimiza y los adversarios maximizan. La invarianza se evalúa
  sobre su propia fórmula de camino (valor 0 al tocar Target_C).
* coalition_quantifier = "forall" (extensión fuera de la gramática de la tesis) cuantifica también
  universalmente sobre sigma_C.

Modelo de transición del juego (cadena embebida en los instantes de evento):

* El estado global es la instantánea: agentes, con la sesión en curso del agente disparador, y entornos.
* Participan la coalición y los adversarios; los demás agentes permanecen congelados.
* Una ronda activa a cada participante una vez, en orden de agent_id. Activarlo es ejecutar una sesión
  completa y atómica de un autómata: el participante con sesión en curso la termina; los demás eligen,
  según su estrategia, cuál de sus autómatas asignados iniciar. delta cuenta rondas.
* Dentro de una sesión, un umbral probabilístico abre una rama por caso con probabilidad exacta tomada de
  la CDF (continuas), de la masa de cada valor del soporte (Poisson) o de cada etiqueta (categóricas). En
  continuas el valor que se propaga a las acciones es la esperanza condicional dentro de la región del caso.
  No se muestrea. Un umbral determinista o dinámico se evalúa sobre el estado y produce una sola rama.
* Los eventos emitidos durante la verificación no se propagan.

Complejidad por predicado: O(|Seq_{C,k}| * delta * |S_delta| * A_A * b), con b el número de desenlaces
distintos de una ronda.
"""

import random
from itertools import product

import numpy as np
from scipy import stats

from core.agents import Agents
from core.environments import Environments
from core.automata import AutomatonSession
from core.utils.evaluator import resolve_ephemeral
from core.trace import tracer
from metrics.metrics_collector import MetricsCollector

MAX_STRATEGIES = 10_000
MAX_NODES = 200_000
MAX_BRANCHES = 50_000
MAX_SESSION_STEPS = 500
PARTITION_TOLERANCE = 1e-6

_DISABLED = MetricsCollector(enabled=False)
_WRITE_PREFIXES = ("write_", "add_", "remove_")


class VerificationError(Exception):
    pass


# ----------------------------------------------------------------------
# Estado del juego con copia en escritura
# ----------------------------------------------------------------------
class _Node:
    """Estado global del juego: modificaciones sobre la instantánea base y situación de los participantes."""

    __slots__ = ("amods", "emods", "last", "pending", "_owned_a", "_owned_e", "_key", "_agents", "_envs")

    def __init__(self, amods, emods, last, pending):
        self.amods = amods        # agent_id -> dict del agente (copia) o None si fue eliminado
        self.emods = emods        # env_id -> dict del entorno (copia)
        self.last = last          # agent_id -> (autómata, estado final) de su última sesión
        self.pending = pending    # agent_id -> (autómata, estado, ctx) de una sesión en curso
        self._owned_a = set()
        self._owned_e = set()
        self._key = None
        self._agents = None
        self._envs = None

    def child(self):
        return _Node(dict(self.amods), dict(self.emods), dict(self.last), dict(self.pending))

    def key(self):
        if self._key is None:
            self._key = (
                tuple(sorted(self.last.items())),
                tuple(sorted((pid, a, s, repr(sorted(ctx.items()))) for pid, (a, s, ctx) in self.pending.items())),
                repr(sorted(self.amods.items())),
                repr(sorted(self.emods.items())),
            )
        return self._key


class _AgentsProxy:
    """API de Agents sobre un nodo. Las escrituras copian el agente afectado antes de modificarlo."""

    def __init__(self, verifier, node):
        self._v = verifier
        self._node = node
        self._tmp = Agents.__new__(Agents)
        self._tmp.metrics_collector = _DISABLED
        self._tmp.distributions = verifier.distributions
        self._tmp.config_data = verifier.agents_config

    def _get(self, agent_id):
        if agent_id in self._node.amods:
            return self._node.amods[agent_id]
        return self._v.base_agents.get(agent_id)

    def _own(self, agent_id):
        node = self._node
        if agent_id in node._owned_a:
            return node.amods[agent_id]
        src = self._get(agent_id)
        if src is None:
            return None
        copy = dict(src)
        copy["params"] = dict(src.get("params", {}))
        node.amods[agent_id] = copy
        node._owned_a.add(agent_id)
        return copy

    def _current(self):
        node = self._node
        agents = [node.amods.get(i, a) for i, a in self._v.base_agents.items()]
        agents.extend(a for i, a in node.amods.items() if i not in self._v.base_agents)
        return [a for a in agents if a is not None]

    def get_all_agents(self, agent_type):
        return [a["agent_id"] for a in self._current() if a.get("agent_type") == agent_type]

    def add_agent(self, agent_type):
        entry = next((a for a in self._v.agents_config if a["agent_type"] == agent_type), None)
        if entry is None:
            return None
        numbers = [int(a["agent_id"].rsplit("_", 1)[-1]) for a in self._current()
                   if a.get("agent_type") == agent_type and a["agent_id"].rsplit("_", 1)[-1].isdigit()]
        agent_id = f"{agent_type}_{max(numbers, default=0) + 1}"
        params = {k: self._v.expected_value(vdef) for k, vdef in entry.get("params", {}).items()}
        agent = {"agent_id": agent_id, "agent_type": agent_type,
                 "automata": entry.get("automata", []), "params": params}
        self._node.amods[agent_id] = agent
        self._node._owned_a.add(agent_id)
        return agent_id

    def remove_agent(self, agent_id):
        if self._get(agent_id) is None:
            return False
        self._node.amods[agent_id] = None
        return True

    def __getattr__(self, name):
        fn = getattr(Agents, name)

        def call(*args):
            agent = self._own(args[0]) if name.startswith(_WRITE_PREFIXES) else self._get(args[0])
            self._tmp.data = [agent] if agent is not None else []
            return fn(self._tmp, *args)

        return call


class _EnvsProxy:
    """API de Environments sobre un nodo. Las escrituras copian el entorno afectado antes de modificarlo."""

    def __init__(self, verifier, node):
        self._v = verifier
        self._node = node
        self._tmp = Environments.__new__(Environments)
        self._tmp.metrics_collector = _DISABLED
        self._tmp.distributions = verifier.distributions
        self._tmp.config_data = verifier.envs_config

    def _get(self, env_id):
        if env_id in self._node.emods:
            return self._node.emods[env_id]
        return self._v.base_envs.get(env_id)

    def _own(self, env_id):
        node = self._node
        if env_id in node._owned_e:
            return node.emods[env_id]
        src = self._get(env_id)
        if src is None:
            return None
        copy = {k: (dict(v) if isinstance(v, dict) else
                    [dict(x) if isinstance(x, dict) else x for x in v] if isinstance(v, list) else v)
                for k, v in src.items()}
        for key in ("relations", "channels", "members"):
            for item in copy.get(key, []):
                for sub in ("members", "events"):
                    if isinstance(item.get(sub), list):
                        item[sub] = list(item[sub])
                for sub in ("metadata", "agent_rol"):
                    if isinstance(item.get(sub), dict):
                        item[sub] = dict(item[sub])
        node.emods[env_id] = copy
        node._owned_e.add(env_id)
        return copy

    def __getattr__(self, name):
        fn = getattr(Environments, name)

        def call(*args):
            env = self._own(args[0]) if name.startswith(_WRITE_PREFIXES) else self._get(args[0])
            self._tmp.data = [env] if env is not None else []
            return fn(self._tmp, *args)

        return call


# ----------------------------------------------------------------------
# Verificador
# ----------------------------------------------------------------------
class PATLVerifier:
    def __init__(self, automata, distributions, configs=None, default_memory=1):
        self.automata = automata
        self.distributions = distributions
        self.configs = configs or {}
        self.agents_config = self.configs.get("agents", [])
        self.envs_config = self.configs.get("environments", [])
        memories = default_memory if isinstance(default_memory, (list, tuple)) else [default_memory]
        self.default_memories = list(memories)
        self.base_agents = {}
        self.base_envs = {}

    def verify(self, snapshot, predicates):
        """Una fila por predicado y por cota de memoria k."""
        rows = []
        for pred in predicates:
            rows.extend(self._verify_predicate(snapshot, pred))
        return rows

    # --------------------------------------------------------------
    def _verify_predicate(self, snapshot, pred):
        bound = pred["probability_bound"]
        operator = pred.get("probability_operator", ">=")
        memories = [pred["max_memory"]] if "max_memory" in pred else self.default_memories
        rows = []
        cache = {}
        for k in memories:
            base = {"predicate_id": pred["predicate_id"], "bound": bound, "operator": operator, "memory_k": k}
            try:
                value = self._value(snapshot, pred, k, cache)
            except VerificationError as e:
                tracer.emit("patl", "predicate_error", predicate=pred["predicate_id"], memory_k=k, reason=str(e))
                rows.append({**base, "result": "ERROR", "value": "", "reason": str(e)})
                continue
            satisfied = self._compare(value, bound, operator)
            tracer.emit("patl", "predicate_result", predicate=pred["predicate_id"], memory_k=k, value=value,
                        operator=operator, bound=bound, result="SATISFIED" if satisfied else "VIOLATED")
            rows.append({**base, "result": "SATISFIED" if satisfied else "VIOLATED",
                         "value": round(value, 12), "reason": ""})
        return rows

    def _value(self, snapshot, pred, memory, cache):
        agents_data = snapshot["agents_data"] or []
        self.base_agents = {a["agent_id"]: a for a in agents_data}
        self.base_envs = {e["env_id"]: e for e in (snapshot.get("environments_data") or [])}
        trigger_id = snapshot.get("agent_id")

        coalition = self._resolve_group(pred["coalition"], trigger_id, with_targets=True)
        if not coalition:
            raise VerificationError("coalición vacía")
        if not any(m["targets"] for m in coalition):
            raise VerificationError("la coalición no declara ningún estado objetivo")
        coalition_ids = {m["id"] for m in coalition}
        if pred.get("adversaries"):
            adversaries = self._resolve_group(pred["adversaries"], trigger_id, with_targets=False)
        else:
            # Sin adversarios explícitos se consideran todos los agentes fuera de C.
            adversaries = [{"id": a["agent_id"], "opts": list(a.get("automata", []))}
                           for a in agents_data if a["agent_id"] not in coalition_ids and a.get("automata")]
        if coalition_ids & {m["id"] for m in adversaries}:
            raise VerificationError("un agente aparece en la coalición y en los adversarios")

        participants = {m["id"]: m for m in coalition + adversaries}
        self._order = sorted(participants)
        self._participants = participants

        pending = {}
        ag = self.base_agents.get(trigger_id)
        if ag and trigger_id in participants:
            active = ag.get("current_automaton")
            if active in participants[trigger_id]["opts"] and ag.get("current_state") is not None:
                pending[trigger_id] = (active, ag["current_state"], dict(ag.get("session_ctx") or {}))
        root = _Node({}, {}, {}, pending)

        depth = pred.get("max_depth", 20)
        sequences = self._coalition_sequences(coalition, memory, depth, pending)
        cache_key = tuple(sequences)
        if cache_key in cache:
            return cache[cache_key]

        self._pred_type = pred["type"]
        self._depth = depth
        self._coalition = coalition
        self._adversaries = adversaries
        self._targets = {m["id"]: m["targets"] for m in coalition}
        maximize = pred.get("probability_operator", ">=") in (">=", ">")
        self._adversary_ext = min if maximize else max
        forall = pred.get("coalition_quantifier", "exists") == "forall"
        coalition_ext = self._adversary_ext if forall else (max if maximize else min)
        self._nodes = 0

        if tracer.on("patl"):
            tracer.emit("patl", "predicate_start", predicate=pred["predicate_id"], memory_k=memory,
                        snapshot_agent=trigger_id, snapshot_automaton=snapshot.get("automaton_name"),
                        snapshot_state=snapshot.get("state"), type=pred["type"], depth=depth,
                        operator=pred.get("probability_operator", ">="), bound=pred["probability_bound"],
                        quantifier=pred.get("coalition_quantifier", "exists"),
                        coalition=[(m["id"], m["opts"], sorted(m["targets"])) for m in coalition],
                        adversaries=[(m["id"], m["opts"]) for m in adversaries],
                        pending={pid: (a, s) for pid, (a, s, _) in pending.items()},
                        coalition_ext=coalition_ext.__name__, adversary_ext=self._adversary_ext.__name__,
                        sequences=[list(s) for s in sequences])

        best = None
        for seq in sequences:
            self._seq = seq
            self._memo = {}
            v = self._V(root, 0)
            tracer.emit("patl", "sequence_value", predicate=pred["predicate_id"], memory_k=memory,
                        sequence=list(seq), value=v, nodes=self._nodes)
            best = v if best is None else coalition_ext(best, v)
        cache[cache_key] = best
        return best

    # --------------------------------------------------------------
    def _resolve_group(self, spec, trigger_id, with_targets):
        group = []
        for entry in spec:
            opts = [a["automaton_name"] for a in entry["automata"]]
            tmap = {a["automaton_name"]: set(a.get("target_states", [])) for a in entry["automata"]}
            max_agents = entry.get("max_agents", 10)

            if "agent_type" in entry:
                matching = sorted((a for a in self.base_agents.values() if a.get("agent_type") == entry["agent_type"]),
                                  key=lambda x: x["agent_id"])
            else:
                matching = [self.base_agents[i] for i in entry.get("agent_id", []) if i in self.base_agents]

            # El agente disparador, si pertenece al grupo, siempre se incluye.
            chosen = [a for a in matching if a["agent_id"] == trigger_id][:1]
            rest = [a for a in matching if a["agent_id"] != trigger_id]
            slots = max(max_agents - len(chosen), 0)
            if len(rest) > slots:
                rng = random.Random("".join(a["agent_id"] for a in rest))
                rest = sorted(rng.sample(rest, slots), key=lambda x: x["agent_id"])
            chosen.extend(rest)

            for ag in chosen:
                allowed = [o for o in opts if o in ag.get("automata", [])]
                if not allowed:
                    raise VerificationError(
                        f"{ag['agent_id']} no tiene asignado ninguno de los autómatas {opts}")
                member = {"id": ag["agent_id"], "opts": allowed}
                if with_targets:
                    targets = set()
                    for a in allowed:
                        finals = set(self.automata.by_name[a]["states"].get("final", []))
                        if not tmap[a] <= finals:
                            raise VerificationError(
                                f"los estados objetivo {sorted(tmap[a] - finals)} de {a} no son finales")
                        targets |= {(a, s) for s in tmap[a]}
                    member["targets"] = targets
                group.append(member)
        return group

    def _coalition_sequences(self, coalition, memory, depth, pending):
        """
        Secuencias de acciones de las estrategias deterministas con k modos (act: Q -> Act, Delta: Q -> Q,
        start = 0). En la ronda donde el agente termina una sesión en curso no decide.
        """
        per_agent = []
        for m in coalition:
            decision_rounds = depth - (1 if m["id"] in pending else 0)
            if decision_rounds <= 0:
                per_agent.append([()])
                continue
            seqs = set()
            if len(m["opts"]) == 1:
                seqs.add((m["opts"][0],) * decision_rounds)
            else:
                for act in product(m["opts"], repeat=memory):
                    for delta in product(range(memory), repeat=memory):
                        mode, seq = 0, []
                        for _ in range(decision_rounds):
                            seq.append(act[mode])
                            mode = delta[mode]
                        seqs.add(tuple(seq))
            per_agent.append(sorted(seqs))
        total = 1
        for s in per_agent:
            total *= len(s)
        if total > MAX_STRATEGIES:
            raise VerificationError(f"{total} estrategias de coalición exceden el límite {MAX_STRATEGIES}")
        return [tuple(zip([m["id"] for m in coalition], combo)) for combo in product(*per_agent)]

    # --------------------------------------------------------------
    def _V(self, node, r):
        key = (node.key(), r)
        if key in self._memo:
            return self._memo[key]

        reach = self._pred_type == "reachability"
        if self._in_target(node):
            val = 1.0 if reach else 0.0
        elif r == self._depth:
            val = 0.0 if reach else 1.0
        else:
            self._nodes += 1
            if self._nodes > MAX_NODES:
                raise VerificationError(f"el espacio de estados excede {MAX_NODES} nodos")

            actions = {pid: self._coalition_action(pid, seq, r, node) for pid, seq in self._seq}

            idle_adv = [m for m in self._adversaries if m["id"] not in node.pending]
            choices = list(product(*[m["opts"] for m in idle_adv])) if idle_adv else [()]

            values = []
            for choice in choices:
                acts = dict(actions)
                for m, c in zip(idle_adv, choice):
                    acts[m["id"]] = c
                outcomes = self._round(node, acts)
                values.append(sum(p * self._V(child, r + 1) for p, child in outcomes))
                if tracer.on("patl_rounds"):
                    tracer.emit("patl_rounds", "round", round=r + 1, last_before=dict(node.last),
                                actions=acts, value=values[-1],
                                outcomes=[{"p": p, "last": dict(child.last),
                                           "agents_modified": sorted(child.amods), "envs_modified": sorted(child.emods)}
                                          for p, child in outcomes])
            val = self._adversary_ext(values)
            if tracer.on("patl_rounds") and len(values) > 1:
                tracer.emit("patl_rounds", "adversary_choice", round=r + 1, choices=[list(ch) for ch in choices],
                            values=values, extremum=self._adversary_ext.__name__, value=val)

        self._memo[key] = val
        return val

    def _coalition_action(self, pid, seq, r, node):
        if pid in node.pending:
            return None
        # Si el agente empezó con una sesión en curso, su primera decisión ocurre en la ronda 1.
        started_pending = len(seq) < self._depth
        idx = r - 1 if started_pending else r
        if idx < 0 or idx >= len(seq):
            return None
        return seq[idx]

    def _in_target(self, node):
        for pid, tset in self._targets.items():
            last = node.last.get(pid)
            if last is not None and last in tset:
                return True
        return False

    def _round(self, node, actions):
        current = [(1.0, node.child())]
        for pid in self._order:
            nxt = {}
            for p, n in current:
                outcomes = self._session_outcomes(n, pid, actions.get(pid))
                for q, c in outcomes:
                    k = c.key()
                    if k in nxt:
                        nxt[k] = (nxt[k][0] + p * q, nxt[k][1])
                    else:
                        nxt[k] = (p * q, c)
            if len(nxt) > MAX_BRANCHES:
                raise VerificationError(f"la ronda excede {MAX_BRANCHES} desenlaces")
            current = list(nxt.values())
        return current

    # --------------------------------------------------------------
    def _is_final(self, aut_def, state):
        if state in aut_def["states"].get("final", []):
            return True
        return not any(t["from"] == state for t in aut_def["transitions"])

    def _bind(self, node):
        if node._agents is None:
            node._agents = _AgentsProxy(self, node)
            node._envs = _EnvsProxy(self, node)
        self.automata.agents = node._agents
        self.automata.environments = node._envs

    def _session_outcomes(self, node, pid, action):
        """Desenlaces (probabilidad, nodo) de una sesión completa del participante pid."""
        if pid in node.pending:
            aut, state, ctx = node.pending.pop(pid)
            node._key = None
        elif action is None:
            return [(1.0, node)]
        else:
            aut, state, ctx = action, self.automata.by_name[action]["states"]["initial"], None

        aut_def = self.automata.by_name[aut]
        finished = []
        stack = [(1.0, node, state, ctx, 0)]
        while stack:
            prob, n, st, cx, steps = stack.pop()
            if self._is_final(aut_def, st):
                n.last[pid] = (aut, st)
                n._key = None
                finished.append((prob, n))
                continue
            if steps >= MAX_SESSION_STEPS:
                raise VerificationError(f"{aut} no llega a un estado final en {MAX_SESSION_STEPS} pasos")

            self._bind(n)
            session = AutomatonSession(self.automata, aut_def, {"signal": aut, "agent_id": pid},
                                       live=False, ctx=cx, state=st)
            transition = session.transition()
            _, tv_def = session.threshold_value(transition)

            if isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic":
                cases = self._probabilistic_cases(session, transition, tv_def["distribution"])
                base_ctx = session.ctx
                for i, (mass, value, case) in enumerate(cases):
                    child = n if i == len(cases) - 1 else n.child()
                    self._bind(child)
                    s = AutomatonSession(self.automata, aut_def, {"signal": aut, "agent_id": pid},
                                         live=False, ctx=base_ctx, state=st)
                    s.apply_case(transition, case, value)
                    child._key = None
                    stack.append((prob * mass, child, s.current_state, s.ctx, steps + 1))
            else:
                X = session.resolve_threshold(transition)
                case = session.select_case(transition, X)
                if case is None:
                    raise VerificationError(f"{aut}::{st} ningún caso coincide con el valor {X}")
                session.apply_case(transition, case, X)
                n._key = None
                stack.append((prob, n, session.current_state, session.ctx, steps + 1))
        return finished

    def _probabilistic_cases(self, session, transition, dist_name):
        """Ramas (probabilidad, valor representativo, caso) de un umbral probabilístico, sin muestrear."""
        entry = self.distributions.samplers[dist_name]
        family = entry["family"]
        where = f"{session.automaton_name}::{session.current_state}"
        out = []

        if family in ("categorical", "poisson"):
            support = (list(zip(entry["labels"], entry["probabilities"])) if family == "categorical"
                       else self._poisson_support(entry))
            by_case = {}
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
                interval = self._interval(tv_key, case["threshold_case"], session.ctx)
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

    def expected_value(self, vdef):
        """Valor determinista de un parámetro para agentes creados durante la verificación."""
        if vdef.get("type") != "probabilistic":
            return vdef.get("value")
        entry = self.distributions.samplers[vdef["distribution"]]
        if entry["family"] == "categorical":
            return max(zip(entry["probabilities"], entry["labels"]))[1]
        return self.distributions.conditional_mean(vdef["distribution"], float("-inf"), float("inf"))

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
    def _interval(tv_key, conditions, ctx):
        """Región de un caso sobre la variable continua; None si tiene medida cero."""
        name = tv_key.lstrip("$")
        low, high = float("-inf"), float("inf")
        li = ui = True
        for c in conditions:
            if c["variable"].lstrip("$") != name:
                raise VerificationError(f"el caso condiciona una variable distinta de {tv_key}")
            op, v = c["operator"], c["value"]
            if isinstance(v, str) and "$" in v:
                v = resolve_ephemeral(v, ctx)
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
