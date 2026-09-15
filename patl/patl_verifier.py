"""
Verificador de predicados PATL_b sobre instantáneas del motor DES.

Semántica (sección de Verificación con PATL de la tesis):

    G, pi |= <<C>>_k^{op d} psi   sii   existe sigma_C en prod_{a en C} Str_{a,k}
                                        tal que para toda sigma_A:  P(psi | sigma_C, sigma_A) op d

Fórmulas de camino acotadas (delta rondas), sobre fórmulas de estado sin coaliciones (verdadero,
pertenencia a Target_C, proposiciones atómicas (autómata, estados finales), negación, conjunción y
disyunción):

* until       phi1 U^{<=delta} phi2
* release     phi1 R^{<=delta} phi2
* next        X phi (una ronda)
* reachability = true U^{<=delta} Target_C      (abreviatura <>^{<=delta} Target_C)
* invariance   = false R^{<=delta} not Target_C (abreviatura []^{<=delta} not Target_C)

Estrategias de la coalición, basadas en observación y con memoria a lo sumo k:

* Una estrategia es sigma_a = (Q_a, act_a, Delta_a, start_a) con |Q_a| = k, act_a y Delta_a definidas sobre
  (modo, observación), donde la observación es el conjunto de proposiciones atómicas de la fórmula que se
  cumplen. Solo importan las observaciones compatibles con continuar la evaluación de la fórmula.
* Deterministas: se enumeran las tablas (act_a, Delta_a) con start_a = 0 (sin pérdida de generalidad).
* Aleatorizadas sin memoria (k = 1): act_a asigna a cada observación una distribución sobre las acciones.
  Se optimiza cuando un solo miembro de la coalición tiene elección y la dimensión de la mezcla es a lo
  sumo 2, por rejilla y refinamiento local (tolerancia del orden de 1e-6). Para k >= 2 se toma el mejor valor
  entre las deterministas con k modos y las aleatorizadas sin memoria, que también tienen memoria a lo sumo k.
* Los adversarios no tienen restricción de memoria ni de información: fijada sigma_C, su mejor respuesta en
  horizonte acotado se obtiene por inducción hacia atrás; eligen al inicio de cada ronda sin conocer la
  realización de la mezcla de la coalición en esa ronda.
* Dirección: para >= y > la coalición maximiza y los adversarios minimizan; para <= y < al revés.
  coalition_quantifier = "forall" cuantifica universalmente sobre sigma_C (extensión declarada).

Modelo de transición del juego (cadena embebida en los instantes de evento):

* El estado global es la instantánea: agentes, con la sesión en curso del agente disparador, y entornos.
* Participan la coalición y los adversarios; los demás agentes permanecen congelados.
* Una ronda activa a cada participante una vez. Activarlo es ejecutar una sesión completa y atómica de un
  autómata: quien tiene una sesión en curso la termina; los demás inician el autómata que su estrategia elige.
  El orden de activación dentro de la ronda es aleatorio y uniforme sobre las permutaciones, y el valor
  promedia sobre él (entrelazado estocástico de eventos).
* Umbral probabilístico: una rama por caso bajo la regla del primer caso que coincide, con masa exacta (CDF,
  soporte Poisson o etiqueta). En continuas, si el valor muestreado se usa después en el autómata, la región
  se divide en celdas de igual masa y cada celda propaga su esperanza condicional (converge al núcleo exacto
  al crecer el número de celdas); si no se usa, basta la masa exacta del caso. Una continua con salida entera
  se trata como discreta sobre los enteros resultantes. Umbral determinista o dinámico: una sola rama.
* Los eventos emitidos durante la verificación no se propagan.

Complejidad por predicado: O(|Str_{C,k}| * delta * |S_delta| * A_A * n! * b), con n los participantes de la
ronda y b los desenlaces distintos de una sesión; la parte aleatorizada multiplica por el tamaño de la rejilla.
"""

import heapq
import itertools
import json
import random
import re
from itertools import product

import numpy as np
from scipy import stats

from core.agents import Agents
from core.environments import Environments
from core.automata import AutomatonSession, SessionError
from core.utils.evaluator import resolve_ephemeral
from core.trace import tracer
from metrics.metrics_collector import MetricsCollector

MAX_STRATEGIES = 10_000
MAX_ADVERSARY_CHOICES = 1_000
MAX_PARTICIPANTS = 50
MAX_ORDER_PARTICIPANTS = 5
MAX_NODES = 200_000
MAX_BRANCHES = 50_000
MAX_SESSION_STEPS = 500
MAX_ATOMS = 4
MAX_INTEGER_SUPPORT = 2_000
PARTITION_TOLERANCE = 1e-6
COMPARE_TOLERANCE = 1e-9
DEFAULT_QUANTILES = 8
GRID_1D = 201
GRID_2D = 41
REFINEMENTS = 3

_DISABLED = MetricsCollector(enabled=False)
_WRITE_PREFIXES = ("write_", "add_", "remove_")


class VerificationError(Exception):
    pass


# ----------------------------------------------------------------------
# Fórmulas de estado
# ----------------------------------------------------------------------
TRUE = {"true": True}
TARGET = {"target": True}


def _atoms(formula, acc):
    if "target" in formula or "prop" in formula:
        acc.append(json.dumps(formula, sort_keys=True))
    elif "not" in formula:
        _atoms(formula["not"], acc)
    elif "and" in formula or "or" in formula:
        for f in formula.get("and", formula.get("or", [])):
            _atoms(f, acc)
    return acc


def _evaluate(formula, truth):
    """truth: función átomo_serializado -> bool."""
    if "true" in formula:
        return True
    if "target" in formula or "prop" in formula:
        return truth(json.dumps(formula, sort_keys=True))
    if "not" in formula:
        return not _evaluate(formula["not"], truth)
    if "and" in formula:
        return all(_evaluate(f, truth) for f in formula["and"])
    if "or" in formula:
        return any(_evaluate(f, truth) for f in formula["or"])
    raise VerificationError(f"fórmula de estado no reconocida: {formula}")


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
        self.pending = pending    # agent_id -> (autómata, estado, ctx, evento) de una sesión en curso
        self._owned_a = set()
        self._owned_e = set()
        self._key = None
        self._agents = None
        self._envs = None

    def child(self):
        # El hijo comparte las copias de entidades del padre. El padre pierde su propiedad, de modo que una
        # escritura posterior sobre el padre vuelve a copiar y no altera a los hijos ya creados.
        self._owned_a = set()
        self._owned_e = set()
        return _Node(dict(self.amods), dict(self.emods), dict(self.last), dict(self.pending))

    def key(self):
        if self._key is None:
            self._key = (
                tuple(sorted(self.last.items())),
                tuple(sorted((pid, a, s, repr(sorted(ctx.items())))
                             for pid, (a, s, ctx, _) in self.pending.items())),
                tuple((i, _content(x)) for i, x in sorted(self.amods.items())),
                tuple((i, _content(x)) for i, x in sorted(self.emods.items())),
            )
        return self._key


# Representación textual de cada copia de entidad, reutilizada mientras no se escriba sobre ella. La entrada
# conserva la referencia al objeto, así que su id no se recicla mientras siga en la caché. Toda escritura pasa
# por los proxies, que invalidan la entrada del objeto modificado.
_CONTENT = {}


def _content(obj):
    if obj is None:
        return None
    entry = _CONTENT.get(id(obj))
    if entry is None or entry[0] is not obj:
        entry = (obj, repr(obj))
        _CONTENT[id(obj)] = entry
    return entry[1]


def _touched(obj):
    if obj is not None:
        _CONTENT.pop(id(obj), None)


class _VerifierDistributions:
    """Vista de las distribuciones para el verificador: nunca muestrea ni consume el generador del motor."""

    def __init__(self, distributions):
        self._d = distributions
        self.rng = random.Random(0)

    def sample(self, name):
        raise SessionError(f"el verificador no muestrea la distribución {name}")

    def __getattr__(self, name):
        return getattr(self._d, name)


class _AgentsProxy:
    """API de Agents sobre un nodo. Las escrituras copian el agente afectado antes de modificarlo."""

    def __init__(self, verifier, node):
        self._v = verifier
        self._node = node
        self._tmp = Agents.__new__(Agents)
        self._tmp.metrics_collector = _DISABLED
        self._tmp.distributions = verifier.safe_distributions
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
        known = list(self._v.base_agents) + list(self._node.amods)
        numbers = [int(i.rsplit("_", 1)[-1]) for i in known
                   if i.rsplit("_", 1)[0] == agent_type and i.rsplit("_", 1)[-1].isdigit()]
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
        # Como en el motor: se retira de membresías, relaciones y canales de todo entorno.
        envs = _EnvsProxy(self._v, self._node)
        for env_id in list(self._v.base_envs) + [i for i in self._node.emods if i not in self._v.base_envs]:
            env = envs._get(env_id)
            if env is None:
                continue
            members = [m.get("agent_id") for m in env.get("members", [])]
            relations = [x for r in env.get("relations", []) for x in r.get("members", [])]
            channels = [x for ch in env.get("channels", []) for x in ch.get("members", [])]
            if agent_id in members or agent_id in relations or agent_id in channels:
                owned = envs._own(env_id)
                _touched(owned)
                envs._tmp.data = [owned]
                Environments.purge_agent(envs._tmp, agent_id)
        return True

    def __getattr__(self, name):
        fn = getattr(Agents, name)

        def call(*args):
            agent = self._own(args[0]) if name.startswith(_WRITE_PREFIXES) else self._get(args[0])
            self._tmp.data = [agent] if agent is not None else []
            if name.startswith(_WRITE_PREFIXES):
                _touched(agent)
            return fn(self._tmp, *args)

        return call


class _EnvsProxy:
    """API de Environments sobre un nodo. Las escrituras copian el entorno afectado antes de modificarlo."""

    def __init__(self, verifier, node):
        self._v = verifier
        self._node = node
        self._tmp = Environments.__new__(Environments)
        self._tmp.metrics_collector = _DISABLED
        self._tmp.distributions = verifier.safe_distributions
        self._tmp.config_data = verifier.envs_config
        self._tmp.event_sink = None
        self._tmp.clock = lambda: verifier.base_time

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
            self._tmp.distributions._d = self._v.distributions
            self._tmp.distributions.rng = random.Random(0)
            if name.startswith(_WRITE_PREFIXES):
                _touched(env)
            return fn(self._tmp, *args)

        return call


# ----------------------------------------------------------------------
# Estrategias
# ----------------------------------------------------------------------
class _Constant:
    """Miembro con una sola acción habilitada."""

    kind = "deterministic"

    def __init__(self, action):
        self.action = action

    def decide(self, mode, obs):
        return self.action, mode

    def describe(self):
        return self.action


class _Table:
    """Estrategia determinista con memoria: act y Delta sobre (modo, observación)."""

    kind = "deterministic"

    def __init__(self, act, delta):
        self.act = act
        self.delta = delta

    def decide(self, mode, obs):
        return self.act[(mode, obs)], self.delta[(mode, obs)]

    def describe(self):
        return {f"q{m}|{sorted(o)}": a for (m, o), a in sorted(self.act.items(), key=str)}


class _Mixed:
    """Estrategia aleatorizada sin memoria: una distribución de acciones por observación, evaluada en rejilla."""

    kind = "mixed"

    def __init__(self, opts, observations, weights):
        self.opts = opts
        self.observations = observations
        self.weights = weights          # obs -> {acción: arreglo de pesos sobre la rejilla}

    def decide(self, mode, obs):
        return self.weights[obs], mode

    def describe(self):
        return "mezcla"


# ----------------------------------------------------------------------
# Verificador
# ----------------------------------------------------------------------
class PATLVerifier:
    def __init__(self, automata, distributions, configs=None, default_memory=1,
                 quantiles=DEFAULT_QUANTILES, mixed_strategies=True):
        self.automata = automata
        self.distributions = distributions
        self.safe_distributions = _VerifierDistributions(distributions)
        self.configs = configs or {}
        self.agents_config = self.configs.get("agents", [])
        self.envs_config = self.configs.get("environments", [])
        memories = default_memory if isinstance(default_memory, (list, tuple)) else [default_memory]
        self.default_memories = list(memories)
        self.quantiles = int(quantiles)
        self.mixed_strategies = bool(mixed_strategies)
        self.base_agents = {}
        self.base_envs = {}
        self.base_time = None
        self._propagates_cache = {}
        self._layout_cache = {}
        self._round_cache = {}

    def verify(self, snapshot, predicates):
        """Una fila por predicado y por cota de memoria k."""
        # Las claves de nodo son relativas a la instantánea base: la caché de rondas no cruza instantáneas.
        self._round_cache = {}
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
                value, strategy = self._value(snapshot, pred, k, cache)
            except (VerificationError, SessionError, ValueError, KeyError, ZeroDivisionError) as e:
                reason = f"{type(e).__name__}: {e}" if not isinstance(e, (VerificationError, SessionError)) else str(e)
                tracer.emit("patl", "predicate_error", predicate=pred["predicate_id"], memory_k=k, reason=reason)
                rows.append({**base, "result": "ERROR", "value": "", "strategy": "", "reason": reason})
                continue
            satisfied = self._compare(value, bound, operator)
            tracer.emit("patl", "predicate_result", predicate=pred["predicate_id"], memory_k=k, value=value,
                        operator=operator, bound=bound, strategy=strategy,
                        result="SATISFIED" if satisfied else "VIOLATED")
            rows.append({**base, "result": "SATISFIED" if satisfied else "VIOLATED",
                         "value": value, "strategy": strategy, "reason": ""})
        return rows

    def _value(self, snapshot, pred, memory, cache):
        agents_data = snapshot["agents_data"] or []
        self.base_agents = {a["agent_id"]: a for a in agents_data}
        self.base_envs = {e["env_id"]: e for e in (snapshot.get("environments_data") or [])}
        # Tiempo simulado de la captura: fecha los eventos de canal escritos durante el juego.
        self.base_time = snapshot.get("sim_time")
        trigger_id = snapshot.get("agent_id")

        coalition = self._resolve_group(pred["coalition"], trigger_id, with_targets=True)
        if not coalition:
            raise VerificationError("coalición vacía")
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
        if len(participants) > MAX_PARTICIPANTS:
            raise VerificationError(f"{len(participants)} participantes exceden el límite {MAX_PARTICIPANTS}")
        self._participants = participants
        self._coalition = coalition
        self._adversaries = adversaries
        self._targets = {m["id"]: m["targets"] for m in coalition}

        # Fórmula de camino.
        ptype = pred["type"]
        if ptype == "reachability":
            left, right = TRUE, TARGET
        elif ptype == "invariance":
            left, right = {"not": TRUE}, {"not": TARGET}
        elif ptype in ("until", "release"):
            left, right = pred["left"], pred["right"]
        elif ptype == "next":
            left, right = TRUE, pred["right"]
        else:
            raise VerificationError(f"tipo de predicado no reconocido: {ptype}")
        self._ptype = "until" if ptype == "reachability" else "release" if ptype == "invariance" else ptype
        self._left, self._right = left, right
        atoms = sorted(set(_atoms(left, []) + _atoms(right, [])))
        if len(atoms) > MAX_ATOMS:
            raise VerificationError(f"la fórmula usa {len(atoms)} proposiciones; el límite es {MAX_ATOMS}")
        if any("target" in json.loads(a) for a in atoms) and not any(m["targets"] for m in coalition):
            raise VerificationError("la coalición no declara ningún estado objetivo")
        self._atoms = atoms
        self._observations = self._decision_observations(atoms)
        self._depth = 1 if ptype == "next" else pred.get("max_depth", 20)

        # La sesión en curso del agente disparador es parte del estado global de la instantánea.
        pending = {}
        ag = self.base_agents.get(trigger_id)
        if ag and trigger_id in participants:
            active = ag.get("current_automaton")
            if active in self.automata.by_name and ag.get("current_state") is not None:
                event = snapshot.get("trigger_event") or {"signal": active, "agent_id": trigger_id}
                pending[trigger_id] = (active, ag["current_state"], dict(ag.get("session_ctx") or {}), event)
        root = _Node({}, {}, {}, pending)

        maximize = pred.get("probability_operator", ">=") in (">=", ">")
        self._adversary_ext = np.minimum if maximize else np.maximum
        forall = pred.get("coalition_quantifier", "exists") == "forall"
        pick = (lambda a, b: min(a, b)) if (forall == maximize) else (lambda a, b: max(a, b))

        cache_key = (memory, json.dumps(pred, sort_keys=True))
        if cache_key in cache:
            return cache[cache_key]

        strategies = self._deterministic_strategies(coalition, memory)
        if tracer.on("patl"):
            tracer.emit("patl", "predicate_start", predicate=pred["predicate_id"], memory_k=memory,
                        snapshot_agent=trigger_id, snapshot_automaton=snapshot.get("automaton_name"),
                        snapshot_state=snapshot.get("state"), type=ptype, depth=self._depth,
                        operator=pred.get("probability_operator", ">="), bound=pred["probability_bound"],
                        quantifier=pred.get("coalition_quantifier", "exists"),
                        coalition=[(m["id"], m["opts"], sorted(m["targets"])) for m in coalition],
                        adversaries=[(m["id"], m["opts"]) for m in adversaries],
                        pending={pid: (a, s) for pid, (a, s, _, _) in pending.items()},
                        observations=[sorted(o) for o in self._observations],
                        deterministic_strategies=len(strategies))

        best, best_kind = None, "deterministic"
        for sigma in strategies:
            self._grid = None
            self._nodes = 0
            self._memo = {}
            _CONTENT.clear()
            v = float(self._V(root, self._initial_modes(sigma), 0, sigma))
            tracer.emit("patl", "strategy_value", predicate=pred["predicate_id"], memory_k=memory,
                        strategy={pid: s.describe() for pid, s in sigma.items()}, value=v, nodes=self._nodes)
            if best is None or pick(best, v) != best:
                best = v

        # Las estrategias aleatorizadas son sin memoria: su valor es el mismo para toda cota k.
        mixed_key = ("mixed", json.dumps(pred, sort_keys=True))
        if mixed_key not in cache:
            cache[mixed_key] = self._mixed_value(root, coalition, maximize, forall, pred)
        mixed = cache[mixed_key]
        if mixed is not None and pick(best, mixed) != best and abs(mixed - best) > 1e-12:
            best, best_kind = mixed, "mixed"
        cache[cache_key] = (best, best_kind)
        return best, best_kind

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

    # --------------------------------------------------------------
    def _truth(self, node):
        def truth(atom):
            f = json.loads(atom)
            if "target" in f:
                return any(node.last.get(pid) in tset for pid, tset in self._targets.items())
            p = f["prop"]
            group = p.get("group", "coalition")
            ids = ([m["id"] for m in self._coalition] if group == "coalition" else
                   [m["id"] for m in self._adversaries] if group == "adversaries" else list(self._participants))
            wanted = {(p["automaton"], s) for s in p["states"]}
            return any(node.last.get(pid) in wanted for pid in ids)
        return truth

    def _observation(self, node):
        truth = self._truth(node)
        return frozenset(a for a in self._atoms if truth(a))

    def _status(self, node, r):
        """Valor terminal de la fórmula de camino en el nodo, o None si la evaluación continúa."""
        truth = self._truth(node)
        if self._ptype == "until":
            if _evaluate(self._right, truth):
                return 1.0
            if not _evaluate(self._left, truth):
                return 0.0
            return 0.0 if r == self._depth else None
        if self._ptype == "release":
            if not _evaluate(self._right, truth):
                return 0.0
            if _evaluate(self._left, truth):
                return 1.0
            return 1.0 if r == self._depth else None
        # next
        if r == 1:
            return 1.0 if _evaluate(self._right, truth) else 0.0
        return None

    def _decision_observations(self, atoms):
        """Observaciones posibles en los nodos donde la evaluación continúa y alguien decide."""
        observations = []
        for bits in product([False, True], repeat=len(atoms)):
            true_atoms = {a for a, b in zip(atoms, bits) if b}
            truth = lambda a, s=true_atoms: a in s
            if self._ptype == "until":
                keep = (not _evaluate(self._right, truth)) and _evaluate(self._left, truth)
            elif self._ptype == "release":
                keep = _evaluate(self._right, truth) and not _evaluate(self._left, truth)
            else:
                keep = True
            if keep:
                observations.append(frozenset(true_atoms))
        return observations or [frozenset()]

    def _deterministic_strategies(self, coalition, memory):
        per_agent = []
        for m in coalition:
            if len(m["opts"]) == 1:
                per_agent.append([_Constant(m["opts"][0])])
                continue
            cells = [(q, o) for q in range(memory) for o in self._observations]
            options = []
            for acts in product(m["opts"], repeat=len(cells)):
                for nexts in product(range(memory), repeat=len(cells)):
                    options.append(_Table(dict(zip(cells, acts)), dict(zip(cells, nexts))))
                    if len(options) > MAX_STRATEGIES:
                        raise VerificationError(f"las estrategias de {m['id']} exceden el límite {MAX_STRATEGIES}")
            per_agent.append(options)
        total = 1
        for s in per_agent:
            total *= len(s)
        if total > MAX_STRATEGIES:
            raise VerificationError(f"{total} estrategias de coalición exceden el límite {MAX_STRATEGIES}")
        ids = [m["id"] for m in coalition]
        return [dict(zip(ids, combo)) for combo in product(*per_agent)]

    @staticmethod
    def _initial_modes(sigma):
        return tuple(sorted((pid, 0) for pid in sigma))

    # --------------------------------------------------------------
    def _mixed_value(self, root, coalition, maximize, forall, pred):
        """Mejor valor con estrategias aleatorizadas sin memoria, o None si no aplica."""
        if not self.mixed_strategies:
            return None
        choosers = [m for m in coalition if len(m["opts"]) > 1]
        if len(choosers) != 1:
            return None
        chooser = choosers[0]
        opts, observations = chooser["opts"], self._observations
        dims = (len(opts) - 1) * len(observations)
        if dims > 2:
            return None

        want_max = maximize != forall

        def evaluate(points):
            # points: arreglo (G, dims) con coordenadas libres de cada simplex
            weights, col = {}, 0
            for o in observations:
                w = {}
                rest = np.ones(len(points))
                for a in opts[:-1]:
                    w[a] = points[:, col]
                    rest = rest - points[:, col]
                    col += 1
                w[opts[-1]] = rest
                weights[o] = w
            sigma = {m["id"]: (_Mixed(opts, observations, weights) if m is chooser else _Constant(m["opts"][0]))
                     for m in coalition}
            self._grid = len(points)
            self._nodes = 0
            self._memo = {}
            _CONTENT.clear()
            values = np.broadcast_to(self._V(root, self._initial_modes(sigma), 0, sigma), (len(points),))
            return np.asarray(values, dtype=float)

        def feasible(points):
            ok = np.all(points >= -1e-12, axis=1)
            if len(opts) == 3:
                ok &= points.sum(axis=1) <= 1 + 1e-12
            return points[ok]

        if dims == 1:
            points = np.linspace(0.0, 1.0, GRID_1D).reshape(-1, 1)
        else:
            axis = np.linspace(0.0, 1.0, GRID_2D)
            points = feasible(np.array(list(product(axis, axis))))
        values = evaluate(points)
        idx = int(np.argmax(values) if want_max else np.argmin(values))
        center, best, width = points[idx], float(values[idx]), 1.0
        for _ in range(REFINEMENTS):
            width /= 10.0
            if dims == 1:
                local = np.clip(np.linspace(center[0] - width, center[0] + width, 41), 0, 1).reshape(-1, 1)
            else:
                ax0 = np.linspace(center[0] - width, center[0] + width, 11)
                ax1 = np.linspace(center[1] - width, center[1] + width, 11)
                local = np.clip(np.array(list(product(ax0, ax1))), 0, 1)
                local = feasible(local)
            vals = evaluate(local)
            j = int(np.argmax(vals) if want_max else np.argmin(vals))
            if (vals[j] > best) if want_max else (vals[j] < best):
                best, center = float(vals[j]), local[j]
        self._grid = None
        tracer.emit("patl", "mixed_strategy", predicate=pred["predicate_id"], chooser=chooser["id"],
                    options=opts, mixture=[float(x) for x in center], value=best)
        return best

    # --------------------------------------------------------------
    def _V(self, node, modes, r, sigma):
        key = (node.key(), modes, r)
        if key in self._memo:
            return self._memo[key]

        status = self._status(node, r)
        if status is not None:
            self._memo[key] = status
            return status

        self._nodes += 1
        if self._nodes > MAX_NODES:
            raise VerificationError(f"el espacio de estados excede {MAX_NODES} nodos")

        obs = self._observation(node)
        mode_of = dict(modes)
        pure, mixtures, next_modes = {}, {}, dict(mode_of)
        for pid, strat in sigma.items():
            if pid in node.pending:
                continue
            if obs not in self._observations and not isinstance(strat, _Constant):
                raise VerificationError(f"observación no prevista {sorted(obs)} en un nodo de decisión")
            decision, nxt = strat.decide(mode_of[pid], obs)
            next_modes[pid] = nxt
            if isinstance(decision, dict):
                mixtures[pid] = decision
            else:
                pure[pid] = decision
        next_modes = tuple(sorted(next_modes.items()))

        idle_adv = [m for m in self._adversaries if m["id"] not in node.pending]
        n_choices = 1
        for m in idle_adv:
            n_choices *= len(m["opts"])
            if n_choices > MAX_ADVERSARY_CHOICES:
                raise VerificationError(
                    f"las elecciones conjuntas de los adversarios exceden el límite {MAX_ADVERSARY_CHOICES}")
        choices = list(product(*[m["opts"] for m in idle_adv])) if idle_adv else [()]

        realizations = [({}, 1.0)]
        for pid, weights in mixtures.items():
            realizations = [({**acts, pid: a}, w * weights[a]) for acts, w in realizations for a in weights]

        values = []
        for choice in choices:
            total = 0.0
            for mixed_acts, weight in realizations:
                acts = {**pure, **mixed_acts}
                for m, c in zip(idle_adv, choice):
                    acts[m["id"]] = c
                outcomes = self._round(node, acts)
                expected = sum(p * self._V(child, next_modes, r + 1, sigma) for p, child in outcomes)
                total = total + weight * expected
                if tracer.on("patl_rounds") and self._grid is None:
                    tracer.emit("patl_rounds", "round", round=r + 1, last_before=dict(node.last),
                                actions=acts, value=float(expected),
                                outcomes=[{"p": p, "last": dict(child.last),
                                           "agents_modified": sorted(child.amods), "envs_modified": sorted(child.emods)}
                                          for p, child in outcomes])
            values.append(total)
        if len(values) == 1:
            val = values[0]
        else:
            val = values[0]
            for v in values[1:]:
                val = self._adversary_ext(val, v)
            if tracer.on("patl_rounds") and self._grid is None:
                tracer.emit("patl_rounds", "adversary_choice", round=r + 1, choices=[list(ch) for ch in choices],
                            values=[float(v) for v in values], extremum=self._adversary_ext.__name__,
                            value=float(val))
        if self._grid is None:
            val = float(val)
        self._memo[key] = val
        return val

    def _round(self, node, actions):
        """Desenlaces de una ronda: promedio uniforme sobre los órdenes de activación de quienes actúan."""
        # Los desenlaces dependen solo del estado global, de las acciones y de los participantes, no de la
        # estrategia: se reutilizan entre estrategias, puntos de la malla y memorias de la misma instantánea.
        round_key = (node.key(), tuple(sorted((p, a) for p, a in actions.items() if a is not None)),
                     tuple(sorted(self._participants)))
        cached = self._round_cache.get(round_key)
        if cached is not None:
            return cached
        outcomes = self._round_outcomes(node, actions)
        self._round_cache[round_key] = outcomes
        return outcomes

    def _round_outcomes(self, node, actions):
        acting = sorted(pid for pid in self._participants if pid in node.pending or actions.get(pid) is not None)
        if len(acting) > MAX_ORDER_PARTICIPANTS:
            raise VerificationError(
                f"{len(acting)} participantes activos exceden el límite {MAX_ORDER_PARTICIPANTS} del orden uniforme")
        orders = list(itertools.permutations(acting)) or [()]
        weight = 1.0 / len(orders)
        merged = {}
        for order in orders:
            current = [(weight, node.child())]
            for pid in order:
                nxt = {}
                for p, n in current:
                    for q, c in self._session_outcomes(n, pid, actions.get(pid)):
                        k = c.key()
                        if k in nxt:
                            nxt[k] = (nxt[k][0] + p * q, nxt[k][1])
                        else:
                            nxt[k] = (p * q, c)
                if len(nxt) > MAX_BRANCHES:
                    raise VerificationError(f"la ronda excede {MAX_BRANCHES} desenlaces")
                current = list(nxt.values())
            for p, c in current:
                k = c.key()
                if k in merged:
                    merged[k] = (merged[k][0] + p, merged[k][1])
                else:
                    merged[k] = (p, c)
        return list(merged.values())

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

    def _session_layout(self, aut_def):
        """Orden topológico de los estados (None si hay ciclos) y variables muestreadas que no se propagan."""
        name = aut_def["automaton_name"]
        if name not in self._layout_cache:
            succ = {}
            for t in aut_def["transitions"]:
                succ.setdefault(t["from"], set()).update(th["to"] for th in t["thresholds"])
            states = set(succ) | {s for targets in succ.values() for s in targets}
            indegree = {s: 0 for s in states}
            for targets in succ.values():
                for s in targets:
                    indegree[s] += 1
            queue = sorted(s for s in states if indegree[s] == 0)
            order = []
            while queue:
                s = queue.pop(0)
                order.append(s)
                for x in sorted(succ.get(s, ())):
                    indegree[x] -= 1
                    if indegree[x] == 0:
                        queue.append(x)
            rank = {s: i for i, s in enumerate(order)} if len(order) == len(states) else None
            dead = set()
            for t in aut_def["transitions"]:
                tv_key, tv_def = AutomatonSession.threshold_value(t)
                if isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic" and not self._propagates(aut_def, t):
                    dead.add(tv_key)
            self._layout_cache[name] = (rank, dead)
        return self._layout_cache[name]

    def _session_outcomes(self, node, pid, action):
        """
        Desenlaces (probabilidad, nodo) de una sesión completa del participante pid.

        Las ramas se expanden en orden topológico de los estados del autómata (por número de pasos si tiene
        ciclos). Dos ramas en el mismo estado, con el mismo estado global y el mismo contexto se fusionan
        sumando su probabilidad. Del contexto se omiten los valores muestreados que no se propagan, porque
        ninguna acción ni transición los vuelve a leer; la fusión no altera la distribución de desenlaces.
        """
        if pid in node.pending:
            aut, state, ctx, event = node.pending.pop(pid)
            node._key = None
        elif action is None:
            return [(1.0, node)]
        else:
            aut, state, ctx = action, self.automata.by_name[action]["states"]["initial"], None
            event = {"signal": action, "agent_id": pid}

        aut_def = self.automata.by_name[aut]
        rank, dead = self._session_layout(aut_def)
        frontier, heap, seq, finished = {}, [], itertools.count(), {}

        def push(prob, n, st, cx, steps):
            if self._is_final(aut_def, st):
                n.last[pid] = (aut, st)
                n._key = None
                k = n.key()
                finished[k] = (finished[k][0] + prob, finished[k][1]) if k in finished else (prob, n)
                return
            if steps >= MAX_SESSION_STEPS:
                raise VerificationError(f"{aut} no llega a un estado final en {MAX_SESSION_STEPS} pasos")
            live_ctx = repr(sorted((v, x) for v, x in (cx or {}).items() if v not in dead))
            k = (st, n.key(), live_ctx)
            if k in frontier:
                entry = frontier[k]
                entry[0] += prob
                entry[4] = max(entry[4], steps)
            else:
                frontier[k] = [prob, n, st, cx, steps]
                heapq.heappush(heap, (rank[st] if rank is not None else steps, next(seq), k))

        push(1.0, node, state, ctx, 0)
        while heap:
            _, _, k = heapq.heappop(heap)
            prob, n, st, cx, steps = frontier.pop(k)

            self._bind(n)
            session = AutomatonSession(self.automata, aut_def, event, live=False, ctx=cx, state=st)
            transition = session.transition()
            _, tv_def = session.threshold_value(transition)

            if isinstance(tv_def, dict) and tv_def.get("type") == "probabilistic":
                cases = self._probabilistic_cases(session, transition, tv_def["distribution"])
                base_ctx = session.ctx
                for i, (mass, value, case) in enumerate(cases):
                    child = n if i == len(cases) - 1 else n.child()
                    self._bind(child)
                    s = AutomatonSession(self.automata, aut_def, event, live=False, ctx=base_ctx, state=st)
                    s.apply_case(transition, case, value)
                    child._key = None
                    push(prob * mass, child, s.current_state, s.ctx, steps + 1)
            else:
                X = session.resolve_threshold(transition)
                case = session.select_case(transition, X)
                if case is None:
                    raise VerificationError(f"{aut}::{st} ningún caso coincide con el valor {X}")
                session.apply_case(transition, case, X)
                n._key = None
                push(prob, n, session.current_state, session.ctx, steps + 1)
        return list(finished.values())

    def _propagates(self, aut_def, transition):
        """¿El valor muestreado de este umbral se usa fuera de sus propias condiciones de caso?"""
        key = (aut_def["automaton_name"], transition["from"])
        if key not in self._propagates_cache:
            tv_key, _ = AutomatonSession.threshold_value(transition)
            name = re.escape(tv_key.lstrip("$"))
            # Se busca el uso en las acciones de sus propios casos y en las demás transiciones; las condiciones
            # de sus casos y su propia declaración no cuentan como uso.
            # El nombre de una distribución no es una variable: se omite para no confundir una variable $x con
            # una distribución llamada x que usa otra transición.
            own_actions = [th.get("actions", []) for th in transition["thresholds"]]
            others = [{**t, "threshold_value": {k: ({f: x for f, x in v.items() if f != "distribution"}
                                                    if isinstance(v, dict) else v)
                                                for k, v in t["threshold_value"].items()}}
                      for t in aut_def["transitions"] if t is not transition]
            blob = json.dumps({"own": own_actions, "others": others})
            self._propagates_cache[key] = bool(re.search(rf'"\$?{name}"|\${name}\b', blob))
        return self._propagates_cache[key]

    def _probabilistic_cases(self, session, transition, dist_name):
        """Ramas (probabilidad, valor representativo, caso) de un umbral probabilístico, sin muestrear."""
        entry = self.distributions.samplers[dist_name]
        family = entry["family"]
        output = entry.get("output_type", "float")
        where = f"{session.automaton_name}::{session.current_state}"
        out = []

        if family == "categorical" or family == "poisson" or output == "int":
            if family == "categorical":
                cast = {"int": lambda x: int(float(x)), "float": float}.get(output, str)
                support = []
                for label, p in zip(entry["labels"], entry["probabilities"]):
                    try:
                        support.append((cast(label), p))
                    except (TypeError, ValueError):
                        raise VerificationError(f"{where}: la etiqueta {label} no se convierte a {output}")
            elif family == "poisson":
                support = self._poisson_support(entry)
                if output == "int":
                    support = [(int(v), p) for v, p in support]
            else:
                support = self._integer_support(dist_name, entry)
            for value, mass in support:
                if mass <= 0:
                    continue
                case = session.select_case(transition, value)
                if case is None:
                    raise VerificationError(f"{where}: ningún caso coincide con el valor {value}")
                out.append((mass, value, case))
            total = sum(m for m, _, _ in out)
        else:
            # Regla del primer caso que coincide: la región efectiva de un caso es su intervalo menos las
            # regiones de los casos anteriores.
            tv_key, _ = session.threshold_value(transition)
            propagates = self._propagates(session.automaton_def, transition)
            total = 0.0
            covered = []
            for case in transition.get("thresholds", []):
                interval = self._interval(tv_key, case["threshold_case"], session.ctx)
                if interval is None:
                    continue
                pieces = [interval]
                for previous in covered:
                    pieces = [p for piece in pieces for p in self._minus(piece, previous)]
                covered.append(interval)
                if propagates:
                    for low, high, li, ui in pieces:
                        for m, v in self.distributions.quantile_cells(dist_name, low, high, li, ui, self.quantiles):
                            if m > 0:
                                out.append((m, v, case))
                                total += m
                else:
                    weighted = []
                    for low, high, li, ui in pieces:
                        m = self.distributions.probability_interval(dist_name, low, high, li, ui)
                        if m > 0:
                            weighted.append((m, self.distributions.conditional_mean(dist_name, low, high, li, ui)))
                    mass = sum(m for m, _ in weighted)
                    if mass <= 0:
                        continue
                    if any(v is None for _, v in weighted):
                        raise VerificationError(f"{where}: región sin valor representativo")
                    out.append((mass, sum(m * v for m, v in weighted) / mass, case))
                    total += mass

        if abs(total - 1.0) > PARTITION_TOLERANCE:
            raise VerificationError(f"{where}: la masa de los casos suma {total:.6f}, no 1")
        return out

    def _integer_support(self, dist_name, entry):
        """Continua con salida entera: masa de cada entero producido por int() (truncamiento hacia cero)."""
        trunc = entry["truncation"]
        lo, hi = int(np.floor(trunc["min"])), int(np.ceil(trunc["max"]))
        if hi - lo > MAX_INTEGER_SUPPORT:
            raise VerificationError(f"el soporte entero de {dist_name} excede {MAX_INTEGER_SUPPORT} valores")
        support = []
        for v in range(lo, hi + 1):
            if v > 0:
                m = self.distributions.probability_interval(dist_name, v, v + 1, True, False)
            elif v < 0:
                m = self.distributions.probability_interval(dist_name, v - 1, v, False, True)
            else:
                m = self.distributions.probability_interval(dist_name, -1, 1, False, False)
            if m > 0:
                support.append((v, m))
        return support

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
    def _minus(a, r):
        """a menos r, con extremos abiertos o cerrados; descarta puntos aislados (medida cero)."""
        alo, ahi, ali, aui = a
        rlo, rhi, rli, rui = r
        disjoint = (rhi < alo or (rhi == alo and not (rui and ali)) or
                    rlo > ahi or (rlo == ahi and not (rli and aui)))
        if disjoint:
            return [a]
        pieces = []
        if rlo > alo or (rlo == alo and ali and not rli):
            pieces.append((alo, rlo, ali, not rli))
        if rhi < ahi or (rhi == ahi and aui and not rui):
            pieces.append((rhi, ahi, not rui, aui))
        return [p for p in pieces if p[0] < p[1]]

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
        # La cota es un racional; se tolera el error de redondeo de punto flotante.
        if operator == ">=": return value >= bound - COMPARE_TOLERANCE
        if operator == ">":  return value > bound + COMPARE_TOLERANCE
        if operator == "<=": return value <= bound + COMPARE_TOLERANCE
        if operator == "<":  return value < bound - COMPARE_TOLERANCE
        return False
