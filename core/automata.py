import os
import sys
import json
from jsonschema import validate
from typing import Any, Optional

from core.utils.evaluator import resolve_value, call_method, resolve_ephemeral
from core.events import Events


class Automata:
    def __init__(self, config_data, distributions, metrics_collector):
        schema_path = "schemas/automata.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)

        validate(instance=config_data, schema=schema)

        names = [d["automaton_name"] for d in config_data]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"automaton_name must be unique. Duplicates: {list(set(duplicates))}"
            )

        self.data = config_data
        self.by_name = {d["automaton_name"]: d for d in config_data}
        self.distributions = distributions
        self.metrics_collector = metrics_collector
        self.agents = None
        self.environments = None
        # Receptor de eventos dinámicos: el motor DES los inserta en el calendario Q.
        self.event_sink = None

    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments

    def set_event_sink(self, sink):
        self.event_sink = sink

    def create_session(self, signal: str, event: dict, live=True) -> Optional["AutomatonSession"]:
        automaton_def = self.by_name.get(signal)
        if not automaton_def:
            return None
        return AutomatonSession(self, automaton_def, event, live)


class AutomatonSession:
    """
    Sesión de un autómata estocástico. La función de evolución se separa en
    tau_dist (resolver el valor del umbral) y tau_dec (elegir el caso y aplicarlo).

    live=True: ejecución dentro del motor DES (muestrea, registra métricas y emite eventos).
    live=False: ejecución dentro del verificador PATL (no muestrea ni emite eventos).
    """

    def __init__(self, automata: Automata, automaton_def: dict, event: dict, live=True,
                 ctx: Optional[dict] = None, state: Optional[str] = None):
        self.automata = automata
        self.automaton_def = automaton_def
        self.automaton_name = automaton_def["automaton_name"]
        self.event = event
        self.current_state = state if state is not None else automaton_def["states"]["initial"]
        self.final_states = set(automaton_def["states"].get("final", []))
        self.live = live
        if ctx is None:
            self.ctx = {}
            self._init_params()
        else:
            self.ctx = dict(ctx)

    def _init_params(self):
        for k, vdef in self.automaton_def.get("params", {}).items():
            self.ctx[k] = self._resolve_value(vdef)

    def _resolve_value(self, vdef: dict) -> Any:
        # En el verificador un parámetro probabilístico no se muestrea: se usa su esperanza.
        if not self.live and vdef.get("type") == "probabilistic":
            return self.automata.distributions.conditional_mean(
                vdef.get("distribution"), float("-inf"), float("inf"))
        return resolve_value(vdef, self.automata.distributions)

    def _resolve(self, vdef: Any) -> Any:
        if isinstance(vdef, dict) and "target" in vdef:
            return self._exec_logic(vdef)
        if isinstance(vdef, dict) and "type" in vdef:
            return self._resolve_value(vdef)
        return resolve_ephemeral(vdef, self.ctx)

    def _exec_logic(self, logic_def: dict) -> Any:
        target = logic_def["target"]
        method = logic_def["method"]
        param_keys = logic_def.get("params", [])

        if target == "system":
            try:
                return call_method(target, method, [param_keys],
                                 self.automata.agents, self.automata.environments,
                                 resolve_fn=self._resolve)
            except Exception as e:
                print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                      f"math_pipeline execution failed. Error: {type(e).__name__} - {e}", file=sys.stderr, flush=True)
                print(f"        pipeline_def: {param_keys}", file=sys.stderr, flush=True)
                os._exit(1)

        resolved_keys = [resolve_ephemeral(k, self.ctx) for k in param_keys]
        args = [self.ctx.get(k) if isinstance(k, str) and k.startswith("$") else k for k in resolved_keys]
        event_obj = self.event if target == "events" else None
        result = call_method(target, method, args, self.automata.agents, self.automata.environments, event_obj)

        if result is None:
            print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                  f"Target: {target} | Method: {method} | Params/Args: {args} -> Resolved Value: {result}",
                  flush=True)
            os._exit(1)

        return result

    # ------------------------------------------------------------------
    # Estructura de la transición
    # ------------------------------------------------------------------
    def is_final(self) -> bool:
        return self.current_state in self.final_states

    def transition(self) -> Optional[dict]:
        return next(
            (t for t in self.automaton_def["transitions"] if t["from"] == self.current_state), None)

    @staticmethod
    def threshold_value(transition: dict):
        tv = transition["threshold_value"]
        tv_key = next(iter(tv))
        return tv_key, tv[tv_key]

    def resolve_threshold(self, transition: dict) -> Any:
        """tau_dist: resuelve (muestrea, en ejecución viva) el valor que se compara con los umbrales."""
        _, tv_def = self.threshold_value(transition)
        return self._resolve(tv_def)

    def select_case(self, transition: dict, X: Any) -> Optional[dict]:
        """tau_dec: primer caso cuyas condiciones cumple X."""
        tv_key, _ = self.threshold_value(transition)
        previous = self.ctx.get(tv_key, _MISSING)
        self.ctx[tv_key] = X
        chosen = next((th for th in transition.get("thresholds", [])
                       if self._check_case(th["threshold_case"])), None)
        if previous is _MISSING:
            self.ctx.pop(tv_key, None)
        else:
            self.ctx[tv_key] = previous
        return chosen

    def apply_case(self, transition: dict, case: dict, X: Any) -> str:
        """Fija X en el contexto, mueve al estado destino del caso y ejecuta sus acciones."""
        tv_key, _ = self.threshold_value(transition)
        self.ctx[tv_key] = X
        self.current_state = case["to"]
        self._apply_actions(case.get("actions", []))
        return self.current_state

    def step(self) -> Optional[str]:
        if self.is_final():
            if self.live:
                self.automata.metrics_collector.record_automaton_execution(
                    self.automaton_name, "success", self.current_state)
            return None

        transition = self.transition()
        if not transition:
            if self.live:
                self.automata.metrics_collector.record_automaton_execution(
                    self.automaton_name, "failure", self.current_state)
            return None

        X = self.resolve_threshold(transition)
        chosen = self.select_case(transition, X)

        if not chosen:
            print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                  f"no threshold_case matched for value {X}", file=sys.stderr, flush=True)
            os._exit(1)

        return self.apply_case(transition, chosen, X)

    def _check_case(self, conditions: list) -> bool:
        for cond in conditions:
            raw_var = cond["variable"]
            op = cond["operator"]

            expected = resolve_ephemeral(cond["value"], self.ctx) if isinstance(cond["value"], str) else cond["value"]
            actual = None
            found = False

            if isinstance(raw_var, str):
                var_with_dollar = raw_var if raw_var.startswith("$") else f"${raw_var}"
                var_no_dollar = raw_var[1:] if raw_var.startswith("$") else raw_var

                if raw_var in self.ctx:
                    actual = self.ctx[raw_var]
                    found = True
                elif var_with_dollar in self.ctx:
                    actual = self.ctx[var_with_dollar]
                    found = True
                elif var_no_dollar in self.ctx:
                    actual = self.ctx[var_no_dollar]
                    found = True
            else:
                if raw_var in self.ctx:
                    actual = self.ctx[raw_var]
                    found = True

            if not found:
                print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                      f"undefined ephemeral variable '{raw_var}' in threshold_rule", file=sys.stderr, flush=True)
                os._exit(1)

            try:
                if op == "<" and not (actual < expected):
                    return False
                elif op == "<=" and not (actual <= expected):
                    return False
                elif op == ">" and not (actual > expected):
                    return False
                elif op == ">=" and not (actual >= expected):
                    return False
                elif op == "==" and not (actual == expected):
                    return False
                elif op == "!=" and not (actual != expected):
                    return False
                elif op not in ["<", "<=", ">", ">=", "==", "!="]:
                    return False
            except TypeError:
                return False

        return True

    def _apply_actions(self, actions: list):
        for action_obj in actions:
            if "update_params" in action_obj:
                for k, vdef in action_obj["update_params"].items():
                    self.ctx[k] = self._resolve(vdef)

            elif "action_required" in action_obj:
                for k, vdef in action_obj["action_required"].items():
                    result = self._resolve(vdef)
                    self.ctx[k] = result

            elif "event_emit" in action_obj:
                # En el verificador los eventos no se propagan: la interacción entre
                # agentes queda representada por las estrategias de la coalición y de los adversarios.
                if not self.live:
                    continue

                event_array = action_obj["event_emit"]
                resolved = [resolve_ephemeral(k, self.ctx) for k in event_array]

                signal = self.ctx.get(resolved[0]) if resolved[0].startswith("$") else resolved[0]
                if signal is None: signal = "UNKNOWN_SIGNAL"

                agent_id = self.ctx.get(resolved[1]) if resolved[1].startswith("$") else resolved[1]
                if agent_id is None: agent_id = "UNKNOWN_AGENT"

                event_data = {
                    "event_category": "dynamic",
                    "signal": str(signal),
                    "agent_id": str(agent_id)
                }

                if len(resolved) > 2:
                    env_id = self.ctx.get(resolved[2]) if resolved[2].startswith("$") else resolved[2]
                    if env_id is not None: event_data["env_id"] = str(env_id)
                if len(resolved) > 3:
                    channel_id = self.ctx.get(resolved[3]) if resolved[3].startswith("$") else resolved[3]
                    if channel_id is not None: event_data["channel_id"] = str(channel_id)

                try:
                    Events([event_data], self.automata.distributions)
                except Exception as e:
                    print(f"[FATAL] {self.automaton_name}::{self.current_state}: "
                          f"event_emit failed validation: {e}", file=sys.stderr, flush=True)
                    os._exit(1)

                if self.automata.event_sink is not None:
                    self.automata.event_sink(event_data)


_MISSING = object()
