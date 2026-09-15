from typing import Any, Dict, Callable
import re
import math


_VAR_TOKEN = re.compile(r'\$[a-zA-Z_][a-zA-Z0-9_]*')


class EvaluationError(ValueError):
    """Error explícito de evaluación: variable sin resolver, operando no numérico o división entre cero."""


def resolve_value(vdef: Dict[str, Any], distributions: Any) -> Any:
    """Resolve a param_definition (deterministic or probabilistic)."""
    t = vdef.get("type")
    if t == "deterministic":
        return vdef.get("value")
    if t == "probabilistic":
        return distributions.sample(vdef.get("distribution"))
    return None


def unresolved_variables(value: Any, ctx: Dict[str, Any]) -> list:
    """
    Referencias "$x" de una cadena que resolve_ephemeral no puede resolver contra ctx.
    Una referencia completa "$x" se resuelve con "$x" o con "x"; dentro de un texto solo con "$x".
    """
    if not isinstance(value, str) or "$" not in value:
        return []
    if value.startswith("$") and " " not in value and (value in ctx or value[1:] in ctx):
        return []
    return [tok for tok in _VAR_TOKEN.findall(value) if tok not in ctx]


def call_method(target: str, method: str, args: list, agents_obj: Any, environments_obj: Any, event_obj: Any = None, resolve_fn: Callable = None) -> Any:
    """
    Llama un método sobre agents, environments, events o system.

    Un None devuelto por el método es el valor vacío y se devuelve tal cual. Una excepción dentro del
    método, un target desconocido o un método inexistente no se convierten en None: se propagan.
    """
    if target == "agents":
        target_obj = agents_obj
    elif target == "environments":
        target_obj = environments_obj
    elif target == "events":
        if isinstance(event_obj, dict):
            mapping = {
                "event_signal": "signal",
                "event_agent_id": "agent_id",
                "event_env_id": "env_id",
                "event_channel_id": "channel_id"
            }
            if method in mapping:
                return event_obj.get(mapping[method])
        raise EvaluationError(f"unknown events method '{method}'")
    elif target == "system":
        return _call_system(method, args, resolve_fn)
    else:
        raise EvaluationError(f"unknown target '{target}'")

    if target_obj is None:
        raise EvaluationError(f"target '{target}' is not bound")

    fn = getattr(target_obj, method, None)
    if fn is None or not callable(fn):
        raise EvaluationError(f"target '{target}' has no method '{method}'")
    return fn(*args)


def _call_system(method: str, args: list, resolve_fn: Callable = None) -> Any:
    """Execute system methods (math_pipeline)."""
    if method == "math_pipeline":
        pipeline_def = args[0] if args else {}
        return _execute_math_pipeline(pipeline_def, resolve_fn)
    raise EvaluationError(f"unknown system method '{method}'")


def _force_numeric(val, role):
    """Convierte un operando a número; rechaza None, cadenas no numéricas y referencias sin resolver."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        if "$" in val:
            raise EvaluationError(f"math_pipeline {role} references an unresolved variable: '{val}'")
        try:
            return float(val) if ("." in val or "e" in val.lower()) else int(val)
        except ValueError:
            raise EvaluationError(f"math_pipeline {role} is not numeric: '{val}'") from None
    raise EvaluationError(f"math_pipeline {role} is not numeric: {val!r}")


def _execute_math_pipeline(pipeline_def: dict, resolve_fn: Callable = None) -> Any:
    """Execute a math_pipeline definition, resolving inner variables dynamically."""
    if not pipeline_def:
        return 0

    # 1. Resolver el valor inicial usando la función de resolución del autómata si existe
    initial = pipeline_def.get("initial_value", 0)
    if resolve_fn:
        initial = resolve_fn(initial)
    if isinstance(initial, dict) and "type" in initial:
        initial = resolve_value(initial, None)

    value = _force_numeric(initial, "initial_value")

    # 2. Iterar y resolver dinámicamente cada operación
    for i, op_def in enumerate(pipeline_def.get("operations", [])):
        op = op_def["operator"]
        if op in ("ceil", "floor"):
            value = math.ceil(value) if op == "ceil" else math.floor(value)
            continue

        operand = op_def.get("with", 0)
        if resolve_fn:
            operand = resolve_fn(operand)
        if isinstance(operand, dict) and "type" in operand:
            operand = resolve_value(operand, None)

        operand = _force_numeric(operand, f"operand {i} ('{op}')")

        if op == "+":
            value = value + operand
        elif op == "-":
            value = value - operand
        elif op == "*":
            value = value * operand
        elif op == "/":
            if operand == 0:
                raise EvaluationError(f"math_pipeline division by zero at operation {i}")
            value = value / operand
        elif op == "max":
            value = max(value, operand)
        elif op == "min":
            value = min(value, operand)
        else:
            raise EvaluationError(f"math_pipeline unknown operator '{op}'")

    return value


def resolve_ephemeral(value: Any, ctx: Dict[str, Any]) -> Any:
    """Resolve $var references in a value using ctx."""
    if isinstance(value, dict) and "target" in value:
        return value

    if isinstance(value, str) and "$" in value:
        if value.startswith("$") and " " not in value:
            if value in ctx:
                return ctx[value]
            elif value[1:] in ctx:
                return ctx[value[1:]]

        def replacer(m):
            var_name = m.group(0)
            if var_name not in ctx:
                return var_name
            return str(ctx[var_name])

        res = _VAR_TOKEN.sub(replacer, value)

        try:
            return float(res) if "." in res else int(res)
        except ValueError:
            return res

    return value
