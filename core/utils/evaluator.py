from typing import Any, Dict, Callable
import re
import math


def resolve_value(vdef: Dict[str, Any], distributions: Any) -> Any:
    """Resolve a param_definition (deterministic or probabilistic)."""
    t = vdef.get("type")
    if t == "deterministic":
        return vdef.get("value")
    if t == "probabilistic":
        return distributions.sample(vdef.get("distribution"))
    return None


def call_method(target: str, method: str, args: list, agents_obj: Any, environments_obj: Any, event_obj: Any = None, resolve_fn: Callable = None) -> Any:
    """Call a method on agents, environments, events, or system."""
    if target == "agents":
        target_obj = agents_obj
    elif target == "environments":
        target_obj = environments_obj
    elif target == "events":
        target_obj = event_obj
    elif target == "system":
        return _call_system(method, args, resolve_fn)
    else:
        return None

    if target_obj is None:
        return None

    try:
        fn = getattr(target_obj, method)
        return fn(*args)
    except Exception:
        return None


def _call_system(method: str, args: list, resolve_fn: Callable = None) -> Any:
    """Execute system methods (math_pipeline)."""
    if method == "math_pipeline":
        pipeline_def = args[0] if args else {}
        return _execute_math_pipeline(pipeline_def, resolve_fn)
    return None


def _execute_math_pipeline(pipeline_def: dict, resolve_fn: Callable = None) -> Any:
    """Execute a math_pipeline definition, resolving inner variables dynamically."""
    
    def _force_numeric(val):
        if isinstance(val, str):
            try:
                return float(val) if "." in val else int(val)
            except ValueError:
                return 0
        return val if val is not None else 0

    if not pipeline_def:
        return 0

    # 1. Resolver el valor inicial usando la función de resolución del autómata si existe
    initial = pipeline_def.get("initial_value", 0)
    if resolve_fn:
        initial = resolve_fn(initial)
    if isinstance(initial, dict) and "type" in initial:
        initial = resolve_value(initial, None)

    value = _force_numeric(initial)
    
    # 2. Iterar y resolver dinámicamente cada operación
    for op_def in pipeline_def.get("operations", []):
        op = op_def["operator"]
        operand = op_def.get("with", 0)
        
        if resolve_fn:
            operand = resolve_fn(operand)
        if isinstance(operand, dict) and "type" in operand:
            operand = resolve_value(operand, None)

        operand = _force_numeric(operand)

        if op == "+":
            value = value + operand
        elif op == "-":
            value = value - operand
        elif op == "*":
            value = value * operand
        elif op == "/":
            value = value / operand if operand != 0 else 0
        elif op == "ceil":
            value = math.ceil(value)
        elif op == "floor":
            value = math.floor(value)
        elif op == "max":
            value = max(value, operand)
        elif op == "min":
            value = min(value, operand)

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
        
        res = re.sub(r'\$[a-zA-Z_][a-zA-Z0-9_]*', replacer, value)
        
        try:
            return float(res) if "." in res else int(res)
        except ValueError:
            return res

    return value