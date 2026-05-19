from typing import Any, Dict
import re


def resolve_value(vdef: Dict[str, Any], distributions: Any) -> Any:
    """
    Resolve a param_definition (deterministic or probabilistic).

    vdef formats:
        {"type": "deterministic", "value": <string|number|boolean>}
        {"type": "probabilistic", "distribution": <string>}
    """
    t = vdef.get("type")
    
    if t == "deterministic":
        return vdef.get("value")
    
    if t == "probabilistic":
        return distributions.sample(vdef.get("distribution"))
    
    return None


def call_method(target: str, method: str, args: list, agents_obj: Any, environments_obj: Any, event_obj: Any = None) -> Any:
    """
    Call a method on agents, environments, events, or system.
    """
    if target == "agents":
        target_obj = agents_obj
    elif target == "environments":
        target_obj = environments_obj
    elif target == "events":
        target_obj = event_obj
    elif target == "system":
        return _call_system(method, args)
    else:
        return None

    if target_obj is None:
        return None

    try:
        fn = getattr(target_obj, method)
        return fn(*args)
    except Exception:
        return None


def _call_system(method: str, args: list) -> Any:
    """Execute system methods (math_pipeline)."""
    if method == "math_pipeline":
        pipeline_def = args[0] if args else {}
        return _execute_math_pipeline(pipeline_def)
    return None


def _execute_math_pipeline(pipeline_def: dict) -> Any:
    """Execute a math_pipeline definition."""
    import math
    
    initial = pipeline_def.get("initial_value", 0)
    if isinstance(initial, dict):
        initial = resolve_value(initial, None) if initial.get("type") else None
    if initial is None:
        initial = 0

    value = initial
    for op_def in pipeline_def.get("operations", []):
        op = op_def["operator"]
        operand = op_def.get("with", 0)
        
        if isinstance(operand, dict):
            operand = resolve_value(operand, None) if operand.get("type") else operand

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
    """
    Resolve $var references in a value using ctx.
    If value is a string containing $var patterns, interpolate from ctx.
    If value is a dict with 'target', treat as logic and execute it.
    Otherwise return value as-is.
    """
    if isinstance(value, dict) and "target" in value:
        return value

    if isinstance(value, str) and "$" in value:
        def replacer(m):
            var_name = m.group(0)
            if var_name not in ctx:
                return var_name
            return str(ctx[var_name])
        return re.sub(r'\$[a-zA-Z_][a-zA-Z0-9_]*', replacer, value)

    return value