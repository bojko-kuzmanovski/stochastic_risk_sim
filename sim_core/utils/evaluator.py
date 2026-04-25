"""
Evaluator functions for deterministic, probabilistic, and logic value resolution.
Based on the critical logic originally inside Automata.process_event.
"""

from typing import Any, Dict, List, Optional, Union
import re


def eval_expr(expr: Any, ctx: Dict[str, Any]) -> Any:
    """
    Evaluate a Python expression string within a restricted context.
    If expr is not a string, return it unchanged.
    If evaluation fails, return the original string.
    """
    if isinstance(expr, str):
        try:
            # Restrict builtins for safety
            return eval(expr, {"__builtins__": {}}, ctx)
        except Exception:
            return expr
    return expr


def resolve_refs(refs: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a dictionary of references by evaluating each value with eval_expr.
    """
    return {rk: eval_expr(rv, ctx) for rk, rv in refs.items()}


def resolve_args(keys: List[str], ctx: Dict[str, Any]) -> List[Any]:
    """
    Extract values from context for a list of keys.
    """
    return [ctx.get(key) for key in keys]


def call_method(
    target_name: str,
    method_name: str,
    args: List[Any],
    agents_obj: Any,
    environments_obj: Any,
) -> Optional[Any]:
    """
    Dynamically call a method on either agents_obj or environments_obj.
    Returns None if any error occurs (method missing, exception, etc.).
    """
    target_obj = agents_obj if target_name == "agents" else environments_obj
    try:
        method = getattr(target_obj, method_name)
        return method(*args)
    except Exception:
        return None


def resolve_value(
    vdef: Dict[str, Any],
    ctx: Dict[str, Any],
    distributions: Any,
    agents_obj: Any,
    environments_obj: Any,
) -> Any:
    """
    Resolve a value definition (deterministic, probabilistic, or logic).
    
    vdef format:
        {"type": "deterministic", "value": ...}
        or {"type": "probabilistic", "distribution": str, "refs": {...}}
        or {"type": "logic", "query": {"target": "agents"/"environments",
                                       "method": str, "params": [str]}}
    """
    t = vdef.get("type")
    
    if t == "deterministic":
        val = vdef.get("value")
        if isinstance(val, str) and "${" in val:
            def replacer(m):
                var_name = m.group(1)
                return str(ctx.get(var_name, m.group(0)))
            interpolated = re.sub(r'\$\{(\w+)\}', replacer, val)
            try:
                return eval(interpolated, {"__builtins__": {}}, {**ctx, "ceil": __import__("math").ceil, "max": max})
            except Exception:
                return interpolated
        return eval_expr(val, ctx)
    
    elif t == "probabilistic":
        dist_name = vdef.get("distribution")
        refs_dict = vdef.get("refs", {})
        refs = resolve_refs(refs_dict, ctx) if refs_dict else {}
        if refs:
            return distributions.sample(dist_name, refs)
        else:
            return distributions.sample(dist_name)
    
    elif t == "logic":
        q = vdef.get("query", {})
        args_keys = q.get("params", [])
        args = resolve_args(args_keys, ctx)
        return call_method(
            q.get("target"),
            q.get("method"),
            args,
            agents_obj,
            environments_obj,
        )
    
    # Fallback for unknown type
    return None