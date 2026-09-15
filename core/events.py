import json
from jsonschema import validate

from core.utils.evaluator import resolve_value


def validate_periodicity_def(pdef, distributions, signal):
    """
    Rechaza al cargar una periodicidad que podría resolverse en un valor no positivo.

    Determinista: número estrictamente positivo. Probabilista: la distribución debe existir y su soporte
    efectivo debe quedar en (0, inf): truncamiento con min > 0 (min >= 1 si la salida es entera y la
    familia no es Poisson, porque int() trunca), o, en categóricas, etiquetas numéricas positivas.
    """
    where = f"periodicity of static signal '{signal}'"
    if pdef.get("type") == "deterministic":
        v = pdef.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{where} must be a number, got {type(v).__name__}: {v}")
        # Una periodicidad nula reprogramaría el evento en el mismo instante indefinidamente.
        if v <= 0:
            raise ValueError(f"{where} must be strictly positive, got {v}")
        return
    if pdef.get("type") != "probabilistic":
        raise ValueError(f"{where} has unknown type {pdef.get('type')!r}")

    name = pdef.get("distribution")
    entry = getattr(distributions, "samplers", {}).get(name)
    if entry is None:
        raise ValueError(f"{where} references unknown distribution '{name}'")
    family, out_t = entry["family"], entry["output_type"]
    if out_t not in ("int", "float"):
        raise ValueError(f"{where}: distribution '{name}' must have numeric output_type, got '{out_t}'")

    if family == "categorical":
        for label, p in zip(entry["labels"], entry["probabilities"]):
            if p <= 0:
                continue
            try:
                v = int(label) if out_t == "int" else float(label)
            except (TypeError, ValueError):
                raise ValueError(f"{where}: categorical label {label!r} of '{name}' is not numeric") from None
            if v <= 0:
                raise ValueError(f"{where}: categorical label {label!r} of '{name}' is not strictly positive")
        return

    trunc = entry.get("truncation")
    if not trunc or trunc["min"] <= 0:
        raise ValueError(f"{where}: distribution '{name}' must declare truncation with min > 0 "
                         f"(got {trunc}), otherwise a non-positive periodicity can be sampled")
    if out_t == "int" and family != "poisson" and trunc["min"] < 1:
        raise ValueError(f"{where}: distribution '{name}' has int output, so truncation min must be >= 1 "
                         f"(int() of a value in (0, 1) is 0), got {trunc['min']}")


def validate_static_signals(static_events, agents_config, automata_by_name=None):
    """Cada señal estática debe ser un autómata existente asignado al agent_type destino."""
    assigned = {a["agent_type"]: set(a.get("automata", [])) for a in agents_config}
    for ev in static_events:
        signal, agent_type = ev["signal"], ev["agent_type"]
        if automata_by_name is not None and signal not in automata_by_name:
            raise ValueError(f"static event signal '{signal}' is not a declared automaton")
        if agent_type not in assigned:
            raise ValueError(f"static event '{signal}' targets unknown agent_type '{agent_type}'")
        if signal not in assigned[agent_type]:
            raise ValueError(f"static event signal '{signal}' is not an automaton assigned to "
                             f"agent_type '{agent_type}' (assigned: {sorted(assigned[agent_type])})")


class Events:
    def __init__(self, config_data, distributions, agents_config=None, automata_by_name=None):
        # Load schema file
        schema_path = "schemas/events.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate data against schema
        validate(instance=config_data, schema=schema)

        # Process events
        self.data = []

        static = [e for e in config_data if e["event_category"] == "static"]
        if agents_config is not None and static:
            validate_static_signals(static, agents_config, automata_by_name)

        for event_entry in config_data:

            event_resolved = event_entry.copy()

            # STATIC EVENTS
            if event_entry["event_category"] == "static":
                pdef = event_entry["periodicity"]
                validate_periodicity_def(pdef, distributions, event_entry["signal"])
                periodicity = resolve_value(pdef, distributions)

                if isinstance(periodicity, bool) or not isinstance(periodicity, (int, float)):
                    raise TypeError(
                        f"periodicity must resolve to a number, got {type(periodicity).__name__}: "
                        f"{periodicity} for signal '{event_entry['signal']}'"
                    )
                if periodicity <= 0:
                    raise ValueError(
                        f"periodicity must be strictly positive, got {periodicity} "
                        f"for signal '{event_entry['signal']}'"
                    )

                event_resolved["periodicity"] = periodicity
                # La periodicidad probabilista se vuelve a muestrear en cada reprogramación.
                event_resolved["periodicity_def"] = pdef

            self.data.append(event_resolved)
