"""
Traza estructurada opcional para seguir una corrida paso a paso.

Cada registro es una línea JSON con el componente ("c"), el tipo de registro ("e"), la corrida ("run"),
el reloj simulado ("T", cuando existe) y los campos propios del registro. Componentes:

  des          instantes del calendario, eventos despachados y reprogramados, fin de la corrida
  agents       encolado en Q_i, inicio y fin de cada sesión atómica
  automata     cada transición (valor del umbral, caso elegido, estado destino), llamadas y eventos emitidos
  snapshots    cada instantánea capturada
  patl         cada predicado verificado: participantes, secuencias de estrategia, valor por secuencia, resultado
  patl_rounds  cada nodo del juego: elección del adversario, desenlaces de la ronda con su probabilidad, valor

Si ningún componente está activo, emit() retorna de inmediato.
"""

import json

COMPONENTS = ("des", "agents", "automata", "snapshots", "patl", "patl_rounds")


def parse_components(spec):
    if not spec:
        return frozenset()
    parts = {p.strip() for p in spec.split(",") if p.strip()}
    if "all" in parts:
        return frozenset(COMPONENTS)
    unknown = parts - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown trace components: {sorted(unknown)}; valid: {', '.join(COMPONENTS)} or all")
    return frozenset(parts)


class Tracer:
    def __init__(self):
        self._fh = None
        self._components = frozenset()
        self.run_id = None
        self.clock = None

    def configure(self, path, components, run_id):
        self.close()
        if not components:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w")
        self._components = frozenset(components)
        self.run_id = run_id
        self.clock = None

    def on(self, component):
        return component in self._components

    def emit(self, component, event, **fields):
        if component not in self._components:
            return
        record = {"run": self.run_id, "c": component, "e": event}
        if self.clock is not None:
            record["T"] = self.clock
        record.update(fields)
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def close(self):
        if self._fh:
            self._fh.close()
        self._fh = None
        self._components = frozenset()
        self.clock = None


tracer = Tracer()
