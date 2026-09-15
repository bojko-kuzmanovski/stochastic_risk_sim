import os
import json
from collections import deque
from jsonschema import validate
from copy import deepcopy

# Importar funciones de evaluación estandarizadas
from core.utils.evaluator import resolve_value

# Cota de pasos por sesión: una sesión que no llega a estado final en este número
# de transiciones indica un ciclo en la configuración del autómata.
MAX_SESSION_STEPS = 10_000


class Agents:
    def __init__(self, config_data, distributions, metrics_collector, worker_mode=True):
        # Load schema file
        schema_path = "schemas/agents.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate data against schema
        validate(instance=config_data, schema=schema)

        # Process agents attributes
        self.data = []
        self.config_data = config_data
        self.distributions = distributions
        self.metrics_collector = metrics_collector
        self.automata = None
        self.snapshot_manager = None
        self.worker_mode = worker_mode
        self._running = False
        # Se invoca con cada agente creado en ejecución para programar sus eventos estáticos.
        self._on_agent_added = None

        for agent_entry in config_data:
            for n in range(1, agent_entry["quantity"] + 1):
                agent_resolved = self._build_agent(agent_entry, n)
                self.data.append(agent_resolved)


    def _build_agent(self, agent_entry, n):
        agent_type = agent_entry["agent_type"]
        agent_resolved = {
            "agent_id": f"{agent_type}_{n}",
            "agent_type": agent_type,
            "automata": agent_entry.get("automata", []),
        }

        resolved_params = {}
        for pname, pdef in agent_entry.get("params", {}).items():
            resolved_params[pname] = resolve_value(vdef=pdef, distributions=self.distributions)
        agent_resolved["params"] = resolved_params

        if self.worker_mode:
            # Cola Q_i de eventos pendientes del agente.
            agent_resolved["event_queue"] = deque()

        return agent_resolved


    def set_objects(self, automata, snapshot_manager):
        self.automata = automata
        self.snapshot_manager = snapshot_manager


    def set_on_agent_added(self, callback):
        self._on_agent_added = callback


    def _find(self, agent_id):
        return next((a for a in self.data if a.get("agent_id") == agent_id), None)


    def start(self):
        self._running = True


    def stop(self):
        self._running = False


    def receive_event(self, agent_id, event):
        """Paso 1 (despacho): encola el evento en Q_i. Devuelve False si el agente ya no existe."""
        agent = self._find(agent_id)
        if agent is None or "event_queue" not in agent:
            return False
        agent["event_queue"].append(event)
        if self.metrics_collector:
            self.metrics_collector.record_agent_event(event.get("event_category"), event.get("signal"))
        return True


    def pending(self, agent_id):
        agent = self._find(agent_id)
        if agent is None or "event_queue" not in agent:
            return 0
        return len(agent["event_queue"])


    def process_queue(self, agent_id, max_events):
        """Paso 2 (procesamiento): extrae hasta K eventos de Q_i y ejecuta cada sesión de forma atómica."""
        agent = self._find(agent_id)
        if agent is None:
            return 0
        processed = 0
        while agent["event_queue"] and processed < max_events and self._find(agent_id) is not None:
            event = agent["event_queue"].popleft()
            self._run_event(agent, event)
            processed += 1
        return processed


    def _run_event(self, agent, event):
        signal = event.get("signal")
        automaton_def = self.automata.by_name.get(signal)
        if not automaton_def:
            return

        if signal not in agent.get("automata", []):
            print(f"[FATAL] Agent {agent['agent_id']} rejected signal '{signal}' because it does not have that automaton assigned.")
            os._exit(1)

        session = self.automata.create_session(signal, event)
        if not session:
            return

        # El estado de la sesión activa forma parte del estado global que capturan las instantáneas.
        agent["current_automaton"] = session.automaton_name
        agent["current_state"] = session.current_state
        agent["session_ctx"] = session.ctx

        for _ in range(MAX_SESSION_STEPS):
            new_state = session.step()
            if new_state is None:
                break
            agent["current_state"] = new_state
            self.snapshot_manager.capture(agent["agent_id"], session.automaton_name, new_state)
        else:
            print(f"[FATAL] Agent {agent['agent_id']}: automaton '{signal}' did not reach a final state "
                  f"after {MAX_SESSION_STEPS} steps.")
            os._exit(1)

        agent.pop("current_automaton", None)
        agent.pop("current_state", None)
        agent.pop("session_ctx", None)


    # DES / PATL METHODS
    def get_all_agents(self, agent_type):
        if self.metrics_collector:
            self.metrics_collector.record_agent_action(agent_type, "get_all_agents")
        return [a["agent_id"] for a in self.data if a.get("agent_type") == agent_type]


    def add_agent(self, agent_type):
        """
        Crea un nuevo agente del tipo dado usando la definición en config_data.
        Retorna el agent_id si se creó, None si no existe el tipo en config_data.
        """
        agent_entry = next((a for a in self.config_data if a["agent_type"] == agent_type), None)
        if agent_entry is None:
            return None

        # Next agent_id
        max_n = 0
        for a in self.data:
            if a["agent_type"] == agent_type:
                suffix = a["agent_id"].split("_")[-1]
                try:
                    n = int(suffix)
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        n = max_n + 1

        agent_resolved = self._build_agent(agent_entry, n)
        self.data.append(agent_resolved)

        # Sus eventos estáticos se programan desde el instante actual del calendario.
        if self._running and self._on_agent_added:
            self._on_agent_added(agent_resolved)

        if self.metrics_collector:
            self.metrics_collector.record_agent_action(agent_type, "add_agent")
        return agent_resolved["agent_id"]


    def remove_agent(self, agent_id):
        """
        Elimina el agente con agent_id; sus eventos pendientes se descartan.
        Retorna True si se eliminó, False si no existe.
        """
        for i, a in enumerate(self.data):
            if a["agent_id"] == agent_id:
                agent_type = a["agent_type"]
                del self.data[i]
                if self.metrics_collector:
                    self.metrics_collector.record_agent_action(agent_type, "remove_agent")
                return True
        return False


    def read_agent_param(self, agent_id, param_name):
        """
        Lee un parámetro específico del agente.
        Retorna el valor si existe, None si no existe el agente o el parámetro.
        """
        agent = self._find(agent_id)
        if agent is None:
            return None
        if self.metrics_collector:
            self.metrics_collector.record_agent_action(agent["agent_type"], "read_agent_param")
        return agent.get("params", {}).get(param_name)


    def write_agent_param(self, agent_id, param_name, value):
        """
        Escribe un parámetro del agente.
        Retorna True si se escribió, False si el agente no existe.
        """
        agent = self._find(agent_id)
        if agent is None:
            return False
        if "params" not in agent:
            agent["params"] = {}
        agent["params"][param_name] = value
        if self.metrics_collector:
            self.metrics_collector.record_agent_action(agent["agent_type"], "write_agent_param")
        return True
