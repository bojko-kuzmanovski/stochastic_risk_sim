import json
import asyncio
from jsonschema import validate

# Importar funciones de evaluación estandarizadas
from sim_core.utils.evaluator import resolve_value

class Agents:
    def __init__(self, config_data, distributions, metrics_collector):
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
        self._tasks = {}
        self._running = False

        for agent_entry in config_data:
            for n in range(1, agent_entry["quantity"] + 1):
                agent_resolved = self._build_agent(agent_entry, n)
                self.data.append(agent_resolved)


    def _build_agent(self, agent_entry, n):
        """Construye un agente resuelto a partir de una entrada de configuración y un número n."""
        agent_type = agent_entry["agent_type"]
        agent_resolved = {
            "agent_id": f"{agent_type}_{n}",
            "agent_type": agent_type,
            "automata": agent_entry.get("automata", []),
        }

        # Resolve params
        resolved_params = {}
        ctx = resolved_params
        for pname, pdef in agent_entry.get("params", {}).items():
            resolved_params[pname] = resolve_value(
                vdef=pdef, ctx=ctx, distributions=self.distributions,
                agents_obj=None, environments_obj=None
            )
            ctx[pname] = resolved_params[pname]
        agent_resolved["params"] = resolved_params

        # Event queue
        agent_resolved["event_queue"] = asyncio.Queue()

        return agent_resolved


    def set_objects(self, automata, snapshot_manager):
        self.automata = automata
        self.snapshot_manager = snapshot_manager


    def _start_worker(self, agent):
        async def _worker():
            while True:
                event = await agent["event_queue"].get()

                signal = event.get("signal")
                if signal not in agent.get("automata", []):
                    continue

                session = self.automata.create_session(signal, event)
                if not session:
                    continue

                automaton_name = session.automaton_name

                while True:
                    # Esperar si hay muestreo en curso
                    while self.snapshot_manager.is_sampling():
                        await asyncio.sleep(0.01)

                    self.snapshot_manager.enter_transition()
                    new_state = session.step()
                    self.snapshot_manager.exit_transition()

                    if new_state is None:
                        break

                    await self.snapshot_manager.capture(agent["agent_id"], automaton_name, new_state)

        self._tasks[agent["agent_id"]] = asyncio.create_task(_worker())


    def _stop_worker(self, agent_id):
        """Cancela y limpia el worker de un agente específico."""
        task = self._tasks.pop(agent_id, None)
        if task:
            task.cancel()
            return True
        return False


    async def start(self):
        """Start async workers for each agent."""
        self._running = True
        for agent in self.data:
            self._start_worker(agent)


    async def stop(self):
        """Stop all agent loops."""
        self._running = False
        for agent_id in list(self._tasks.keys()):
            self._stop_worker(agent_id)


    async def receive_event(self, agent_id, event):
        agent = next((a for a in self.data if a["agent_id"] == agent_id), None)
        if agent is None:
            return
        await agent["event_queue"].put(event)
        if self.metrics_collector:
            event_category = event.get("event_category")
            signal = event.get("signal")
            self.metrics_collector.record_agent_event(event_category, signal)


    def get_all_agents(self, agent_type):
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

        # Start worker si ya está corriendo
        if self._running and self.automata:
            self._start_worker(agent_resolved)

        self.metrics_collector.record_agent_action(agent_type, "add_agent")
        return agent_resolved["agent_id"]


    def remove_agent(self, agent_id):
        """
        Elimina el agente con agent_id y detiene su worker.
        Retorna True si se eliminó, False si no existe.
        """
        for i, a in enumerate(self.data):
            if a["agent_id"] == agent_id:
                agent_type = a["agent_type"]
                del self.data[i]
                self._stop_worker(agent_id)
                self.metrics_collector.record_agent_action(agent_type, "remove_agent")
                return True
        return False
    

    def read_agent_param(self, agent_id, param_name):
        """
        Lee un parámetro específico del agente.
        Retorna el valor si existe, None si no existe el agente o el parámetro.
        """
        agent = next((a for a in self.data if a.get("agent_id") == agent_id), None)
        if agent is None:
            return None
        self.metrics_collector.record_agent_action(agent["agent_type"], "read_agent_param")
        return agent.get("params", {}).get(param_name)


    def write_agent_param(self, agent_id, param_name, value):
        """
        Escribe un parámetro del agente.
        Retorna True si se escribió, False si el agente no existe.
        """
        agent = next((a for a in self.data if a.get("agent_id") == agent_id), None)
        if agent is None:
            return False
        if "params" not in agent:
            agent["params"] = {}
        agent["params"][param_name] = value
        self.metrics_collector.record_agent_action(agent["agent_type"], "write_agent_param")
        return True
