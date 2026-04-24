import json
import asyncio
from jsonschema import validate

# Importar funciones de evaluación estandarizadas
from core.utils.evaluator import resolve_value

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
        self.metrics_collector = metrics_collector
        self.automata = None
        self._tasks = {}

        for agent_entry in config_data:
            for n in range(1, agent_entry["quantity"] + 1):
                # Copy entry to avoid mutating original data
                agent_resolved = agent_entry.copy()

                # Assign agent_id
                agent_resolved["agent_id"] = f"{agent_entry['agent_type']}_{n}"

                # Resolve params usando resolve_value
                resolved_params = {}
                ctx = resolved_params
                
                for pname, pdef in agent_entry.get("params", {}).items():
                    resolved_params[pname] = resolve_value(vdef=pdef, ctx=ctx, distributions=distributions, agents_obj=None, environments_obj=None)
                    ctx[pname] = resolved_params[pname]
                
                agent_resolved["params"] = resolved_params

                # Asyncio event_queue
                agent_resolved["event_queue"] = asyncio.Queue()

                # Store in internal list
                self.data.append(agent_resolved)


    def set_objects(self, automata):
        self.automata = automata
    
    async def start(self):
        """Start async workers for each agent."""
        self._running = True
        for agent in self.data:
            self._tasks[agent["agent_id"]] = asyncio.create_task(
                self._agent_loop(agent)
            )

    async def stop(self):
        """Stop all agent loops."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    def get_by_type(self, agent_type: str):
        """Return all agents of a specific type."""
        return [a for a in self.data if a.get('agent_type') == agent_type]

    def get_all_agents(self):
        """Return all agents."""
        return self.data
    
    def set_param(self, agent_id: str, key: str, value) -> None:
        agent = next((a for a in self.data if a.get("agent_id") == agent_id), None)
        if agent:
            if "params" not in agent:
                agent["params"] = {}
            agent["params"][key] = value
            self.metrics_collector.record_agent_action(agent["agent_type"], "set_param")
    
    def get_param(self, agent_id: str, key: str, default=None):
        agent = next((a for a in self.data if a.get("agent_id") == agent_id), None)
        if not agent:
            return default

        self.metrics_collector.record_agent_action(agent["agent_type"], "get_param")
        return agent.get("params", {}).get(key, default)
    
    async def receive_event(self, agent_id, event):
        agent = next(a for a in self.data if a["agent_id"] == agent_id)
        await agent["event_queue"].put(event)
        if self.metrics_collector:
            event_category = event.get("event_category")
            signal = event.get("signal")
            self.metrics_collector.record_agent_event(event_category, signal)

    async def _agent_loop(self, agent):
        max_concurrency = 5
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_event(event):
            async with semaphore:
                try:
                    signal = event.get("signal")

                    if signal not in agent.get("automata", []):
                        return

                    await self.automata.process_event(event)

                except Exception as e:
                    pass

        while True:
            event = await agent["event_queue"].get()
            asyncio.create_task(_process_event(event))