import json
import time
from copy import deepcopy
from jsonschema import validate
import asyncio


class SnapshotManager:
    def __init__(self, patl_config, metrics_collector):
        schema_path = "schemas/patl.schema.json"
        with open(schema_path, "r") as f:
            schema = json.load(f)
        validate(instance=patl_config, schema=schema)

        self.metrics_collector = metrics_collector

        self.data = {}
        for obs in patl_config.get("observations", []):
            key = (obs["automaton_name"], obs["trigger_state"])
            self.data[key] = obs.get("predicates", [])

        self._active_transitions = 0
        self._sampling = False
        self._lock = asyncio.Lock()

        self.agents = None
        self.environments = None

        self._snapshots = []


    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments


    def is_sampling(self):
        return self._sampling


    def enter_transition(self):
        self._active_transitions += 1


    def exit_transition(self):
        self._active_transitions -= 1


    async def capture(self, agent_id, automaton_name, state):
        if (automaton_name, state) not in self.data:
            return

        async with self._lock:
            self._sampling = True

            while self._active_transitions > 0:
                await asyncio.sleep(0.01)

            agents_data = deepcopy(self.agents.data)
            environments_data = deepcopy(self.environments.data) if self.environments else None

            self._snapshots.append({
                "timestamp": time.time(),
                "agent_id": agent_id,
                "automaton_name": automaton_name,
                "state": state,
                "agents_data": agents_data,
                "environments_data": environments_data
            })

            if self.metrics_collector:
                self.metrics_collector.record_patl_sampling(automaton_name, state)

            self._sampling = False