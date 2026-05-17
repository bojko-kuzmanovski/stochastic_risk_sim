import json
import time
from copy import deepcopy
from jsonschema import validate
import asyncio
import pickle


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


    def _safe_deepcopy(self, obj, depth=0):
        """
        Safe deepcopy that handles asyncio futures and other non-picklable objects.
        """
        max_depth = 10
        
        if depth > max_depth:
            return None
        
        # Skip asyncio objects that can't be pickled
        if isinstance(obj, (asyncio.Future, asyncio.Task, asyncio.Queue, asyncio.Event, asyncio.Lock)):
            return None
        
        # Skip common non-picklable types
        if hasattr(obj, '__class__') and obj.__class__.__name__ in ['_UnixSelectorEventLoop', 'ProactorEventLoop']:
            return None
        
        try:
            # Try normal deepcopy first
            return deepcopy(obj)
        except (TypeError, pickle.PicklingError) as e:
            # If deepcopy fails, try to recursively copy dictionaries and lists
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    safe_key = self._safe_deepcopy(k, depth + 1)
                    safe_value = self._safe_deepcopy(v, depth + 1)
                    if safe_key is not None:
                        result[safe_key] = safe_value
                return result
            elif isinstance(obj, (list, tuple)):
                result = []
                for item in obj:
                    safe_item = self._safe_deepcopy(item, depth + 1)
                    if safe_item is not None:
                        result.append(safe_item)
                return type(obj)(result) if isinstance(obj, tuple) else result
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                # For other types, return None or a string representation
                return str(obj) if obj else None


    async def capture(self, agent_id, automaton_name, state):
        if (automaton_name, state) not in self.data:
            return

        async with self._lock:
            self._sampling = True

            while self._active_transitions > 0:
                await asyncio.sleep(0.01)

            # Use safe deepcopy
            agents_data = self._safe_deepcopy(self.agents.data) if self.agents else None
            environments_data = self._safe_deepcopy(self.environments.data) if self.environments else None

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


    def get_all_snapshots(self):
        """Return all captured snapshots"""
        return self._snapshots


    def clear_all_snapshots(self):
        """Clear all captured snapshots"""
        self._snapshots = []