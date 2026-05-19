import json
import time
import os
from copy import deepcopy
from jsonschema import validate
import asyncio
import pickle
from pathlib import Path


class SnapshotManager:
    def __init__(self, patl_config, metrics_collector, disk_dir=None):
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

        # Disk storage
        if disk_dir is None:
            disk_dir = Path(__file__).parent.parent / "data" / ".snapshots"
        self._disk_dir = Path(disk_dir)
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_index = 0


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
        Also protects against concurrent modification during iteration.
        """
        max_depth = 10
        
        if depth > max_depth:
            return None
        
        # Filter asyncio objects and anything from _asyncio module
        if isinstance(obj, (asyncio.Future, asyncio.Task, asyncio.Queue, 
                            asyncio.Event, asyncio.Lock, asyncio.Semaphore)):
            return None
        
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            module = getattr(obj.__class__, '__module__', '')
            if module.startswith('_asyncio') or module == 'asyncio':
                return None
            if class_name in ['_UnixSelectorEventLoop', 'ProactorEventLoop', 
                            'TaskStepMethWrapper', 'TaskWakeupMethWrapper']:
                return None
        
        try:
            return deepcopy(obj)
        except (TypeError, pickle.PicklingError, RuntimeError) as e:
            if isinstance(obj, dict):
                result = {}
                # Iterate over a snapshot of keys to avoid mutation during iteration
                try:
                    items = list(obj.items())
                except RuntimeError:
                    return None
                for k, v in items:
                    safe_key = self._safe_deepcopy(k, depth + 1)
                    safe_value = self._safe_deepcopy(v, depth + 1)
                    if safe_key is not None:
                        result[safe_key] = safe_value
                return result
            elif isinstance(obj, (list, tuple)):
                result = []
                # Snapshot to avoid mutation
                try:
                    items = list(obj)
                except RuntimeError:
                    return None
                for item in items:
                    safe_item = self._safe_deepcopy(item, depth + 1)
                    if safe_item is not None:
                        result.append(safe_item)
                return type(obj)(result) if isinstance(obj, tuple) else result
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj) if obj else None
            

    async def capture(self, agent_id, automaton_name, state):
        """
        Capture a snapshot and WRITE it to disk immediately.
        Does NOT keep the snapshot in memory.
        """
        if (automaton_name, state) not in self.data:
            return

        async with self._lock:
            self._sampling = True

            while self._active_transitions > 0:
                await asyncio.sleep(0.01)

            agents_data = self._safe_deepcopy(self.agents.data) if self.agents else None
            environments_data = self._safe_deepcopy(self.environments.data) if self.environments else None

            snapshot = {
                "timestamp": time.time(),
                "agent_id": agent_id,
                "automaton_name": automaton_name,
                "state": state,
                "agents_data": agents_data,
                "environments_data": environments_data
            }

            # Write to disk
            filepath = self._disk_dir / f"snap_{self._snapshot_index:010d}.json"
            with open(filepath, "w") as f:
                json.dump(snapshot, f, default=str)
            self._snapshot_index += 1

            if self.metrics_collector:
                self.metrics_collector.record_patl_sampling(automaton_name, state)

            self._sampling = False


    def read_and_delete(self, batch_size=50):
        """
        Generator that reads snapshots from disk in batches,
        yields them, and DELETES each file after reading.
        Nothing is kept in memory between batches.
        """
        files = sorted(
            [f for f in os.listdir(self._disk_dir) if f.startswith("snap_") and f.endswith(".json")]
        )
        
        for i in range(0, len(files), batch_size):
            batch = []
            batch_files = files[i:i + batch_size]
            
            for filename in batch_files:
                filepath = self._disk_dir / filename
                try:
                    with open(filepath, "r") as f:
                        batch.append(json.load(f))
                    os.remove(filepath)
                except (OSError, json.JSONDecodeError):
                    pass
            
            if batch:
                yield batch


    def get_snapshot_filepaths(self):
        """Return list of snapshot file paths sorted by creation order."""
        files = sorted(
            [f for f in os.listdir(self._disk_dir) if f.startswith("snap_") and f.endswith(".json")]
        )
        return [str(self._disk_dir / f) for f in files]
    

    def get_snapshot_count(self):
        """Return number of snapshots currently on disk."""
        files = [f for f in os.listdir(self._disk_dir) if f.startswith("snap_") and f.endswith(".json")]
        return len(files)


    def clear_all_snapshots(self):
        """Delete all remaining snapshot files from disk."""
        for filename in os.listdir(self._disk_dir):
            if filename.startswith("snap_") and filename.endswith(".json"):
                try:
                    os.remove(self._disk_dir / filename)
                except OSError:
                    pass
        try:
            self._disk_dir.rmdir()
        except OSError:
            pass