import json
import time
import os
from jsonschema import validate
from pathlib import Path

from core.trace import tracer

# Campos de ejecución que no forman parte del estado global.
_RUNTIME_KEYS = {"event_queue"}


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
            bucket = self.data.setdefault(key, [])
            declared = {p.get("predicate_id") for p in bucket}
            for pred in obs.get("predicates", []):
                if pred.get("predicate_id") in declared:
                    continue
                bucket.append(pred)
                declared.add(pred.get("predicate_id"))

        self.agents = None
        self.environments = None

        if disk_dir is None:
            disk_dir = Path(__file__).parent.parent / "data" / ".snapshots"
        self._disk_dir = Path(disk_dir)
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        # Instantáneas de una corrida anterior en el mismo directorio no pertenecen a esta corrida.
        for filename in os.listdir(self._disk_dir):
            if filename.startswith("snap_") and filename.endswith(".json"):
                os.remove(self._disk_dir / filename)
        self._snapshot_index = 0
        # Reloj simulado (callable sin argumentos); el motor DES lo fija al iniciar la corrida.
        self._clock = None


    def set_objects(self, agents, environments):
        self.agents = agents
        self.environments = environments


    def set_clock(self, clock):
        self._clock = clock


    def capture(self, agent_id, automaton_name, state, trigger_event=None):
        """
        Congela el estado global (agentes, incluida la sesión activa, y entornos) cuando un
        agente alcanza un estado disparador, y lo escribe a disco sin conservarlo en memoria.
        Como cada sesión es atómica dentro del motor DES, ningún otro agente modifica el
        estado durante la captura.

        Campos de la instantánea: timestamp (reloj de pared), sim_time (tiempo simulado T, None fuera
        del motor), trigger_event (copia del evento cuya sesión está ejecutando el agente, o None),
        agent_id, automaton_name, state, agents_data y environments_data.
        """
        if (automaton_name, state) not in self.data:
            return

        agents_data = [
            {k: v for k, v in ag.items() if k not in _RUNTIME_KEYS}
            for ag in self.agents.data
        ] if self.agents else None
        environments_data = self.environments.data if self.environments else None

        snapshot = {
            "timestamp": time.time(),
            "sim_time": self._clock() if callable(self._clock) else None,
            "trigger_event": dict(trigger_event) if isinstance(trigger_event, dict) else None,
            "agent_id": agent_id,
            "automaton_name": automaton_name,
            "state": state,
            "agents_data": agents_data,
            "environments_data": environments_data
        }

        filepath = self._disk_dir / f"snap_{self._snapshot_index:010d}.json"
        with open(filepath, "w") as f:
            json.dump(snapshot, f, default=str)
        tracer.emit("snapshots", "capture", index=self._snapshot_index, agent=agent_id,
                    automaton=automaton_name, state=state,
                    predicates=[p.get("predicate_id") for p in self.data[(automaton_name, state)]])
        self._snapshot_index += 1

        if self.metrics_collector:
            self.metrics_collector.record_patl_sampling(automaton_name, state)


    def read_and_delete(self, batch_size=50):
        """
        Generator that reads snapshots from disk in batches,
        yields them, and DELETES each file after reading.
        Nothing is kept in memory between batches.
        """
        if not self._disk_dir.exists():
            return

        try:
            files = sorted(
                [f for f in os.listdir(self._disk_dir) if f.startswith("snap_") and f.endswith(".json")]
            )
        except FileNotFoundError:
            return

        for i in range(0, len(files), batch_size):
            batch = []
            batch_files = files[i:i + batch_size]

            for filename in batch_files:
                filepath = self._disk_dir / filename
                # Una instantánea ilegible es un error: omitirla cambiaría en silencio los resultados PATL.
                try:
                    with open(filepath, "r") as f:
                        batch.append(json.load(f))
                except (OSError, json.JSONDecodeError) as e:
                    raise RuntimeError(f"unreadable snapshot file {filepath}: {e}") from e
                os.remove(filepath)

            if batch:
                yield batch


    def get_snapshot_count(self):
        """Return number of snapshots currently on disk."""
        if not self._disk_dir.exists():
            return 0
        try:
            files = [f for f in os.listdir(self._disk_dir) if f.startswith("snap_") and f.endswith(".json")]
            return len(files)
        except FileNotFoundError:
            return 0


    def clear_all_snapshots(self):
        """Delete all remaining snapshot files and the run directory."""
        if not self._disk_dir.exists():
            return
        try:
            for filename in os.listdir(self._disk_dir):
                if filename.startswith("snap_") and filename.endswith(".json"):
                    try:
                        os.remove(self._disk_dir / filename)
                    except OSError:
                        pass
            self._disk_dir.rmdir()
        except (FileNotFoundError, OSError):
            return
