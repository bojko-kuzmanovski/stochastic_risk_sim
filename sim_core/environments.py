import time
import random
import string
import json
from jsonschema import validate
from typing import Dict, List

# Importar funciones de evaluación estandarizadas
from sim_core.utils.evaluator import resolve_value

class Environments:
    def __init__(self, config_data, distributions, metrics_collector):
        # Load schema file
        schema_path = "schemas/environments.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Validate data against schema
        validate(instance=config_data, schema=schema)

        # Process environment data
        self.data = []
        self.config_data = config_data
        self.distributions = distributions
        self.metrics_collector = metrics_collector

        for env_entry in config_data:
            for n in range(1, env_entry["quantity"] + 1):
                env_resolved = self._build_env(env_entry, n)
                self.data.append(env_resolved)
    

    def _build_env(self, env_entry, n):
        """Construye un entorno resuelto a partir de una entrada de configuración y un número n."""
        env_type = env_entry["environment_type"]
        env_resolved = {
            "env_id": f"{env_type}_{n}",
            "environment_type": env_type,
        }

        # Resolve params
        resolved_params = {}
        ctx = resolved_params
        for pname, pdef in env_entry.get("params", {}).items():
            resolved_params[pname] = resolve_value(
                vdef=pdef, ctx=ctx, distributions=self.distributions,
                agents_obj=None, environments_obj=None
            )
            ctx[pname] = resolved_params[pname]
        env_resolved["params"] = resolved_params

        # Members
        resolved_members = []
        for member in env_entry.get("members", []):
            resolved_rol = {}
            rol_ctx = {}
            for rname, rdef in member.get("agent_rol", {}).items():
                resolved_rol[rname] = resolve_value(
                    vdef=rdef, ctx={**resolved_params, **rol_ctx},
                    distributions=self.distributions,
                    agents_obj=None, environments_obj=None
                )
                rol_ctx[rname] = resolved_rol[rname]
            resolved_members.append({
                "agent_id": member["agent_id"],
                "agent_rol": resolved_rol
            })
        env_resolved["members"] = resolved_members

        # Relations
        resolved_relations = []
        for rel in env_entry.get("relations", []):
            resolved_meta = {}
            meta_ctx = {}
            for mname, mdef in rel.get("rel_metadata", {}).items():
                resolved_meta[mname] = resolve_value(
                    vdef=mdef, ctx={**resolved_params, **meta_ctx},
                    distributions=self.distributions,
                    agents_obj=None, environments_obj=None
                )
                meta_ctx[mname] = resolved_meta[mname]
            resolved_relations.append({
                "agent_a_id": rel["agent_a_id"],
                "agent_b_id": rel["agent_b_id"],
                "rel_metadata": resolved_meta
            })
        env_resolved["relations"] = resolved_relations

        # Channels
        resolved_channels = []
        for ch in env_entry.get("channels", []):
            # Resolver channel_metadata
            resolved_ch_meta = {}
            ch_meta_ctx = {}
            for mname, mdef in ch.get("channel_metadata", {}).items():
                resolved_ch_meta[mname] = resolve_value(
                    vdef=mdef, ctx={**resolved_params, **ch_meta_ctx},
                    distributions=self.distributions,
                    agents_obj=None, environments_obj=None
                )
                ch_meta_ctx[mname] = resolved_ch_meta[mname]

            # Eventos (ya resueltos, sin metadata)
            resolved_events = []
            for ev in ch.get("events", []):
                resolved_events.append({
                    "agent_id": ev["agent_id"],
                    "signal": ev["signal"]
                })

            resolved_channels.append({
                "channel_id": ch["channel_id"],
                "members": ch["members"],
                "events": resolved_events,
                "channel_metadata": resolved_ch_meta
            })
        env_resolved["channels"] = resolved_channels

        return env_resolved


    def get_all_envs(self, env_type):
        self.metrics_collector.record_environment_action(env_type, "get_all_envs")
        return [e["env_id"] for e in self.data if e.get("environment_type") == env_type]


    def add_env(self, env_type):
        """Crea un nuevo entorno del tipo dado usando la definición en config_data.
        Retorna el env_id si se creó, None si no existe el tipo en config_data."""
        env_entry = next((e for e in self.config_data if e["environment_type"] == env_type), None)
        if env_entry is None:
            return None

        # Next env_id
        max_n = 0
        for e in self.data:
            if e["environment_type"] == env_type:
                suffix = e["env_id"].split("_")[-1]
                try:
                    n = int(suffix)
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        n = max_n + 1

        env_resolved = self._build_env(env_entry, n)
        self.data.append(env_resolved)
        self.metrics_collector.record_environment_action(env_type, "add_env")
        return env_resolved["env_id"]


    def remove_env(self, env_id):
        """Elimina el entorno con env_id y todo su contenido."""
        for i, e in enumerate(self.data):
            if e["env_id"] == env_id:
                env_type = e["environment_type"]
                del self.data[i]
                self.metrics_collector.record_environment_action(env_type, "remove_env")
                return True
        return False


    def read_env_param(self, env_id, param_name):
        """ Lee un parámetro específico del entorno."""
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_env_param")
        return env.get("params", {}).get(param_name)


    def write_env_param(self, env_id, param_name, value):
        """Escribe un parámetro del entorno."""
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        if "params" not in env:
            env["params"] = {}
        env["params"][param_name] = value
        self.metrics_collector.record_environment_action(env["environment_type"], "write_env_param")
        return True
    

    def read_members(self, env_id):
        """Retorna un arreglo de agent_id (cadenas) que son miembros del entorno."""
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_members")
        return [m["agent_id"] for m in env.get("members", [])]


    def write_member(self, env_id, agent_id):
        """
        Agrega un agente como miembro del entorno con rol vacío.
        Retorna True si se agregó, False si el entorno no existe o ya es miembro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        if any(m["agent_id"] == agent_id for m in env.get("members", [])):
            return False
        env["members"].append({
            "agent_id": agent_id,
            "agent_rol": {}
        })
        self.metrics_collector.record_environment_action(env["environment_type"], "write_member")
        return True


    def read_member_param(self, env_id, agent_id, param_name):
        """
        Lee un parámetro de rol de un miembro específico.
        Retorna el valor del parámetro, o None si no existe el entorno, miembro o parámetro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        member = next((m for m in env.get("members", []) if m["agent_id"] == agent_id), None)
        if member is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_member_param")
        return member.get("agent_rol", {}).get(param_name)


    def write_member_param(self, env_id, agent_id, param_name, value):
        """
        Escribe un parámetro de rol de un miembro específico.
        Retorna True si se escribió, False si no existe el entorno o el miembro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        member = next((m for m in env.get("members", []) if m["agent_id"] == agent_id), None)
        if member is None:
            return False
        if "agent_rol" not in member:
            member["agent_rol"] = {}
        member["agent_rol"][param_name] = value
        self.metrics_collector.record_environment_action(env["environment_type"], "write_member_param")
        return True


    def read_rel(self, env_id, agent_a_id, agent_b_id):
        """
        Verifica si existe una relación entre agent_a_id y agent_b_id.
        Retorna True si existe, False si no existe o el entorno no existe.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        self.metrics_collector.record_environment_action(env["environment_type"], "read_rel")
        for r in env.get("relations", []):
            if r["agent_a_id"] == agent_a_id and r["agent_b_id"] == agent_b_id:
                return True
        return False


    def add_rel(self, env_id, agent_a_id, agent_b_id):
        """
        Crea una nueva relación con metadata vacía.
        Retorna True si se creó, False si el entorno no existe o ya existe la relación.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        for r in env.get("relations", []):
            if r["agent_a_id"] == agent_a_id and r["agent_b_id"] == agent_b_id:
                return False
        env["relations"].append({
            "agent_a_id": agent_a_id,
            "agent_b_id": agent_b_id,
            "rel_metadata": {}
        })
        self.metrics_collector.record_environment_action(env["environment_type"], "add_rel")
        return True


    def remove_rel(self, env_id, agent_a_id, agent_b_id):
        """
        Elimina una relación entre agent_a_id y agent_b_id.
        Retorna True si se eliminó, False si el entorno o la relación no existen.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        for i, r in enumerate(env.get("relations", [])):
            if r["agent_a_id"] == agent_a_id and r["agent_b_id"] == agent_b_id:
                del env["relations"][i]
                self.metrics_collector.record_environment_action(env["environment_type"], "remove_rel")
                return True
        return False


    def read_rel_param(self, env_id, agent_a_id, agent_b_id, param_name):
        """
        Lee un parámetro de metadata de una relación.
        Retorna el valor si existe, None si no existe el entorno, relación o parámetro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        rel = next(
            (r for r in env.get("relations", [])
            if r["agent_a_id"] == agent_a_id and r["agent_b_id"] == agent_b_id),
            None
        )
        if rel is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_rel_param")
        return rel.get("rel_metadata", {}).get(param_name)


    def write_rel_param(self, env_id, agent_a_id, agent_b_id, param_name, value):
        """
        Escribe un parámetro de metadata en una relación existente.
        Retorna True si se escribió, False si no existe el entorno o la relación.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        rel = next(
            (r for r in env.get("relations", [])
            if r["agent_a_id"] == agent_a_id and r["agent_b_id"] == agent_b_id),
            None
        )
        if rel is None:
            return False
        if "rel_metadata" not in rel:
            rel["rel_metadata"] = {}
        rel["rel_metadata"][param_name] = value
        self.metrics_collector.record_environment_action(env["environment_type"], "write_rel_param")
        return True


    def read_ch(self, env_id):
        """
        Retorna un arreglo de channel_id del entorno.
        Retorna None si el entorno no existe.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_ch")
        return [ch["channel_id"] for ch in env.get("channels", [])]


    def add_ch(self, env_id):
        """
        Crea un nuevo canal vacío (sin miembros ni eventos) con un channel_id único generado automáticamente.
        Retorna el channel_id si se creó, None si el entorno no existe.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None

        existing_ids = {ch["channel_id"] for ch in env.get("channels", [])}
        while True:
            ch_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            if ch_id not in existing_ids:
                break

        env.setdefault("channels", []).append({
            "channel_id": ch_id,
            "members": [],
            "events": [],
            "channel_metadata": {}
        })
        self.metrics_collector.record_environment_action(env["environment_type"], "add_ch")
        return ch_id


    def remove_ch(self, env_id, ch_id):
        """
        Elimina un canal por su channel_id.
        Retorna True si se eliminó, False si el entorno o el canal no existen.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        for i, ch in enumerate(env.get("channels", [])):
            if ch["channel_id"] == ch_id:
                del env["channels"][i]
                self.metrics_collector.record_environment_action(env["environment_type"], "remove_ch")
                return True
        return False


    def read_ch_members(self, env_id, ch_id):
        """
        Retorna un arreglo de agent_id miembros del canal.
        Retorna None si el entorno o el canal no existen.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_ch_members")
        return ch.get("members", [])


    def write_ch_member(self, env_id, ch_id, agent_id):
        """
        Agrega un agente como miembro del canal.
        Retorna True si se agregó, False si el entorno/canal no existe o ya es miembro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return False
        if agent_id in ch.get("members", []):
            return False
        ch["members"].append(agent_id)
        self.metrics_collector.record_environment_action(env["environment_type"], "write_ch_member")
        return True


    def read_ch_param(self, env_id, ch_id, param_name):
        """
        Lee un parámetro de metadata del canal.
        Retorna el valor si existe, None si no existe el entorno, canal o parámetro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_ch_param")
        return ch.get("channel_metadata", {}).get(param_name)


    def write_ch_param(self, env_id, ch_id, param_name, value):
        """
        Escribe un parámetro de metadata en el canal.
        Retorna True si se escribió, False si no existe el entorno o el canal.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return False
        if "channel_metadata" not in ch:
            ch["channel_metadata"] = {}
        ch["channel_metadata"][param_name] = value
        self.metrics_collector.record_environment_action(env["environment_type"], "write_ch_param")
        return True


    def read_ch_events(self, env_id, ch_id):
        """
        Retorna un arreglo de eventos del canal, ordenados del más reciente al más antiguo.
        Retorna None si el entorno o el canal no existen.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return None
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return None
        self.metrics_collector.record_environment_action(env["environment_type"], "read_ch_events")
        events = ch.get("events", [])
        return list(reversed(events))


    def write_ch_event(self, env_id, ch_id, agent_id, signal):
        """
        Agrega un evento al canal.
        Retorna True si se agregó, False si el entorno/canal no existe o el agente no es miembro.
        """
        env = next((e for e in self.data if e["env_id"] == env_id), None)
        if env is None:
            return False
        ch = next((c for c in env.get("channels", []) if c["channel_id"] == ch_id), None)
        if ch is None:
            return False
        if agent_id not in ch.get("members", []):
            return False

        ch["events"].append({
            "agent_id": agent_id,
            "signal": signal
        })
        self.metrics_collector.record_environment_action(env["environment_type"], "write_ch_event")
        return True