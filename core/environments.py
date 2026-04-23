import time
import random
import string
import json
from jsonschema import validate
from typing import Dict, List

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
        self.metrics_collector = metrics_collector
        
        for env_entry in config_data:
            for n in range(1, env_entry["quantity"] + 1):
                # Copy entry to avoid mutating original data
                env_resolved = env_entry.copy()
                
                # Assign env_id
                env_resolved["env_id"] = f"{env_entry['environment_type']}_{n}"
                
                # Resolve params
                resolved_params = {}
                
                def eval_expr(expr, ctx):
                    try:
                        return eval(expr, {"__builtins__": {}}, ctx)
                    except:
                        return expr
                
                for pname, pdef in env_entry.get("params", {}).items():
                    if pdef["type"] == "deterministic":
                        val = pdef["value"]
                        resolved_params[pname] = eval_expr(val, resolved_params) if isinstance(val, str) else val
                    
                    elif pdef["type"] == "probabilistic":
                        dist = pdef["distribution"]
                        refs = {k: eval_expr(v, resolved_params) if isinstance(v, str) else v 
                                for k, v in pdef.get("refs", {}).items()}
                        resolved_params[pname] = distributions.sample(dist, refs) if refs else distributions.sample(dist)
                
                env_resolved["params"] = resolved_params
                
                # Store in internal list
                self.data.append(env_resolved)

    
    def get_param(self, env_id, agent_id, param_name):
        """Método query para obtener datos del entorno"""
        env = next((e for e in self.data if e.get("env_id") == env_id), None)
        if not env:
            return None
        
        if param_name == "profiles_not_followed":
            profiles = [p["agent_id"] for p in env.get("profile", [])]
            following = [f["to_agent_id"] for f in env.get("follow", []) 
                        if f["from_agent_id"] == agent_id and f["is_approved"]]
            return [p for p in profiles if p != agent_id and p not in following]
        
        elif param_name == "pending_follow_requests":
            return [f for f in env.get("follow", []) 
                    if f["to_agent_id"] == agent_id and not f["is_approved"]]
        
        elif param_name == "following_list":
            return [f["to_agent_id"] for f in env.get("follow", []) 
                    if f["from_agent_id"] == agent_id and f["is_approved"]]
        
        elif param_name == "available_contacts":
            # Contactos mutuos o perfiles públicos
            following = [f["to_agent_id"] for f in env.get("follow", []) 
                        if f["from_agent_id"] == agent_id and f["is_approved"]]
            followers = [f["from_agent_id"] for f in env.get("follow", []) 
                        if f["to_agent_id"] == agent_id and f["is_approved"]]
            return list(set(following) & set(followers))
        
        return None
    
    def get_env_ids_by_type(self, env_type: str) -> List[str]:
        self.metrics_collector.record_environment_action(env_type, "get_env_ids_by_type")
        return [
            env["env_id"]
            for env in self.data
            if env.get("environment_type") == env_type
        ]
    
    def profile_already_registered(self, env_id, agent_id):
        env = next((e for e in self.data if e.get("env_id") == env_id), None)
        if not env or "profile" not in env:
            return False

        self.metrics_collector.record_environment_action(env["environment_type"], "profile_already_registered")
        return any(p["agent_id"] == agent_id for p in env["profile"])

    def profile_register(self, env_id, agent_id, is_private):
        for env in self.data:
            if env.get("env_id") == env_id and "profile" in env:
                for p in env["profile"]:
                    if p["agent_id"] == agent_id:
                        return
                env["profile"].append({
                    "agent_id": agent_id,
                    "is_private": is_private
                })

                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                self.metrics_collector.record_environment_action(env["environment_type"], "profile_register")

                return


    def profile_private_change(self, env_id, agent_id, is_private):
        for env in self.data:
            if env.get("env_id") == env_id and "profile" in env:
                for p in env["profile"]:
                    if p["agent_id"] == agent_id:
                        p["is_private"] = is_private

                        env = next((e for e in self.data if e.get("env_id") == env_id), None)
                        self.metrics_collector.record_environment_action(env["environment_type"], "profile_private_change")

                        return
                    

    def follow_request(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "follow" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for f in env["follow"]:
                        if f["from_agent_id"] == from_agent_id and f["to_agent_id"] == to_agent_id:
                            return
                    env["follow"].append({
                        "from_agent_id": from_agent_id,
                        "to_agent_id": to_agent_id,
                        "is_approved": False
                    })

                    env = next((e for e in self.data if e.get("env_id") == env_id), None)
                    self.metrics_collector.record_environment_action(env["environment_type"], "follow_request")

                return


    def follow_reject(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "follow" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for f in env["follow"]:
                        if f["from_agent_id"] == from_agent_id and f["to_agent_id"] == to_agent_id:
                            env["follow"].remove(f)

                            env = next((e for e in self.data if e.get("env_id") == env_id), None)
                            self.metrics_collector.record_environment_action(env["environment_type"], "follow_reject")

                            return
                return


    def follow_accept(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "follow" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for f in env["follow"]:
                        if (
                            f["from_agent_id"] == from_agent_id and
                            f["to_agent_id"] == to_agent_id
                        ):
                            if not f["is_approved"]:
                                f["is_approved"] = True

                                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                                self.metrics_collector.record_environment_action(env["environment_type"], "follow_accept")

                            return
                    return
        

    def connection_request(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "connection" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for c in env["connection"]:
                        if (
                            (c["from_agent_id"] == from_agent_id and c["to_agent_id"] == to_agent_id) or
                            (c["from_agent_id"] == to_agent_id and c["to_agent_id"] == from_agent_id)
                        ):
                            return
                    env["connection"].append({
                        "from_agent_id": from_agent_id,
                        "to_agent_id": to_agent_id,
                        "is_approved": False
                    })

                    env = next((e for e in self.data if e.get("env_id") == env_id), None)
                    self.metrics_collector.record_environment_action(env["environment_type"], "connection_request")
                    
                return


    def connection_reject(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "connection" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for c in env["connection"]:
                        if (
                            (c["from_agent_id"] == from_agent_id and c["to_agent_id"] == to_agent_id) or
                            (c["from_agent_id"] == to_agent_id and c["to_agent_id"] == from_agent_id)
                        ):
                            env["connection"].remove(c)

                            env = next((e for e in self.data if e.get("env_id") == env_id), None)
                            self.metrics_collector.record_environment_action(env["environment_type"], "connection_reject")

                            return
                return


    def connection_accept(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "connection" in env and "profile" in env:
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                
                if from_exists and to_exists:
                    for c in env["connection"]:
                        if (
                            c["from_agent_id"] == from_agent_id and
                            c["to_agent_id"] == to_agent_id
                        ):
                            if not c["is_approved"]:
                                c["is_approved"] = True

                            env = next((e for e in self.data if e.get("env_id") == env_id), None)
                            self.metrics_collector.record_environment_action(env["environment_type"], "connection_accept")

                            return
                    return
    

    def direct_channels_interact(self, env_id, from_agent_id, to_agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "direct_channels" in env:
                
                # Validate profiles
                if "profile" not in env:
                    return
                from_exists = any(p["agent_id"] == from_agent_id for p in env["profile"])
                to_exists = any(p["agent_id"] == to_agent_id for p in env["profile"])
                if not (from_exists and to_exists):
                    return

                # Validate connection if exists
                if "connection" in env:
                    valid_connection = False
                    for c in env["connection"]:
                        if (
                            (
                                c["from_agent_id"] == from_agent_id and 
                                c["to_agent_id"] == to_agent_id
                            ) or (
                                c["from_agent_id"] == to_agent_id and 
                                c["to_agent_id"] == from_agent_id
                            )
                        ) and c["is_approved"]:
                            valid_connection = True
                            break
                    if not valid_connection:
                        return

                # Find or create direct channel
                channel = None
                for ch in env["direct_channels"]:
                    if set(ch["agents_id"]) == {from_agent_id, to_agent_id}:
                        channel = ch
                        break

                if channel is None:
                    channel = {
                        "agents_id": [from_agent_id, to_agent_id],
                        "interactions": []
                    }
                    env["direct_channels"].append(channel)

                # Update previous messages (was_replied = True)
                for inter in channel["interactions"]:
                    if (
                        inter["from_agent_id"] == to_agent_id and
                        inter["to_agent_id"] == from_agent_id
                    ):
                        inter["was_replied"] = True

                # Add new interaction
                channel["interactions"].append({
                    "from_agent_id": from_agent_id,
                    "to_agent_id": to_agent_id,
                    "timestamp": time.time(),
                    "was_replied": False
                })

                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                self.metrics_collector.record_environment_action(env["environment_type"], "direct_channels_interact")

                return
            

    def group_channels_register(self, env_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "group_channels" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                # generate unique short id
                existing_ids = {ch["channel_id"] for ch in env["group_channels"]}
                while True:
                    channel_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                    if channel_id not in existing_ids:
                        break

                env["group_channels"].append({
                    "channel_id": channel_id,
                    "agents": [{
                        "agent_id": agent_id,
                        "is_admin": True
                    }],
                    "interactions": []
                })

                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                self.metrics_collector.record_environment_action(env["environment_type"], "group_channels_register")

                return


    def group_channels_delete(self, env_id, channel_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "group_channels" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                for ch in env["group_channels"]:
                    if ch["channel_id"] == channel_id:
                        for a in ch["agents"]:
                            if a["agent_id"] == agent_id and a["is_admin"]:
                                env["group_channels"].remove(ch)

                                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                                self.metrics_collector.record_environment_action(env["environment_type"], "group_channels_delete")

                                return
                return
    

    def group_channels_agent_delete(self, env_id, channel_id, admin_agent_id, agent_i):
        for env in self.data:
            if env.get("env_id") == env_id and "group_channels" in env and "profile" in env:
                if not any(p["agent_id"] == admin_agent_id for p in env["profile"]):
                    return
                if not any(p["agent_id"] == agent_i for p in env["profile"]):
                    return

                for ch in env["group_channels"]:
                    if ch["channel_id"] == channel_id:
                        # validate admin
                        is_admin = any(
                            a["agent_id"] == admin_agent_id and a["is_admin"]
                            for a in ch["agents"]
                        )
                        if not is_admin:
                            return

                        # cannot delete self
                        if admin_agent_id == agent_i:
                            return

                        for a in ch["agents"]:
                            if a["agent_id"] == agent_i:
                                ch["agents"].remove(a)

                                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                                self.metrics_collector.record_environment_action(env["environment_type"], "group_channels_agent_delete")

                                return
                return


    def group_channels_join(self, env_id, channel_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "group_channels" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                for ch in env["group_channels"]:
                    if ch["channel_id"] == channel_id:
                        if any(a["agent_id"] == agent_id for a in ch["agents"]):
                            return
                        ch["agents"].append({
                            "agent_id": agent_id,
                            "is_admin": False
                        })

                        env = next((e for e in self.data if e.get("env_id") == env_id), None)
                        self.metrics_collector.record_environment_action(env["environment_type"], "group_channels_join")
                        
                        return
                return


    def group_channels_interact(self, env_id, channel_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "group_channels" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                for ch in env["group_channels"]:
                    if ch["channel_id"] == channel_id:
                        if not any(a["agent_id"] == agent_id for a in ch["agents"]):
                            return
                        ch["interactions"].append({
                            "from_agent_id": agent_id,
                            "timestamp": time.time()
                        })

                        env = next((e for e in self.data if e.get("env_id") == env_id), None)
                        self.metrics_collector.record_environment_action(env["environment_type"], "group_channels_interact")

                        return
                return
    
    
    def applications_create(self, env_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "applications" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                # generate unique short id
                existing_ids = {app["application_id"] for app in env["applications"]}
                while True:
                    application_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                    if application_id not in existing_ids:
                        break

                env["applications"].append({
                    "application_id": application_id,
                    "publisher_agent_id": agent_id,
                    "applicants_agents_id": []
                })

                env = next((e for e in self.data if e.get("env_id") == env_id), None)
                self.metrics_collector.record_environment_action(env["environment_type"], "applications_create")
                
                return


    def applications_apply(self, env_id, application_id, agent_id):
        for env in self.data:
            if env.get("env_id") == env_id and "applications" in env and "profile" in env:
                if not any(p["agent_id"] == agent_id for p in env["profile"]):
                    return

                for app in env["applications"]:
                    if app["application_id"] == application_id:
                        if agent_id not in app["applicants_agents_id"]:
                            app["applicants_agents_id"].append(agent_id)

                            env = next((e for e in self.data if e.get("env_id") == env_id), None)
                            self.metrics_collector.record_environment_action(env["environment_type"], "applications_apply")

                        return
                return