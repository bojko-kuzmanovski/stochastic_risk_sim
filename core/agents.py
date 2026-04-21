import json
from jsonschema import validate
from typing import Dict

class Agents:
    def __init__(self, config_data, distributions):
        # Load schema file
        schema_path = "schemas/agents.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate data against schema
        validate(instance=config_data, schema=schema)

        # Process agents data
        self.data = []

        for agent_entry in config_data:
            for n in range(1, agent_entry["quantity"] + 1):
                # Copy entry to avoid mutating original data
                agent_resolved = agent_entry.copy()

                # Assign agent_id
                agent_resolved["agent_id"] = f"{agent_entry['agent_type']}_{n}"

                # Resolve parameters
                resolved_params = {}
                for param_name, param_def in agent_entry.get("params", {}).items():
                    if param_def["type"] == "deterministic":
                        resolved_params[param_name] = param_def["value"]

                    elif param_def["type"] == "probabilistic":
                        dist_name = param_def["distribution"]
                        resolved_params[param_name] = distributions.sample(dist_name)

                # Assign resolved parameters
                agent_resolved["params"] = resolved_params

                # Ensure event_queue exists (required by schema)
                if "event_queue" not in agent_resolved:
                    agent_resolved["event_queue"] = []

                # Store in internal list
                self.data.append(agent_resolved)

    def get_by_type(self, agent_type: str):
        """Return all agents of a specific type."""
        return [a for a in self.data if a.get('agent_type') == agent_type]

    def get_all_agents(self):
        """Return all agents."""
        return self.data
    
    def receive_event(self, agent_id: str, event: Dict) -> None:
        """Add an event to the agent's event queue."""
        for agent in self.data:
            if agent['agent_id'] == agent_id:
                agent['event_queue'].append(event)
                return