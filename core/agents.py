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

                # Resolve params
                resolved_params = {}
                for param_name, param_def in agent_entry.get("params", {}).items():
                    if param_def["type"] == "deterministic":
                        resolved_params[param_name] = param_def["value"]

                    elif param_def["type"] == "probabilistic":
                        dist_name = param_def["distribution"]
                        resolved_params[param_name] = distributions.sample(dist_name)

                # Assign resolved params
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
    
    def receive_event(self, agent_id, event):
        agent = self._get_agent(agent_id)
        agent["event_queue"].append(event)
        self._process_agent_queue(agent)

    def _process_agent_queue(self, agent):
        batch_size = agent["params"].get("event_processing_batch_size", 1)

        processed = 0
        while agent["event_queue"] and processed < batch_size:
            event = agent["event_queue"].pop(0)
            self._execute_automaton(agent, event)
            processed += 1
    
    def _execute_automaton(self, agent, event):
        signal = event.get("signal")

        automaton = next(
            (a for a in self.automata.data if a["automaton_name"] == signal),
            None
        )
        if not automaton:
            return

        current_state = automaton["states"]["initial"]

        for transition in automaton["transitions"]:
            if transition["from"] != current_state:
                continue

            threshold = transition["thresholds"][0]
            next_state = threshold["to"]

            if "emit_intent" in threshold:
                new_event = {
                    "signal": threshold["emit_intent"],
                    "from_agent": {
                        "agent_id": agent["agent_id"],
                        "agent_type": agent["agent_type"]
                    }
                }

                self.agents.receive_event(agent["agent_id"], new_event)