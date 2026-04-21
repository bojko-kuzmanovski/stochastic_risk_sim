import json
import time
from jsonschema import validate

class Events:
    def __init__(self, config_data, distributions):
        # Load schema file
        schema_path = "schemas/events.schema.json"
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate data against schema
        validate(instance=config_data, schema=schema)

        # Process events
        self.data = []

        for event_entry in config_data:

            event_resolved = event_entry.copy()

            # STATIC EVENTS
            if event_entry["event_category"] == "static":
                # periodicity resolution
                periodicity_def = event_entry["periodicity"]
                if periodicity_def["type"] == "deterministic":
                    periodicity = periodicity_def["value"]
                else:
                    periodicity = distributions.sample(periodicity_def["distribution"])

                event_resolved["periodicity"] = periodicity

            # DYNAMIC EVENTS
            else:
                event_resolved["created_at"] = int(time.time())

                # optional payload passthrough (already dynamic structure)
                if "payload" not in event_resolved:
                    event_resolved["payload"] = {}

                # ensure minimal normalization of nested objects
                if "to_agent" in event_resolved:
                    event_resolved["to_agent"] = event_entry["to_agent"]

                if "from_agent" in event_resolved:
                    event_resolved["from_agent"] = event_entry["from_agent"]

                if "env" in event_resolved:
                    event_resolved["env"] = event_entry["env"]

            self.data.append(event_resolved)