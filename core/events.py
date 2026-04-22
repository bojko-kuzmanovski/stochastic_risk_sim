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
            event_resolved["created_at"] = int(time.time())

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
                # optional payload passthrough
                if "payload" not in event_resolved:
                    event_resolved["payload"] = {}

            self.data.append(event_resolved)