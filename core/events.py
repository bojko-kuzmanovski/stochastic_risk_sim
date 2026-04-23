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
                pdef = event_entry["periodicity"]
                if pdef["type"] == "deterministic":
                    periodicity = pdef["value"]
                else:
                    dist = pdef["distribution"]
                    refs = pdef.get("refs", {})
                    periodicity = distributions.sample(dist, refs) if refs else distributions.sample(dist)

                event_resolved["periodicity"] = periodicity

            # DYNAMIC EVENTS

            self.data.append(event_resolved)