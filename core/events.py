import json
import time
from jsonschema import validate

from core.utils.evaluator import resolve_value

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
                pdef = event_entry["periodicity"]
                periodicity = resolve_value(pdef, {}, distributions, None, None)

                if not isinstance(periodicity, (int, float)):
                    raise TypeError(
                        f"periodicity must resolve to a number, got {type(periodicity).__name__}: "
                        f"{periodicity} for signal '{event_entry['signal']}'"
                    )
                if periodicity <= 0:
                    raise ValueError(
                        f"periodicity must be positive, got {periodicity} "
                        f"for signal '{event_entry['signal']}'"
                    )
                
                event_resolved["periodicity"] = periodicity

            # DYNAMIC EVENTS (no resolution needed)

            self.data.append(event_resolved)