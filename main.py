#!/usr/bin/env python3
import json
from pathlib import Path
import sys
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent))

from simulation_engine.discrete_event_engine import DiscreteEventSimulator

def load_json(path):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)
    
def main():
    script_dir = Path(__file__).parent / 'configs'
    
    # Load all configurations
    distributions_config = load_json(script_dir / 'distributions.json')
    environments_config = load_json(script_dir / 'environments.json')
    automata_config = load_json(script_dir / 'automata.json')
    agents_config = load_json(script_dir / 'agents.json')
    events_config = load_json(script_dir / 'events.json')
    
    # Create and run simulator
    sim = DiscreteEventSimulator(
        distributions_config=distributions_config,
        environments_config=environments_config,
        automata_config=automata_config,
        agents_config=agents_config,
        events_config=events_config
    )
    
    # Run simulation with hardcoded max time
    sim.run_simulation(max_time=40.0)


if __name__ == "__main__":
    main()