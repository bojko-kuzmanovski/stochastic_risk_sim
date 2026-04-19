#!/usr/bin/env python3
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from simulation_engine.discrete_event_engine import DiscreteEventSimulator


def main():
    script_dir = Path(__file__).parent / 'configs'
    
    # Load all configurations
    with open(script_dir / 'agents.json', 'r') as f:
        agents_config = json.load(f)
    
    with open(script_dir / 'automata.json', 'r') as f:
        automata_config = json.load(f)
    
    with open(script_dir / 'distributions.json', 'r') as f:
        distributions_config = json.load(f)
    
    with open(script_dir / 'environments.json', 'r') as f:
        environments_config = json.load(f)
    
    with open(script_dir / 'events.json', 'r') as f:
        events_config = json.load(f)
    
    with open(script_dir / 'metrics.json', 'r') as f:
        metrics_config = json.load(f)
    
    # Create and run simulator
    sim = DiscreteEventSimulator(
        agents_config=agents_config,
        automata_config=automata_config,
        distributions_config=distributions_config,
        environments_config=environments_config,
        events_config=events_config,
        metrics_config=metrics_config
    )
    
    # Run simulation with hardcoded max time
    sim.run_simulation(max_time=40.0)


if __name__ == "__main__":
    main()