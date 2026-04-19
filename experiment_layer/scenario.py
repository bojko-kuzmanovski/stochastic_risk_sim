#!/usr/bin/env python3
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_engine.discrete_event_engine import DiscreteEventSimulator

def main():
    script_dir = Path(__file__).parent
    
    with open(script_dir / 'agents.json', 'r') as f:
        agents_config = json.load(f)
    
    with open(script_dir / 'automata.json', 'r') as f:
        automata_config = json.load(f)
    
    with open(script_dir / 'distributions.json', 'r') as f:
        distributions_config = json.load(f)
    
    with open(script_dir / 'world.json', 'r') as f:
        world_config = json.load(f)
    
    sim = DiscreteEventSimulator(agents_config, automata_config, distributions_config, world_config)
    stats = sim.run_simulation()
    sim.print_report(stats)

if __name__ == "__main__":
    main()