#!/usr/bin/env python3
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Events

from simulation_engine.metrics_collector import MetricsCollector
from simulation_engine.event_scheduler import EventScheduler


@dataclass(order=True)
class SimEvent:
    time: float
    event: Events = field(compare=False)


class DiscreteEventSimulator:
    def __init__(self, agents_config, automata_config, distributions_config, environments_config, events_config):
        self.distributions = Distributions(distributions_config)
        self.automata = Automata(automata_config, self.distributions)
        self.agents = Agents(agents_config, self.distributions)
        self.environments = Environments(environments_config, self.distributions)
        self.event_scheduler = EventScheduler(events_config, self.distributions)
        self.event_queue = []
        self.current_time = 0.0
    
    def _print_stats(self):
        dist_by_family = {}
        for name, info in self.distributions.samplers.items():
            fam = info.get('family', 'unknown')
            dist_by_family[fam] = dist_by_family.get(fam, 0) + 1
        
        env_by_type = {}
        for e in self.environments.data:
            t = e.get('environment_type', 'unknown')
            env_by_type[t] = env_by_type.get(t, 0) + 1
        
        agent_by_type = {}
        for a in self.agents.data:
            t = a.get('agent_type', 'unknown')
            agent_by_type[t] = agent_by_type.get(t, 0) + 1
        
        det_trans = prob_trans = 0
        for aut in self.automata.data:
            for t in aut.get('transitions', []):
                if t.get('type') == 'deterministic':
                    det_trans += 1
                elif t.get('type') == 'probabilistic':
                    prob_trans += 1
        
        static_events = self.event_scheduler.get_static_events_count()
        
        print("\n" + "="*60)
        print("📊 SIMULATION INITIALIZATION STATS")
        print("="*60)
        print(f"\n📈 Distributions: {len(self.distributions.samplers)}")
        for fam, cnt in sorted(dist_by_family.items()):
            print(f"   └── {fam}: {cnt}")
        
        print(f"\n🌍 Environments: {len(self.environments.data)}")
        for t, cnt in sorted(env_by_type.items()):
            print(f"   └── {t}: {cnt}")
        
        print(f"\n👤 Agents: {len(self.agents.data)}")
        for t, cnt in sorted(agent_by_type.items()):
            print(f"   └── {t}: {cnt}")
        
        print(f"\n🤖 Automata: {len(self.automata.data)}")
        print(f"   └── deterministic transitions: {det_trans}")
        print(f"   └── probabilistic transitions: {prob_trans}")
        
        print(f"\n📅 Static Events: {static_events}")
        print("="*60 + "\n")
    
    def run_simulation(self, max_time: float = 10000.0):
        self._print_stats()
        self.event_scheduler.initialize(self.current_time, self.agents)
        
        while self.current_time <= max_time:
            self.event_scheduler.get_due_events(self.current_time, self.agents)
            
            if not self.event_queue:
                next_t = self.event_scheduler.get_next_event_time()
                if next_t is None or next_t > max_time:
                    break
                self.current_time = next_t
            else:
                self.current_time = heapq.heappop(self.event_queue).time