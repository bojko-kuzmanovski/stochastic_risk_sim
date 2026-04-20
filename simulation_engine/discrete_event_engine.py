#!/usr/bin/env python3
import random
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Event
from simulation_engine.metrics import MetricsCollector
from simulation_engine.event_scheduler import EventScheduler


@dataclass(order=True)
class SimEvent:
    time: float
    event: Event = field(compare=False)


class DiscreteEventSimulator:
    def __init__(self, agents_config, automata_config, distributions_config, 
                 environments_config, events_config, metrics_config = {}):
        self.agents_config = agents_config
        self.automata_config = automata_config
        self.distributions_config = distributions_config
        self.environments_config = environments_config
        self.events_config = events_config
        self.metrics_config = metrics_config
        
        self.metrics_collector = MetricsCollector(metrics_config)

        self.distributions = Distributions(distributions_config)
        self.automata = Automata(automata_config, self.distributions)
        self.agents = Agents(agents_config, self.distributions, self.automata, self.metrics_collector)
        self.environments = Environments(environments_config, self.agents)
        self.event_scheduler = EventScheduler(events_config, self.distributions)
        
        self.event_queue = []
        self.current_time = 0.0
        self.transactions = []
        self.max_time = 10000.0
    
    def _create_event(self, signal: str, origin_agent_id: str, target_agent_id: str,
                      delay: float = 0.0, env_id: Optional[str] = None,
                      payload: Optional[Dict] = None) -> Event:
        return Event(
            signal=signal,
            origin_agent_id=origin_agent_id,
            target_agent_id=target_agent_id,
            scheduled_for=self.current_time + delay,
            env_id=env_id,
            payload=payload
        )
    
    def _schedule_event(self, event: Event) -> None:
        """Add an event to the simulation event queue."""
        sim_event = SimEvent(time=event.scheduled_for, event=event)
        heapq.heappush(self.event_queue, sim_event)
    
    def _process_static_events(self) -> None:
        """Process static events from the event scheduler."""
        due_events = self.event_scheduler.get_due_events(self.current_time, self.agents)
        
        for agent, signal in due_events:
            payload = agent.attributes.copy() if agent.attributes else {}
            event = self._create_event(signal, agent.id, agent.id, delay=0.0, payload=payload)
            agent.receive_event(event)
            self._schedule_event(event)
    
    def _process_agent_events(self) -> None:
        """Process events from all agents."""
        for agent in self.agents.get_all_agents():
            agent.process_events()
    
    def _handle_event(self, sim_event: SimEvent) -> None:
        """Handle a simulation event by routing it to the appropriate agent."""
        event = sim_event.event
        target_agent = self.agents.get_by_id(event.target_agent_id)
        if not target_agent:
            return
        
        target_agent.receive_event(event)
    
    def run_simulation(self, max_time: float = 10000.0) -> None:
        self.max_time = max_time
        
        print("\n" + "="*60)
        print("🌱 Initializing simulation...")
        
        self.event_scheduler.initialize(self.current_time, self.agents)
        
        print(f"   Agents created: {self.agents.total_count}")
        print(f"   Environments created: {self.environments.total_count}")
        print(f"   Automata loaded: {len(self.automata)}")
        print(f"   Distributions loaded: {len(self.distributions)}")
        
        print(f"⏰ Running simulation until time {self.max_time}...")
        events_processed = 0
        
        # Create progress bar
        pbar = tqdm(total=self.max_time, desc="", unit="s", bar_format="⏳ {l_bar}{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}, {rate_fmt}]", ncols=80)
        last_time = self.current_time

        while self.current_time <= self.max_time:
            self._process_static_events()
            self._process_agent_events()
            
            if self.event_queue:
                sim_event = heapq.heappop(self.event_queue)
                self.current_time = sim_event.time
                self._handle_event(sim_event)
                events_processed += 1
                
                if self.current_time - last_time >= 0.1:
                    pbar.n = self.current_time
                    pbar.refresh()
                    last_time = self.current_time
                    pbar.set_postfix({"events": events_processed})
            else:
                next_event_time = self.event_scheduler.get_next_event_time()
                if next_event_time is None or next_event_time > self.max_time:
                    break
                jump = next_event_time - self.current_time
                pbar.n = next_event_time
                pbar.refresh()
                self.current_time = next_event_time

        pbar.close()
        print(f"✅ Simulation complete. {events_processed} events processed.")
        self._compute_and_print_metrics()
    
    def _compute_and_print_metrics(self) -> None:
        citizens = self.agents.get_by_type('citizen')
        active_citizens = [c for c in citizens if c.state == 'active']
        
        self.metrics_collector.set_global_metric('total_citizens', len(citizens))
        self.metrics_collector.set_global_metric('active_mules', len(active_citizens))
        self.metrics_collector.set_global_metric('total_transactions', len(self.transactions))
        
        metrics = self.metrics_collector.compute_metrics(
            self.agents, self.environments, self.transactions
        )
        
        self.metrics_collector.print_report(metrics)