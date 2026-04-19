#!/usr/bin/env python3
import random
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.distribution import Distributions
from core.agent import Agents
from core.automaton import Automata
from core.environment import Environments
from core.event import Event
from core.metrics import MetricsCollector
from simulation_engine.event_scheduler import EventScheduler

@dataclass(order=True)
class SimEvent:
    time: float
    event_type: str
    agent_id: str = field(compare=False)
    target_id: str = field(compare=False)


class DiscreteEventSimulator:
    def __init__(self, agents_config, automata_config, distributions_config, 
                 environments_config, events_config, metrics_config):
        """
        Initialize the discrete event simulator.
        
        Args:
            agents_config: agents.json dict
            automata_config: automata.json dict
            distributions_config: distributions.json dict
            environments_config: environments.json dict
            events_config: events.json dict
            metrics_config: metrics.json dict
        """
        self.agents_config = agents_config
        self.automata_config = automata_config
        self.distributions_config = distributions_config
        self.environments_config = environments_config
        self.events_config = events_config
        self.metrics_config = metrics_config
        
        # Initialize core components
        self.distributions = Distributions(distributions_config)
        self.automata = Automata(automata_config, self.distributions)
        self.agents = Agents(agents_config, self.distributions, self.automata)
        self.environments = Environments(environments_config, self.agents)
        self.event_scheduler = EventScheduler(events_config, self.distributions)
        self.metrics_collector = MetricsCollector(metrics_config)
        
        # Simulation state
        self.event_queue = []
        self.current_time = 0.0
        self.transactions = []
        self.max_time = 10000.0  # Default, can be overridden
    
    def _create_event(self, signal: str, origin_agent_id: str, target_agent_id: str,
                      delay: float = 0.0, env_id: Optional[str] = None,
                      payload: Optional[Dict] = None) -> Event:
        """Create a new event with the current time."""
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
        sim_event = SimEvent(
            time=event.scheduled_for,
            event_type=event.signal,
            agent_id=event.target_agent_id,
            target_id=event.origin_agent_id
        )
        heapq.heappush(self.event_queue, sim_event)
    
    def _process_static_events(self) -> None:
        """Process static events from the event scheduler."""
        due_events = self.event_scheduler.get_due_events(self.current_time, self.agents)
        
        for agent, signal in due_events:
            # Create self-event for the agent
            event = self._create_event(signal, agent.id, agent.id, delay=0.0)
            agent.receive_event(event)
    
    def _process_agent_events(self) -> None:
        """Process events from all agents."""
        for agent in self.agents.get_all_agents():
            # Process events according to agent's batch size
            agent.process_events()
    
    def _handle_event(self, event: SimEvent) -> None:
        """
        Handle a simulation event by routing it to the appropriate agent.
        """
        target_agent = self.agents.get_by_id(event.agent_id)
        if not target_agent:
            return
        
        # Create domain event from simulation event
        domain_event = Event(
            signal=event.event_type,
            origin_agent_id=event.target_id,
            target_agent_id=event.agent_id,
            scheduled_for=event.time
        )
        
        # Deliver event to target agent
        target_agent.receive_event(domain_event)
    
    def initialize(self) -> None:
        """Initialize the simulation."""
        print("\n" + "="*60)
        print("🌱 Initializing simulation...")
        
        # Initialize event scheduler
        self.event_scheduler.initialize(self.current_time)
        
        # Process initial static events
        self._process_static_events()
        
        print(f"   Agents created: {self.agents.total_count}")
        print(f"   Environments created: {self.environments.total_count}")
        print(f"   Automata loaded: {len(self.automata)}")
        print(f"   Distributions loaded: {len(self.distributions)}")
    
    def run_simulation(self, max_time: float = 10000.0) -> None:
        """
        Run the discrete event simulation.
        
        Args:
            max_time: Maximum simulation time
        """
        self.max_time = max_time
        
        self.initialize()
        
        print(f"⏰ Running simulation until time {self.max_time}...")
        events_processed = 0
        
        while self.event_queue and self.current_time <= self.max_time:
            # Pop next event from queue
            sim_event = heapq.heappop(self.event_queue)
            self.current_time = sim_event.time
            
            # Handle the event
            self._handle_event(sim_event)
            events_processed += 1
            
            # Process static events periodically
            self._process_static_events()
            
            # Process agent event queues
            self._process_agent_events()
            
            # Progress indicator
            if events_processed % 1000 == 0:
                print(f"   Time: {self.current_time:.1f}, Events: {events_processed}")
        
        print(f"✅ Simulation complete. {events_processed} events processed.")
        
        # Compute and print metrics
        self._compute_and_print_metrics()
    
    def _compute_and_print_metrics(self) -> None:
        """Compute all metrics and print report."""
        # Set global metrics
        citizens = self.agents.get_by_type('citizen')
        active_citizens = [c for c in citizens if c.state == 'active']
        
        self.metrics_collector.set_global_metric('total_citizens', len(citizens))
        self.metrics_collector.set_global_metric('active_mules', len(active_citizens))
        self.metrics_collector.set_global_metric('total_transactions', len(self.transactions))
        
        # Compute all metrics
        metrics = self.metrics_collector.compute_metrics(
            self.agents, self.environments, self.transactions
        )
        
        # Print report
        self.metrics_collector.print_report(metrics)