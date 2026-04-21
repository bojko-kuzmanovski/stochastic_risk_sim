#!/usr/bin/env python3
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Events

from simulation_engine.metrics_collector import MetricsCollector
from simulation_engine.event_scheduler import EventScheduler


@dataclass
class Event:
    signal: str
    target_agent_id: str
    scheduled_for: float
    origin_agent_id: Optional[str] = None
    env_id: Optional[str] = None
    payload: Optional[Dict] = None


@dataclass(order=True)
class SimEvent:
    time: float
    event: Event = field(compare=False)


class DiscreteEventSimulator:
    def __init__(self, agents_config, automata_config, distributions_config, environments_config, events_config):
        
        self.distributions = Distributions(distributions_config)
        self.automata = Automata(automata_config, self.distributions)
        self.agents = Agents(agents_config, self.distributions)
        self.environments = Environments(environments_config, self.distributions)

        self.event_scheduler = EventScheduler(events_config, self.distributions)
        self.metrics_collector = MetricsCollector()
        self.metrics_collector.collect_initial_stats(
            distributions=self.distributions,
            environments=self.environments,
            agents=self.agents,
            automata=self.automata,
            event_scheduler=self.event_scheduler
        )

        self.event_queue = []
        self.current_time = 0.0
        self.events_processed = 0
    
    def _process_agent_queues(self):
        """Process one event from each agent's queue if available."""
        for agent in self.agents.data:
            if agent['event_queue']:
                event = agent['event_queue'].pop(0)
                self._execute_automaton(agent, event)

    def _execute_automaton(self, agent: Dict, event: Dict):
        """Execute the appropriate automaton for this event."""
        signal = event.get('signal')
        automaton = next((a for a in self.automata.data if a['automaton_name'] == signal), None)
        if not automaton:
            return
        
        current_state = automaton['states']['initial']
        
        for transition in automaton['transitions']:
            if transition['from'] != current_state:
                continue
            
            threshold = transition['thresholds'][0]
            next_state = threshold['to']
            
            if next_state in automaton['states']['final']:
                self.metrics_collector.record_automaton_result(signal, next_state)
            
            if 'emit_intent' in threshold:
                intent = threshold['emit_intent']
                new_event = Event(
                    signal=intent,
                    target_agent_id=agent['agent_id'],
                    scheduled_for=self.current_time + 0.1
                )
                sim_event = SimEvent(time=new_event.scheduled_for, event=new_event)
                heapq.heappush(self.event_queue, sim_event)

    def run_simulation(self, max_time: float = 10000.0):
        self.max_time = max_time
        self.event_scheduler.initialize(self.current_time, self.agents)
        
        pbar = tqdm(total=self.max_time, desc="", unit="u", 
                    bar_format="⏳ {l_bar}{bar}| {n:.2f}/{total:.2f}u", ncols=80)
        last_time = self.current_time
        
        while self.current_time <= self.max_time:
            due_events = self.event_scheduler.get_due_events(self.current_time, self.agents)
            for agent, signal in due_events:
                self.agents.receive_event(agent['agent_id'], {'signal': signal})
            
            self._process_agent_queues()
            
            if self.event_queue:
                sim_event = heapq.heappop(self.event_queue)
                self.current_time = sim_event.time
                self.events_processed += 1
            else:
                next_t = self.event_scheduler.get_next_event_time()
                if next_t is None or next_t > self.max_time:
                    break
                self.current_time = next_t
            
            if self.current_time - last_time >= 0.1:
                pbar.n = min(self.current_time, self.max_time)
                pbar.refresh()
                last_time = self.current_time
                pbar.set_postfix({"events": self.events_processed})
        
        pbar.n = self.max_time
        pbar.refresh()
        pbar.close()
        
        self.metrics_collector.set_final_stats(
            events_processed=self.events_processed,
            max_time=self.max_time
        )
        self.metrics_collector.print_report()
