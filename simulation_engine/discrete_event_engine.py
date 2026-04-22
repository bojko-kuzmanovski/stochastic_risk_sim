#!/usr/bin/env python3
import time
from tqdm import tqdm

from core.distributions import Distributions
from core.agents import Agents
from core.automata import Automata
from core.environments import Environments
from core.events import Events

from simulation_engine.metrics_collector import MetricsCollector
from simulation_engine.event_scheduler import EventScheduler


class DiscreteEventSimulator:
    def __init__(self, distributions_config, environments_config, automata_config, agents_config, events_config):
        self.metrics_collector = MetricsCollector()
        
        self.distributions = Distributions(distributions_config, self.metrics_collector)
        self.environments = Environments(environments_config, self.distributions, self.metrics_collector)
        self.automata = Automata(automata_config, self.distributions, self.metrics_collector)
        self.agents = Agents(agents_config, self.distributions, self.metrics_collector)
        
        self.automata.set_objects(self.agents, self.environments)
        self.agents.set_objects(self.automata)
        
        events = Events(events_config, self.distributions)
        static_events = [e for e in events.data if e["event_category"] == "static"]
        self.event_scheduler = EventScheduler(static_events, self.agents)

    def run_simulation(self, max_time: float = 10000.0):
        start_time = time.time()

        # Start async scheduler
        self.event_scheduler.start()

        pbar = tqdm(
            total=max_time,
            desc="Simulation time",
            unit="s",
            bar_format="⏳ {l_bar}{bar}| {n:.2f}/{total:.2f}s",
            ncols=80
        )

        # Real-time loop
        while True:
            now = time.time()
            elapsed = now - start_time

            if elapsed >= max_time:
                break

            pbar.n = elapsed
            pbar.refresh()

            time.sleep(0.1)

        pbar.n = max_time
        pbar.refresh()
        pbar.close()

        # Stop scheduler
        self.event_scheduler.stop()

        # Draining phase
        total_agents = len(self.agents.data)

        pbar = tqdm(
            total=total_agents,
            desc="Draining agent queues",
            unit="agent",
            bar_format="⏳ {l_bar}{bar}| {n}/{total}",
            ncols=80
        )

        for i, agent in enumerate(self.agents.data):
            # Passive wait: agent drains itself
            while agent["event_queue"]:
                time.sleep(0.1)

            pbar.n = i + 1
            pbar.refresh()

        pbar.close()

        # Final metrics
        self.metrics_collector.print_report()