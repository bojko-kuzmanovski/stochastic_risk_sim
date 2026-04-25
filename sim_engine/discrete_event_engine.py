#!/usr/bin/env python3
import time
from tqdm import tqdm
import asyncio

from sim_core.distributions import Distributions
from sim_core.agents import Agents
from sim_core.automata import Automata
from sim_core.environments import Environments
from sim_core.events import Events

from sim_engine.metrics_collector import MetricsCollector
from sim_engine.event_scheduler import EventScheduler


class DiscreteEventSimulator:
    def __init__(self, distributions_config, environments_config, automata_config, agents_config, events_config):
        self.metrics_collector = MetricsCollector()
        
        self.distributions = Distributions(distributions_config, self.metrics_collector)
        self.environments = Environments(environments_config, self.distributions, self.metrics_collector)
        self.automata = Automata(automata_config, self.distributions, self.metrics_collector)
        self.agents = Agents(agents_config, self.distributions, self.metrics_collector)
        
        self.automata.set_objects(self.agents, self.environments)
        self.agents.set_objects(self.automata)
        
        self.events = Events(events_config, self.distributions)
        static_events = [e for e in self.events.data if e["event_category"] == "static"]
        self.event_scheduler = EventScheduler(static_events, self.agents)

    async def run_simulation(self, max_time: float = 10000.0):
        # Start async agents
        await self.agents.start()

        # Start async scheduler
        await self.event_scheduler.start()

        start_time = time.time()

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

            await asyncio.sleep(0.1)

        pbar.n = max_time
        pbar.refresh()
        pbar.close()

        # Stop async scheduler
        await self.event_scheduler.stop()

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
            while not agent["event_queue"].empty():
                await asyncio.sleep(0.1)

            pbar.n = i + 1
            pbar.refresh()

        pbar.close()

        # Detener agentes asíncronos
        await self.agents.stop()

        # Final metrics
        self.metrics_collector.print_report(self.distributions, self.environments, self.agents, self.automata, self.events)