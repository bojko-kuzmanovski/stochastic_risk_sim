#!/usr/bin/env python3
import time
from tqdm import tqdm
import asyncio

from sim_engine.metrics_collector import MetricsCollector
from sim_engine.event_scheduler import EventScheduler

from patl_engine.snapshot_manager import SnapshotManager
from patl_engine.patl_verifier import PATLVerifier

from sim_core.distributions import Distributions
from sim_core.agents import Agents
from sim_core.automata import Automata
from sim_core.environments import Environments
from sim_core.events import Events


class DiscreteEventSimulator:
    def __init__(self, distributions_config, environments_config, automata_config, agents_config, events_config, patl_config):
        # Metrics collector and snapshot manager
        self.metrics_collector = MetricsCollector()
        self.snapshot_manager = SnapshotManager(patl_config, self.metrics_collector)
        
        # Simulation objects
        self.distributions = Distributions(distributions_config, self.metrics_collector)
        self.environments = Environments(environments_config, self.distributions, self.metrics_collector)
        self.automata = Automata(automata_config, self.distributions, self.metrics_collector)
        self.agents = Agents(agents_config, self.distributions, self.metrics_collector)
        
        # Cross objects sharing
        self.automata.set_objects(self.agents, self.environments)
        self.agents.set_objects(self.automata, self.snapshot_manager)
        self.snapshot_manager.set_objects(self.agents, self.environments)
        
        # Static events and event scheduler
        self.events = Events(events_config, self.distributions)
        static_events = [e for e in self.events.data if e["event_category"] == "static"]
        self.event_scheduler = EventScheduler(static_events, self.agents, self.snapshot_manager)

        # Patl model checker
        self.patl_verifier = PATLVerifier(self.automata, self.distributions, self.metrics_collector)

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

        # PATL model checker
        snapshots = self.snapshot_manager.get_all()
        if snapshots:
            self.metrics_collector.set_enabled(False)
            all_results = []

            pbar = tqdm(
                total=len(snapshots),
                desc="PATL verification",
                unit="snapshot",
                bar_format="⏳ {l_bar}{bar}| {n}/{total}",
                ncols=80
            )

            for snap in snapshots:
                key = (snap["automaton_name"], snap["state"])
                predicates = self.snapshot_manager.data.get(key, [])
                if predicates:
                    results = self.patl_verifier.verify(snap, predicates)
                    all_results.append((snap, results))
                pbar.update(1)

            pbar.close()
            self.metrics_collector.set_enabled(True)

        # Final metrics
        self.metrics_collector.print_report(self.distributions, self.environments, self.agents, self.automata, self.events, self.snapshot_manager, patl_results=all_results if snapshots else None)