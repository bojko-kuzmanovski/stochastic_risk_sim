#!/usr/bin/env python3
import time
import asyncio


class DiscreteEventSimulator:
    """
    Discrete Event Simulator.
    Orchestrates a single DES run: starts async agents and event scheduler,
    runs for max_time seconds, then stops everything.
    """

    def __init__(self, distributions, environments, automata, agents, events,
                 event_scheduler, snapshot_manager):
        self.distributions = distributions
        self.environments = environments
        self.automata = automata
        self.agents = agents
        self.events = events
        self.event_scheduler = event_scheduler
        self.snapshot_manager = snapshot_manager

    async def run_simulation(self, max_time: float = 60.0):
        """
        Run DES for max_time seconds (real time).
        """
        # Start async agents
        await self.agents.start()

        # Start async scheduler
        await self.event_scheduler.start()

        start_time = time.time()

        # Real-time loop
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                break
            await asyncio.sleep(0.1)

        # Stop async scheduler
        await self.event_scheduler.stop()

        # Stop async agents
        await self.agents.stop()