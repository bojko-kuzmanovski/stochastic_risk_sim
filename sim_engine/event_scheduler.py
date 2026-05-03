import asyncio

class EventScheduler:
    def __init__(self, static_events, agents, snapshot_manager):
        self._events = static_events
        self._agents = agents
        self._snapshot_manager = snapshot_manager
        self._running = False
        self._tasks = []

    async def start(self):
        self._running = True
        
        for event in self._events:
            task = asyncio.create_task(self._run_event_loop(event))
            self._tasks.append(task)

    async def _run_event_loop(self, event):
        while self._running:
            if self._snapshot_manager:
                while self._snapshot_manager.is_sampling():
                    await asyncio.sleep(0.01)
                self._snapshot_manager.enter_transition()

            target_agents = self._agents.get_all_agents(event["agent_type"])
            for agent_id in target_agents:
                event_copy = event.copy()
                event_copy["agent_id"] = agent_id
                await self._agents.receive_event(agent_id, event_copy)

            if self._snapshot_manager:
                self._snapshot_manager.exit_transition()

            await asyncio.sleep(event["periodicity"])

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)