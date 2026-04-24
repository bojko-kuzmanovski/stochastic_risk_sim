import asyncio

class EventScheduler:
    def __init__(self, static_events, agents):
        self._events = static_events
        self._agents = agents
        self._running = False
        self._tasks = []

    async def start(self):
        self._running = True
        
        for event in self._events:
            task = asyncio.create_task(self._run_event_loop(event))
            self._tasks.append(task)

    async def _run_event_loop(self, event):
        while self._running:
            # Select agents
            target_agents = self._agents.get_by_type(event["agent_type"])
            
            # Emit event
            for agent in target_agents:
                event_copy = event.copy()
                event_copy["agent_id"] = agent["agent_id"]
                await self._agents.receive_event(agent["agent_id"], event_copy)
                
            # Periodicity
            await asyncio.sleep(event["periodicity"])

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)