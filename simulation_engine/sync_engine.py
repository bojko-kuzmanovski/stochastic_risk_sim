from typing import List, Dict, Any

from simulation_engine.event import Event
from world_state.state import WorldState
from rule_engine.rule_engine import RuleEngine
from multi_agent_system.agent import Agent


class SimulationEngine:
    """
    Core simulation loop (discrete time).
    """

    def __init__(
        self,
        agents: List[Agent],
        state: WorldState,
        rule_engine: RuleEngine,
    ):
        self.agents = agents
        self.state = state
        self.rule_engine = rule_engine

        self.current_tick = 0

    # ---------------------------------------------------------
    # INTERNAL
    # ---------------------------------------------------------
    def _dict_to_event(self, raw: Dict[str, Any], agent: Agent) -> Event:
        return Event(
            event_type=raw["event_type"],
            payload=raw.get("payload", {}),
            source=agent.id,
            context=raw.get("context"),
        )

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------
    def step(self) -> None:
        self.current_tick += 1

        for agent in self.agents:
            raw_intention = agent.run(self.state)

            if raw_intention is None:
                continue

            event = self._dict_to_event(raw_intention, agent)

            if not event.is_valid():
                continue

            self.rule_engine.apply(event, self.state)

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------
    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()