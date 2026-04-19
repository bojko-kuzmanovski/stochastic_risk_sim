from typing import Callable, Dict
from simulation_engine.event import Event
from world_state.state import WorldState


class RuleEngine:
    """
    Minimal rule engine:
    event_type → transformation function
    """

    def __init__(self):
        self.rules: Dict[str, Callable[[Event, WorldState], None]] = {}

    # ---------------------------------------------------------
    # REGISTRATION
    # ---------------------------------------------------------
    def register(self, event_type: str, handler: Callable[[Event, WorldState], None]) -> None:
        self.rules[event_type] = handler

    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    def apply(self, event: Event, state: WorldState) -> None:
        handler = self.rules.get(event.event_type)

        if handler:
            handler(event, state)