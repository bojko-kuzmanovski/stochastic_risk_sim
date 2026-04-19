from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass
class Event:
    """
    Core event abstraction (Σ_risk).

    This is the fundamental unit that flows through the system.
    """

    event_type: str
    payload: Dict[str, Any]
    source: str

    timestamp: float = field(default_factory=lambda: time.time())
    context: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    def is_valid(self) -> bool:
        return True  # Placeholder for future validation logic

    # ---------------------------------------------------------
    # UTIL
    # ---------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Event(type={self.event_type}, "
            f"source={self.source})"
        )