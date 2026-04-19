from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time


@dataclass
class Event:
    """
    Event that can be scheduled and processed by agents.
    """
    
    signal: str                   # Automaton name that handles this event
    origin_agent_id: str          # Agent who created the event
    target_agent_id: str          # Agent who receives the event
    scheduled_for: float          # When to execute (timestamp)
    created_at: float = field(default_factory=time.time)
    env_id: Optional[str] = None  # Environment where event occurs
    payload: Optional[Dict[str, Any]] = None  # Additional data for the event
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.signal:
            raise ValueError("Event must have a signal")
        if not self.origin_agent_id:
            raise ValueError("Event must have an origin_agent_id")
        if not self.target_agent_id:
            raise ValueError("Event must have a target_agent_id")
        if self.scheduled_for < 0:
            raise ValueError(f"scheduled_for must be non-negative, got {self.scheduled_for}")
    
    @property
    def is_self_event(self) -> bool:
        """Check if this event targets the origin agent."""
        return self.origin_agent_id == self.target_agent_id
    
    def to_context(self) -> Dict[str, Any]:
        """Convert event to context dict for automaton step."""
        context = {
            'signal': self.signal,
            'origin_agent_id': self.origin_agent_id,
            'target_agent_id': self.target_agent_id,
            'scheduled_for': self.scheduled_for,
            'created_at': self.created_at,
        }
        if self.env_id:
            context['env_id'] = self.env_id
        if self.payload:
            context['payload'] = self.payload
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'signal': self.signal,
            'origin_agent_id': self.origin_agent_id,
            'target_agent_id': self.target_agent_id,
            'created_at': self.created_at,
            'scheduled_for': self.scheduled_for,
            'env_id': self.env_id,
            'payload': self.payload.copy() if self.payload else None
        }
    
    def __repr__(self) -> str:
        return (f"Event(signal={self.signal}, origin={self.origin_agent_id}, "
                f"target={self.target_agent_id}, scheduled={self.scheduled_for:.2f})")