from typing import Dict, Any, Optional, List
import time
import random


class EventScheduler:
    """
    Manages static events that trigger periodically to keep agents active.
    """
    
    def __init__(self, events_config: Dict[str, Any], distributions):
        """
        Initialize event scheduler from configuration.
        
        Args:
            events_config: Full events.json dict
            distributions: Distributions instance for sampling
        """
        self._config = events_config
        self._distributions = distributions
        self._static_events = events_config.get('static_events', [])
        self._last_execution: Dict[str, float] = {}
        self._next_execution: Dict[str, float] = {}
    
    def _get_interval(self, event_config: Dict[str, Any]) -> float:
        """Calculate next interval based on periodicity configuration."""
        periodicity = event_config.get('periodicity', {})
        interval_type = periodicity.get('type', 'deterministic')
        
        if interval_type == 'deterministic':
            return periodicity.get('interval', 1.0)
        elif interval_type == 'probabilistic':
            distribution_name = periodicity.get('distribution')
            if distribution_name and distribution_name in self._distributions:
                return self._distributions.sample(distribution_name)
            return 1.0
        else:
            return 1.0
    
    def _get_event_key(self, event_config: Dict[str, Any]) -> str:
        """Generate unique key for an event."""
        return f"{event_config.get('signal')}_{event_config.get('agent_type')}"
    
    def initialize(self, current_time: float) -> None:
        """Initialize all static events with their first execution times."""
        for event_config in self._static_events:
            key = self._get_event_key(event_config)
            interval = self._get_interval(event_config)
            self._next_execution[key] = current_time + interval
            self._last_execution[key] = current_time
    
    def get_due_events(self, current_time: float, agents) -> List[tuple]:
        """
        Get all events that are due for execution.
        
        Returns:
            List of tuples (agent, event_signal)
        """
        due_events = []
        
        for event_config in self._static_events:
            key = self._get_event_key(event_config)
            
            # Check if event is due
            if self._next_execution.get(key, float('inf')) <= current_time:
                agent_type = event_config.get('agent_type')
                signal = event_config.get('signal')
                
                # Get all agents of this type
                target_agents = agents.get_by_type(agent_type)
                
                for agent in target_agents:
                    due_events.append((agent, signal))
                
                # Schedule next execution
                interval = self._get_interval(event_config)
                self._next_execution[key] = current_time + interval
                self._last_execution[key] = current_time
        
        return due_events
    
    def should_add_event(self, agent, signal: str) -> bool:
        """
        Check if an event should be added to agent's queue.
        Avoids duplicate events of the same signal.
        """
        # Check if agent already has pending event with same signal
        # This requires access to agent's event queue
        return True  # For now, always add
    
    def __repr__(self) -> str:
        return f"EventScheduler(events={len(self._static_events)})"