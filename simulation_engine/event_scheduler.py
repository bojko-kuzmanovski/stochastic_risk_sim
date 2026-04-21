from typing import Dict, Any, Optional, List
import time


class EventScheduler:
    """
    Manages static events that trigger periodically to keep agents active.
    """
    
    def __init__(self, events_config: List[Dict[str, Any]], distributions):
        """
        Initialize event scheduler from configuration.
        
        Args:
            events_config: List of event configs with 'signal', 'agent_type', 'periodicity'
            distributions: Distributions instance for sampling
        """
        self._config = events_config
        self._distributions = distributions
        self._last_execution: Dict[str, float] = {}
        self._next_execution: Dict[str, float] = {}
    
    def _get_interval(self, periodicity: Dict[str, Any]) -> float:
        """Calculate next interval based on periodicity configuration."""
        interval_type = periodicity.get('type', 'deterministic')
        
        if interval_type == 'deterministic':
            return periodicity.get('value', 1.0)  # cambiado de 'interval' a 'value'
        elif interval_type == 'probabilistic':
            distribution_name = periodicity.get('distribution')
            if distribution_name:
                return self._distributions.sample(distribution_name)
            return 1.0
        else:
            return 1.0
    
    def initialize(self, current_time: float, agents) -> None:
        """
        Initialize all static events. Sets next execution times.
        Returns None (no immediate events).
        """
        for event_config in self._config:
            if event_config.get('event_category') != 'static':
                continue
            signal = event_config.get('signal')
            if not signal:
                continue
            periodicity = event_config.get('periodicity', {})
            interval = self._get_interval(periodicity)
            
            self._next_execution[signal] = current_time + interval
            self._last_execution[signal] = current_time
    
    def get_due_events(self, current_time: float, agents) -> List[tuple]:
        """
        Get all events that are due for execution.
        
        Returns:
            List of tuples (agent, signal)
        """
        due_events = []
        
        for event_config in self._config:
            if event_config.get('event_category') != 'static':
                continue
            signal = event_config.get('signal')
            if not signal:
                continue
                
            next_time = self._next_execution.get(signal, float('inf'))
            if next_time <= current_time:
                agent_type = event_config.get('agent_type', '*')
                periodicity = event_config.get('periodicity', {})
                
                # Determine target agents
                if agent_type == '*':
                    target_agents = agents.get_all_agents()
                else:
                    target_agents = agents.get_by_type(agent_type)
                
                for agent in target_agents:
                    due_events.append((agent, signal))
                
                # Schedule next execution
                interval = self._get_interval(periodicity)
                self._next_execution[signal] = current_time + interval
                self._last_execution[signal] = current_time
        
        return due_events

    def get_next_event_time(self) -> Optional[float]:
        """Get the earliest scheduled static event time."""
        if not self._next_execution:
            return None
        return min(self._next_execution.values())
    
    def get_static_events_count(self) -> int:
        """Return count of static events."""
        return sum(1 for e in self._config if e.get('event_category') == 'static')