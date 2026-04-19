from typing import Dict, Any, Optional, List
import heapq
import time

from core.event import Event


class Agent:
    """
    Generic agent that can represent any agent type from config.
    Internal class - should not be used directly from outside.
    """
    
    class EventQueue:
        """
        Internal priority queue for events ordered by scheduled_for.
        Each agent has its own queue.
        """
        
        def __init__(self):
            self._queue: List[tuple] = []
            self._counter = 0
        
        def push(self, event: 'Event') -> None:
            """Add an event to the queue."""
            heapq.heappush(self._queue, (event.scheduled_for, self._counter, event))
            self._counter += 1
        
        def pop(self) -> Optional['Event']:
            """Remove and return the next event."""
            if not self._queue:
                return None
            _, _, event = heapq.heappop(self._queue)
            return event
        
        def peek(self) -> Optional['Event']:
            """Return the next event without removing it."""
            if not self._queue:
                return None
            return self._queue[0][2]
        
        def peek_time(self) -> Optional[float]:
            """Return scheduled_for time of next event."""
            if not self._queue:
                return None
            return self._queue[0][0]
        
        def is_empty(self) -> bool:
            return len(self._queue) == 0
        
        def size(self) -> int:
            return len(self._queue)
        
        def clear(self) -> None:
            self._queue.clear()
            self._counter = 0
        
        def __len__(self) -> int:
            return len(self._queue)
        
        def __repr__(self) -> str:
            return f"EventQueue(size={len(self._queue)})"
    
    def __init__(self, agent_type: str, config: Dict[str, Any], distributions, automata, metrics_collector, agents, index: int):
        """
        Initialize an agent from configuration.
        
        Args:
            agent_type: Type of agent (e.g., 'citizen', 'malicious')
            config: Agent configuration dict
            distributions: Distributions instance for sampling
            automata: Automata instance for event processing
            metrics_collector: MetricsCollector instance for recording results
            agents: Agents registry for finding other agents
            index: Unique index for this agent instance
        """
        self._type = agent_type
        self._config = config
        self._distributions = distributions
        self._automata = automata
        self._metrics_collector = metrics_collector
        self._agents = agents
        self._id = f"{config['id_prefix']}{index+1:06d}"
        self._attributes: Dict[str, Any] = {}
        self._state: str = 'idle'
        self._event_queue = self.EventQueue()
        
        self._validate_distributions()
        self._generate_attributes()
    
    def _validate_distributions(self) -> None:
        """Validate that distributions is a Distributions instance."""
        if not hasattr(self._distributions, 'sample'):
            raise TypeError(f"distributions must have a 'sample' method. Got {type(self._distributions)}")
    
    def _validate_param_config(self, param_name: str, param_config: Dict) -> None:
        """Validate a single parameter configuration."""
        param_type = param_config.get('type')
        if param_type not in ['probabilistic', 'deterministic']:
            raise ValueError(f"Parameter '{param_name}' has invalid type '{param_type}'")
        
        if param_type == 'probabilistic':
            distribution = param_config.get('distribution')
            if not distribution or not isinstance(distribution, str):
                raise ValueError(f"Parameter '{param_name}' is probabilistic but missing 'distribution' string")
            
            if distribution not in self._distributions:
                raise ValueError(f"Distribution '{distribution}' not found for parameter '{param_name}'")
    
    def _evaluate_formula(self, formula: str, context: Dict[str, Any]) -> float:
        """Safely evaluate a formula string with given context."""
        import math
        allowed = {"__builtins__": {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}}
        return float(eval(formula, allowed, context))
    
    def _generate_attribute(self, param_name: str, param_config: Dict) -> Any:
        """Generate a single attribute value from its configuration."""
        param_type = param_config.get('type')
        
        if param_type == 'deterministic':
            return param_config.get('value', 0.0)
        
        elif param_type == 'probabilistic':
            distribution_name = param_config.get('distribution')
            
            refs = param_config.get('refs', {})
            bound_params = {}
            
            for ref_name, formula in refs.items():
                bound_params[ref_name] = self._evaluate_formula(formula, self._attributes)
            
            if bound_params:
                return self._distributions.sample(distribution_name, bound_params)
            else:
                return self._distributions.sample(distribution_name)
        
        else:
            raise ValueError(f"Unknown type '{param_type}' for parameter '{param_name}'")
    
    def _generate_attributes(self) -> None:
        """Generate all attributes for this agent."""
        params_config = self._config.get('params', {})
        
        for param_name, param_config in params_config.items():
            self._validate_param_config(param_name, param_config)
            self._attributes[param_name] = self._generate_attribute(param_name, param_config)
    
    # Public methods for event handling
    
    def send_event(self, target_agent: 'Agent', event: 'Event') -> None:
        """Send an event to another agent."""
        target_agent.receive_event(event)
    
    def receive_event(self, event: 'Event') -> None:
        """Receive an event from another agent."""
        self._event_queue.push(event)
    
    def process_events(self, max_events: Optional[int] = None) -> int:
        """Process events from the queue and generate response events."""
        if max_events is None:
            max_events = int(self.get_attribute('event_processing_batch_size', 1))
        
        processed = 0
        current_time = time.time()
        
        while processed < max_events and not self._event_queue.is_empty():
            next_event = self._event_queue.peek()
            
            if next_event.scheduled_for > current_time:
                break
            
            event = self._event_queue.pop()
            
            # Execute the automaton
            result = self._automata.step(event.signal, event.to_context())
            
            # Record the result in metrics collector
            if result and self._metrics_collector:
                self._metrics_collector.record_automaton_result(event.signal, result)
            
            # Create new event from result (if it's a success signal that needs to be sent back)
            if result and result not in ['CONTACT_REJECTED', 'RECRUITMENT_REJECT']:
                # The target is the agent who sent the original event
                target_agent = self._agents.get_by_id(event.origin_agent_id)
                
                if target_agent:
                    new_event = Event(
                        signal=result,
                        origin_agent_id=self._id,
                        target_agent_id=target_agent.id,
                        scheduled_for=current_time + 0.1,
                        env_id=event.env_id,
                        payload=event.payload
                    )
                    target_agent.receive_event(new_event)
            
            processed += 1
        
        return processed
    
    def get_next_event_time(self) -> Optional[float]:
        """Get scheduled time of next pending event."""
        return self._event_queue.peek_time()
    
    def has_pending_events(self) -> bool:
        """Check if agent has pending events."""
        return not self._event_queue.is_empty()
    
    def pending_event_count(self) -> int:
        """Get number of pending events."""
        return self._event_queue.size()
    
    # Existing public methods
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def type(self) -> str:
        return self._type
    
    @property
    def state(self) -> str:
        return self._state
    
    @state.setter
    def state(self, value: str):
        self._state = value
    
    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes.copy()
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self._attributes.get(key, default)
    
    def update_state(self, new_state: str) -> None:
        self._state = new_state
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self._id,
            'type': self._type,
            'state': self._state,
            **self._attributes
        }
    
    def __repr__(self) -> str:
        return f"Agent(id={self._id}, type={self._type}, state={self._state})"


class Agents:
    """
    Factory that creates all agents from configuration.
    """
    
    def __init__(self, agents_config: Dict[str, Any], distributions, automata, metrics_collector):
        """
        Initialize and create all agents from configuration.
        
        Args:
            agents_config: Full agents.json dict with quantity for each agent type
            distributions: Distributions instance for sampling
            automata: Automata instance for event processing
            metrics_collector: MetricsCollector instance for recording results
        """
        self._config = agents_config
        self._distributions = distributions
        self._automata = automata
        self._metrics_collector = metrics_collector
        self._agents: Dict[str, Agent] = {}  # id -> Agent
        self._by_type: Dict[str, List[Agent]] = {}  # type -> List[Agent]
        
        self._create_all_agents()
    
    def _create_all_agents(self) -> None:
        """Create all agents based on configuration."""
        for agent_type, config in self._config.items():
            quantity = config.get('quantity')
            if quantity is None:
                raise ValueError(f"Agent type '{agent_type}' missing required field 'quantity'")
            
            if not isinstance(quantity, int) or quantity < 0:
                raise ValueError(f"Agent type '{agent_type}' quantity must be a non-negative integer")
            
            agents_list = []
            for i in range(quantity):
                agent = Agent(agent_type, config, self._distributions, self._automata, self._metrics_collector, self, i)
                agents_list.append(agent)
                self._agents[agent.id] = agent
            
            self._by_type[agent_type] = agents_list
    
    def get_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_by_type(self, agent_type: str) -> List[Agent]:
        """Get all agents of a specific type."""
        return self._by_type.get(agent_type, [])
    
    def get_all_ids(self) -> List[str]:
        """Get all agent IDs."""
        return list(self._agents.keys())
    
    def get_all_agents(self) -> List[Agent]:
        """Get all agents as a list."""
        return list(self._agents.values())
    
    @property
    def total_count(self) -> int:
        """Total number of agents across all types."""
        return len(self._agents)
    
    def __getitem__(self, agent_id: str) -> Optional[Agent]:
        """Allow indexing by agent ID."""
        return self._agents.get(agent_id)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if agent exists by ID."""
        return agent_id in self._agents
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __repr__(self) -> str:
        counts = {t: len(agents) for t, agents in self._by_type.items()}
        return f"Agents(types={counts}, total={self.total_count})"