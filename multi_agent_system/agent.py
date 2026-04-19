from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random
import numpy as np

# Import from automaton module
from multi_agent_system.automaton import create_automaton

class Agent(ABC):
    """Base abstraction for all agents."""

    def __init__(self, agent_id: str, agent_type: str, automaton, attributes: Optional[Dict] = None):
        self.id = agent_id
        self.type = agent_type
        self.automaton = automaton
        self.attributes = attributes or {}
        self.state = 'idle'

    def step(self, local_view: Dict[str, Any]) -> Any:
        """Execute automaton step with local view."""
        if self.automaton is None:
            raise ValueError(f"Agent {self.id} has no automaton assigned.")
        return self.automaton.step(local_view)

    def perceive(self, world_state: Any) -> Dict[str, Any]:
        """Project global state into local view."""
        # Combine agent's own attributes with world state
        local_view = dict(self.attributes)
        if isinstance(world_state, dict):
            local_view.update(world_state)
        return local_view

    @abstractmethod
    def propose(self, intention: Any, world_state: Any) -> Dict[str, Any]:
        """Convert automaton output into structured intention."""
        pass

    def run(self, world_state: Any) -> Dict[str, Any]:
        """Full agent lifecycle for one simulation tick."""
        intention_raw = self.step(world_state)
        return self.propose(intention_raw, world_state)

    def update_state(self, new_state: str):
        """Update agent's internal state."""
        self.state = new_state

    def get_attribute(self, key: str, default=None):
        """Get agent attribute by key."""
        return self.attributes.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for statistics."""
        return {
            'id': self.id,
            'type': self.type,
            'state': self.state,
            **self.attributes
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, type={self.type})"

class GenericAgent(Agent):
    """
    Generic agent that can represent any agent type from agents.json.
    No hardcoded citizen or malicious classes.
    """

    def __init__(self, agent_id: str, agent_type: str, automaton, attributes: Dict):
        super().__init__(agent_id, agent_type, automaton, attributes)

    def propose(self, intention: Any, world_state: Any) -> Dict[str, Any]:
        """Generic intention proposal based on automaton output."""
        return {
            'event_type': intention,
            'agent_id': self.id,
            'agent_type': self.type,
            'payload': self.attributes.copy(),
            'context': {'state': self.state}
        }

def evaluate_formula(formula: str, context: Dict[str, Any]) -> float:
    """
    Safely evaluate a formula string with given context.
    Supports arithmetic operations and variables from context.
    """
    # Allowed functions and variables
    allowed = {"__builtins__": {}}
    return float(eval(formula, allowed, context))

def generate_agent_attributes(agent_config: Dict[str, Any], sampler, 
                              agent_type: str) -> Dict[str, Any]:
    """
    Generate attributes for an agent based on its JSON configuration.
    """
    attributes = {}
    params_config = agent_config.get('params', {})
    
    for param_name, param_spec in params_config.items():
        param_type = param_spec.get('type', 'deterministic')
        
        if param_type == 'probabilistic':
            distribution_name = param_spec.get('distribution')
            
            if distribution_name:
                dist_config = sampler.get_distribution(distribution_name)
                
                refs = param_spec.get('refs', {})
                bound_params = {}
                
                for ref_name, formula in refs.items():
                    bound_params[ref_name] = evaluate_formula(formula, attributes)
                
                if bound_params:
                    value = sampler.sample(dist_config, bound_params)
                else:
                    value = sampler.sample(dist_config)
                
                # Apply transform if specified (generic, not hardcoded)
                transform = param_spec.get('transform')
                if transform == 'binary_threshold':
                    threshold = param_spec.get('threshold', 0.5)
                    value = 1 if value > threshold else 0
                # Add other transforms here as needed, but keep generic
                
                attributes[param_name] = value
            else:
                attributes[param_name] = 0.0
        else:
            attributes[param_name] = param_spec.get('value', 0.0)
    
    return attributes

def create_agent(agent_type: str, agent_id: str, agents_config: Dict[str, Any],
                 automata_config: Dict[str, Any], sampler) -> GenericAgent:
    """
    Factory function to create agent purely from JSON config.
    """
    agent_config = agents_config.get(agent_type)
    
    if agent_config is None:
        raise ValueError(f"Agent type '{agent_type}' not found in agents.json")
    
    # Get automaton names from agent config
    automata_names = agent_config.get('automa', [])
    if not automata_names:
        raise ValueError(f"No automaton specified for agent type '{agent_type}'")
    
    # For citizens, we need to handle multiple automata
    # For now, just take the first one (contact acceptance)
    # The recruitment transition will be handled separately
    automaton_name = automata_names[0]
    automaton = create_automaton(automaton_name, automata_config, sampler)
    
    # Generate attributes
    attributes = generate_agent_attributes(agent_config, sampler, agent_type)
    
    # Store all automaton names for later use
    attributes['automata_names'] = automata_names
    
    return GenericAgent(agent_id, agent_type, automaton, attributes)