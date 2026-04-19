from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random


class Automaton(ABC):
    """Abstract automaton driving agent behavior."""

    @abstractmethod
    def step(self, local_view: Dict[str, Any]) -> str:
        pass


class GenericAutomaton(Automaton):
    """
    Generic automaton driven entirely by JSON configuration.
    No hardcoded automaton names or special cases.
    """

    def __init__(self, config: Dict[str, Any], sampler):
        self.config = config
        self.sampler = sampler
        self.automaton_type = config.get('type', 'deterministic')
        self.params = config.get('params', [])
        self.trigger = config.get('trigger')
        self.output_success = config.get('output_success')
        self.output_failure = config.get('output_failure')
        self.distribution_selectors = config.get('distribution_selector', [])
        
        # For automata with state machines
        self.transitions = config.get('transitions', [])
        self.states = config.get('states', {})
        self.current_state = self.states.get('initial', 'IDLE')
        self.intents_fifo = config.get('intents_fifo', [])

    def step(self, local_view: Dict[str, Any]) -> str:
        """Execute one step of the automaton based on its type."""
        
        # State machine style (has transitions)
        if self.transitions:
            return self._step_state_machine(local_view)
        
        # Probabilistic style (has distribution_selector)
        elif self.distribution_selectors:
            return self._step_probabilistic(local_view)
        
        # Deterministic style
        else:
            return self._step_deterministic(local_view)

    def _step_state_machine(self, local_view: Dict[str, Any]) -> str:
        """Handle state machine style automaton."""
        for transition in self.transitions:
            if transition.get('from') == self.current_state:
                self.current_state = transition.get('to', self.current_state)
                
                rule = transition.get('rule', {})
                if rule.get('type') == 'probabilistic':
                    dist_name = rule.get('distribution')
                    if dist_name:
                        dist_config = self.sampler.get_distribution(dist_name)
                        if dist_config:
                            self.sampler.sample(dist_config)
                
                emit_intent = transition.get('emit_intent', '')
                if emit_intent:
                    return emit_intent
                
                break
        
        return 'IDLE'

    def _step_probabilistic(self, local_view: Dict[str, Any]) -> str:
        """Handle probabilistic style automaton."""
        context = self._build_context(local_view)
        probability = self._get_probability(context)
        
        success = random.random() < probability
        return self.output_success if success else self.output_failure

    def _step_deterministic(self, local_view: Dict[str, Any]) -> str:
        """Handle deterministic style automaton."""
        return self.output_success if self.output_success else 'SUCCESS'

    def _build_context(self, local_view: Dict[str, Any]) -> Dict[str, Any]:
        """Build context from local_view using configured params."""
        context = {}
        for param_name in self.params:
            if param_name in local_view:
                context[param_name] = local_view[param_name]
        return context

    def _get_probability(self, context: Dict[str, Any]) -> float:
        """Get probability from distribution selector based on context."""
        for selector in self.distribution_selectors:
            when = selector.get('when', {})
            match = True
            
            for key, condition in when.items():
                if 'lte' in condition:
                    limit = condition['lte']
                    if context.get(key, 0) > limit:
                        match = False
                        break
            
            if match:
                dist_name = selector.get('distribution')
                if dist_name:
                    dist_config = self.sampler.get_distribution(dist_name)
                    if dist_config:
                        return self.sampler.sample(dist_config)
        
        return 0.0


def create_automaton(automaton_name: str, automata_config: Dict[str, Any], sampler) -> Automaton:
    """
    Factory function to create automaton purely from JSON config.
    No hardcoded automaton names or special cases.
    """
    config = automata_config.get(automaton_name)
    
    if config is None:
        raise ValueError(f"Automaton '{automaton_name}' not found in automata.json")
    
    return GenericAutomaton(config, sampler)