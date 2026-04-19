from typing import Dict, Any, Optional, List, Callable
import random
import math


class Automaton:
    """
    Single automaton that defines behavior for a specific signal.
    Internal class - should not be used directly from outside.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], distributions):
        """
        Initialize an automaton from configuration.
        
        Args:
            name: Automaton name (e.g., 'FRIEND_REQUEST', 'MESSAGE')
            config: Automaton configuration with keys:
                - type: 'probabilistic' or 'deterministic'
                - params: list of parameter names expected in context
                - distribution: distribution name for probabilistic automata
                - refs: parameter overrides for distribution
                - output_success: signal to emit on success
                - output_failure: signal to emit on failure
                - transitions: list of state machine transitions
                - states: initial and final states
            distributions: Distributions instance for sampling
        """
        self._name = name
        self._config = config
        self._distributions = distributions
        self._type = config.get('type', 'deterministic')
        self._params = config.get('params', [])
        self._output_success = config.get('output_success')
        self._output_failure = config.get('output_failure')
        self._distribution = config.get('distribution')
        self._refs = config.get('refs', {})
        
        # State machine attributes
        self._transitions = config.get('transitions', [])
        self._states = config.get('states', {})
        self._current_state = self._states.get('initial', 'IDLE')
        self._final_states = set(self._states.get('final', []))
        self._intents_fifo = config.get('intents_fifo', [])
        
        # Validate configuration
        self._validate()
    
    def _validate(self) -> None:
        """Validate automaton configuration."""
        if self._type not in ['probabilistic', 'deterministic']:
            raise ValueError(f"Automaton '{self._name}' has invalid type '{self._type}'")
        
        if self._type == 'probabilistic':
            if not self._distribution:
                raise ValueError(f"Automaton '{self._name}' is probabilistic but missing 'distribution'")
            if self._distribution not in self._distributions:
                raise ValueError(f"Distribution '{self._distribution}' not found for automaton '{self._name}'")
    
    def _extract_context(self, event_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract required parameters from event context."""
        context = {}
        for param_name in self._params:
            if param_name in event_context:
                context[param_name] = event_context[param_name]
            elif param_name in event_context.get('payload', {}):
                context[param_name] = event_context['payload'][param_name]
        return context
    
    def _evaluate_formula(self, formula: str, context: Dict[str, Any]) -> float:
        """Safely evaluate a formula string with given context."""
        allowed = {"__builtins__": {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}}
        return float(eval(formula, allowed, context))
    
    def _get_probability(self, context: Dict[str, Any]) -> float:
        """Calculate probability using distribution and refs."""
        if not self._distribution:
            return 0.0
        
        # Calculate bound parameters from refs
        bound_params = {}
        for param_name, formula in self._refs.items():
            bound_params[param_name] = self._evaluate_formula(formula, context)
        
        # Sample from distribution
        if bound_params:
            return self._distributions.sample(self._distribution, bound_params)
        else:
            return self._distributions.sample(self._distribution)
    
    def _step_state_machine(self, context: Dict[str, Any]) -> Optional[str]:
        """Execute state machine style automaton."""
        for transition in self._transitions:
            if transition.get('from') == self._current_state:
                self._current_state = transition.get('to', self._current_state)
                
                rule = transition.get('rule', {})
                if rule.get('type') == 'probabilistic':
                    dist_name = rule.get('distribution')
                    if dist_name:
                        # Evaluate inputs for the rule
                        inputs = rule.get('inputs', {})
                        bound_params = {}
                        for param_name, formula in inputs.items():
                            bound_params[param_name] = self._evaluate_formula(formula, context)
                        
                        if bound_params:
                            self._distributions.sample(dist_name, bound_params)
                        else:
                            self._distributions.sample(dist_name)
                
                emit_intent = transition.get('emit_intent', '')
                if emit_intent:
                    return emit_intent
                
                break
        
        return None
    
    def _step_probabilistic(self, context: Dict[str, Any]) -> Optional[str]:
        """Execute probabilistic style automaton."""
        probability = self._get_probability(context)
        success = random.random() < probability
        
        if success:
            return self._output_success
        else:
            return self._output_failure
    
    def _step_deterministic(self, context: Dict[str, Any]) -> Optional[str]:
        """Execute deterministic style automaton."""
        return self._output_success
    
    def step(self, event_context: Dict[str, Any]) -> Optional[str]:
        """
        Execute one step of the automaton.
        
        Args:
            event_context: Context from the event being processed
        
        Returns:
            Signal to emit, or None if no emission
        """
        context = self._extract_context(event_context)
        
        # State machine style
        if self._transitions:
            return self._step_state_machine(context)
        
        # Probabilistic style
        elif self._type == 'probabilistic':
            return self._step_probabilistic(context)
        
        # Deterministic style
        else:
            return self._step_deterministic(context)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def current_state(self) -> str:
        return self._current_state
    
    def reset(self) -> None:
        """Reset automaton to initial state."""
        self._current_state = self._states.get('initial', 'IDLE')
    
    def is_final(self) -> bool:
        """Check if automaton is in a final state."""
        return self._current_state in self._final_states
    
    def __repr__(self) -> str:
        return f"Automaton(name={self._name}, type={self._type}, state={self._current_state})"


class Automata:
    """
    Factory that creates all automata from configuration.
    """
    
    def __init__(self, automata_config: Dict[str, Any], distributions):
        """
        Initialize and create all automata from configuration.
        
        Args:
            automata_config: Full automata.json dict
            distributions: Distributions instance for sampling
        """
        self._config = automata_config
        self._distributions = distributions
        self._automata: Dict[str, Automaton] = {}
        
        self._create_all_automata()
    
    def _create_all_automata(self):
        """Create all automata based on configuration."""
        for automaton_name, config in self._config.items():
            self._automata[automaton_name] = Automaton(automaton_name, config, self._distributions)
    
    def get(self, name: str) -> Optional[Automaton]:
        """Get an automaton by name."""
        return self._automata.get(name)
    
    def step(self, name: str, event_context: Dict[str, Any]) -> Optional[str]:
        """Execute an automaton step by name."""
        automaton = self.get(name)
        if not automaton:
            raise ValueError(f"Automaton '{name}' not found")
        return automaton.step(event_context)
    
    def get_all_names(self) -> List[str]:
        """Get all automaton names."""
        return list(self._automata.keys())
    
    def reset_all(self) -> None:
        """Reset all automata to initial states."""
        for automaton in self._automata.values():
            automaton.reset()
    
    def __contains__(self, name: str) -> bool:
        return name in self._automata
    
    def __getitem__(self, name: str) -> Optional[Automaton]:
        return self._automata.get(name)
    
    def __len__(self) -> int:
        return len(self._automata)
    
    def __repr__(self) -> str:
        return f"Automata(automata={list(self._automata.keys())})"