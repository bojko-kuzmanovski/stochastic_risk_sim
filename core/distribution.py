import random
import numpy as np
from typing import Dict, Any, Optional


class Distribution:
    """
    A single probability distribution with family, parameters, and truncation.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], seed: Optional[int] = None):
        self.name = name
        self.config = config
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed if seed is not None else None)
        
        self.family = self._validate_family(config.get('family'))
        self.params = self._validate_params(config.get('params', {}))
        self.truncation = self._validate_truncation(config.get('truncation'))
    
    def _validate_family(self, family: Any) -> str:
        allowed = ["gamma", "beta", "lognormal", "poisson", "exponential"]
        if not family or family not in allowed:
            raise ValueError(f"Distribution '{self.name}' has invalid family '{family}'. Must be one of {allowed}")
        return family
    
    def _validate_params(self, params: Any) -> Dict[str, float]:
        if not isinstance(params, dict):
            raise ValueError(f"Distribution '{self.name}' params must be an object")
        
        validated = {}
        for key, value in params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Distribution '{self.name}' param '{key}' must be a number")
            validated[key] = float(value)
        
        if len(validated) == 0:
            raise ValueError(f"Distribution '{self.name}' must have at least one parameter")
        
        return validated
    
    def _validate_truncation(self, truncation: Any) -> Dict[str, float]:
        if not truncation:
            raise ValueError(f"Distribution '{self.name}' missing required 'truncation'")
        
        if not isinstance(truncation, dict):
            raise ValueError(f"Distribution '{self.name}' truncation must be an object")
        
        min_val = truncation.get('min')
        max_val = truncation.get('max')
        
        if min_val is None or max_val is None:
            raise ValueError(f"Distribution '{self.name}' truncation must have both 'min' and 'max'")
        
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError(f"Distribution '{self.name}' truncation min and max must be numbers")
        
        if min_val >= max_val:
            raise ValueError(f"Distribution '{self.name}' truncation min must be less than max")
        
        return {'min': float(min_val), 'max': float(max_val)}
    
    def _get_param(self, param_name: str, bound_params: Optional[Dict[str, float]] = None, default: float = 0.0) -> float:
        if bound_params and param_name in bound_params:
            return bound_params[param_name]
        return self.params.get(param_name, default)
    
    def sample(self, bound_params: Optional[Dict[str, float]] = None) -> float:
        try:
            if self.family == 'gamma':
                shape = self._get_param('shape', bound_params, 5.0)
                scale = self._get_param('scale', bound_params, 5.0)
                value = self._np_rng.gamma(shape, scale)
            
            elif self.family == 'beta':
                alpha = self._get_param('alpha', bound_params, 2.0)
                beta = self._get_param('beta', bound_params, 2.0)
                value = self._np_rng.beta(alpha, beta)
            
            elif self.family == 'lognormal':
                mean = self._get_param('mean', bound_params, 7.0)
                sigma = self._get_param('sigma', bound_params, 1.0)
                value = self._np_rng.lognormal(mean, sigma)
            
            elif self.family == 'poisson':
                lam = self._get_param('lambda', bound_params, 50.0)
                value = self._np_rng.poisson(lam) if lam > 0 else 0
            
            elif self.family == 'exponential':
                rate = self._get_param('rate', bound_params, 1.0)
                value = self._np_rng.exponential(1.0 / rate) if rate > 0 else 0
            
            else:
                value = self.rng.random()
            
            min_val = self.truncation['min']
            max_val = self.truncation['max']
            value = max(min_val, min(value, max_val))
            
            return float(value)
        
        except Exception as e:
            print(f"Error sampling from {self.family}: {e}")
            return self.truncation['min']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'family': self.family,
            'params': self.params,
            'truncation': self.truncation
        }
    
    def __repr__(self) -> str:
        return f"Distribution(name={self.name}, family={self.family}, params={self.params})"


class Distributions:
    """
    Factory/registry for creating and managing multiple distributions.
    """
    
    def __init__(self, distributions_config: Dict[str, Any], seed: Optional[int] = None):
        self._distributions: Dict[str, Distribution] = {}
        self.seed = seed
        
        for name, config in distributions_config.items():
            self._distributions[name] = Distribution(name, config, seed)
    
    def get(self, name: str) -> Distribution:
        if name not in self._distributions:
            raise ValueError(f"Distribution '{name}' not found")
        return self._distributions[name]
    
    def sample(self, name: str, bound_params: Optional[Dict[str, float]] = None) -> float:
        return self.get(name).sample(bound_params)
    
    def __contains__(self, name: str) -> bool:
        return name in self._distributions
    
    def __iter__(self):
        return iter(self._distributions.items())
    
    def __len__(self) -> int:
        return len(self._distributions)
    
    def __repr__(self) -> str:
        return f"Distributions(keys={list(self._distributions.keys())})"