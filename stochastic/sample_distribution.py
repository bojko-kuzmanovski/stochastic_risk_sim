import random
import numpy as np
from typing import Dict, Any, Optional

class SampleDistribution:
    """
    Unified distribution sampler based on distributions.json configuration.
    """
    
    def __init__(self, distributions_config: Optional[Dict] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed if seed is not None else None)
        self.distributions = distributions_config or {}
    
    def set_distributions(self, distributions_config: Dict):
        """Set or update distributions configuration."""
        self.distributions = distributions_config
    
    def get_distribution(self, dist_name: str) -> Optional[Dict]:
        """Get distribution configuration by name."""
        return self.distributions.get(dist_name)
    
    def _get_param(self, params_config: Dict, param_name: str, bound_params: Optional[Dict], default: float) -> float:
        """
        Extract parameter value.
        
        Priority:
        1. bound_params[param_name] (if exists)
        2. params_config[param_name]['value'] (if exists)
        3. default
        """
        if bound_params and param_name in bound_params:
            return bound_params[param_name]
        
        param_config = params_config.get(param_name, {})
        return param_config.get('value', default)
    
    def sample(self, dist_config: Dict[str, Any], bound_params: Optional[Dict[str, float]] = None) -> float:
        """
        Returns a sample from the distribution defined by dist_config.
        """
        family = dist_config.get('family', '')
        params_config = dist_config.get('params', {})
        truncation = dist_config.get('truncation', {})
        
        try:
            if family == 'gamma':
                shape = self._get_param(params_config, 'shape', bound_params, 5.0)
                scale = self._get_param(params_config, 'scale', bound_params, 5.0)
                value = self._np_rng.gamma(shape, scale)
            
            elif family == 'beta':
                alpha = self._get_param(params_config, 'alpha', bound_params, 2.0)
                beta = self._get_param(params_config, 'beta', bound_params, 2.0)
                value = self._np_rng.beta(alpha, beta)
            
            elif family == 'lognormal':
                mean = self._get_param(params_config, 'mean', bound_params, 7.0)
                sigma = self._get_param(params_config, 'sigma', bound_params, 1.0)
                value = self._np_rng.lognormal(mean, sigma)
            
            elif family == 'poisson':
                lam = self._get_param(params_config, 'lambda', bound_params, 50.0)
                # Use Python's random for Poisson to avoid NumPy hanging
                value = np.random.poisson(lam) if lam > 0 else 0
                # Alternative: use self.rng for Poisson (requires custom implementation)
                # value = self._poisson(lam)
            
            elif family == 'exponential':
                rate = self._get_param(params_config, 'rate', bound_params, 1.0)
                value = self._np_rng.exponential(1.0 / rate)
            
            else:
                value = self.rng.random()
            
            # Apply truncation
            if truncation:
                min_val = truncation.get('min', -float('inf'))
                max_val = truncation.get('max', float('inf'))
                value = max(min_val, min(value, max_val))
            
            return float(value)
        
        except Exception as e:
            print(f"Error sampling from {family}: {e}")
            return 0.0
    
    def sample_by_name(self, dist_name: str, bound_params: Optional[Dict[str, float]] = None) -> float:
        """Look up distribution by name and sample it."""
        dist_config = self.distributions.get(dist_name)
        if not dist_config:
            raise ValueError(f"Distribution '{dist_name}' not found")
        return self.sample(dist_config, bound_params)