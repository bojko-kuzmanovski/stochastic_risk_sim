from typing import Dict, Any, List, Optional
from collections import defaultdict


class MetricsCollector:
    """
    Collects and computes metrics based on metrics.json configuration.
    """
    
    def __init__(self, metrics_config: Dict[str, Any]):
        """
        Initialize metrics collector from configuration.
        
        Args:
            metrics_config: Full metrics.json dict where keys are metric names
                           and values contain 'automaton' and 'terminal_state'
        """
        self._config = metrics_config
        
        # Storage for automaton terminal state counts
        self._automaton_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def record_automaton_result(self, automaton_name: str, result: str) -> None:
        """
        Record the result of an automaton execution.
        
        Args:
            automaton_name: Name of the automaton that executed
            result: Output signal (success or failure)
        """
        self._automaton_counts[automaton_name][result] += 1
    
    def compute_metrics(self, agents, environments, transactions) -> Dict[str, Any]:
        """
        Compute all metrics based on configuration.
        
        Args:
            agents: Agents instance
            environments: Environments instance
            transactions: List of transaction records
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        # Global metrics (always computed, not from config)
        results['total_citizens'] = len(agents.get_by_type('citizen'))
        results['total_malicious'] = len(agents.get_by_type('malicious'))
        results['total_environments'] = environments.total_count
        
        # Environment counts by type
        for env_type, env_list in environments.get_all_by_type().items():
            results[f'total_{env_type}_environments'] = len(env_list)
        
        # Agent counts by type
        for agent_type in ['citizen', 'malicious']:
            results[f'total_{agent_type}'] = len(agents.get_by_type(agent_type))
        
        # Active mules (specific to citizen)
        citizens = agents.get_by_type('citizen')
        active_citizens = [c for c in citizens if c.state == 'active']
        results['active_mules'] = len(active_citizens)
        results['activation_rate'] = len(active_citizens) / len(citizens) if citizens else 0
        results['total_transactions'] = len(transactions)
        
        # Automaton metrics from config
        for metric_name, metric_config in self._config.items():
            automaton_name = metric_config.get('automaton')
            terminal_state = metric_config.get('terminal_state')
            
            count = self._automaton_counts[automaton_name].get(terminal_state, 0)
            results[metric_name] = count
        
        return results
    
    def print_report(self, metrics: Dict[str, Any]) -> None:
        """Print formatted metrics report."""
        print("\n" + "="*60)
        print("📊 SIMULATION METRICS REPORT")
        print("="*60)
        
        # Agent counts
        print("\n👥 AGENTS:")
        print(f"   Total citizens: {metrics.get('total_citizens', 0)}")
        print(f"   Total malicious: {metrics.get('total_malicious', 0)}")
        print(f"   Active mules: {metrics.get('active_mules', 0)}")
        print(f"   Activation rate: {metrics.get('activation_rate', 0):.2%}")
        
        # Environment counts
        print("\n🌍 ENVIRONMENTS:")
        print(f"   Total environments: {metrics.get('total_environments', 0)}")
        for key, value in metrics.items():
            if key.endswith('_environments') and key != 'total_environments':
                env_type = key.replace('total_', '').replace('_environments', '')
                print(f"   {env_type}: {value}")
        
        # Transactions
        print("\n💰 TRANSACTIONS:")
        print(f"   Total: {metrics.get('total_transactions', 0)}")
        
        # Automaton metrics
        print("\n🤖 AUTOMATON METRICS:")
        for metric_name in self._config.keys():
            value = metrics.get(metric_name, 0)
            print(f"   {metric_name}: {value}")
        
        print("\n" + "="*60)
    
    def __repr__(self) -> str:
        return f"MetricsCollector(metrics={list(self._config.keys())})"