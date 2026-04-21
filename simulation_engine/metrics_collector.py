from typing import Dict, Any
from collections import defaultdict


class MetricsCollector:
    """
    Collects and computes all simulation metrics and prints unified report.
    """
    
    def __init__(self):
        self._automaton_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._initial_stats = {}
        self._final_stats = {}
    
    def collect_initial_stats(self, distributions, environments, agents, automata, event_scheduler):
        """Collect all initial configuration stats."""
        # Distributions by family
        dist_by_family = {}
        for name, info in distributions.samplers.items():
            fam = info.get('family', 'unknown')
            dist_by_family[fam] = dist_by_family.get(fam, 0) + 1
        
        # Environments by type
        env_by_type = {}
        for e in environments.data:
            t = e.get('environment_type', 'unknown')
            env_by_type[t] = env_by_type.get(t, 0) + 1
        
        # Agents by type
        agent_by_type = {}
        for a in agents.data:
            t = a.get('agent_type', 'unknown')
            agent_by_type[t] = agent_by_type.get(t, 0) + 1
        
        # Automata transitions
        det_trans = prob_trans = 0
        for aut in automata.data:
            for t in aut.get('transitions', []):
                if t.get('type') == 'deterministic':
                    det_trans += 1
                elif t.get('type') == 'probabilistic':
                    prob_trans += 1
        
        self._initial_stats = {
            'total_distributions': len(distributions.samplers),
            'dist_by_family': dist_by_family,
            'total_environments': len(environments.data),
            'env_by_type': env_by_type,
            'total_agents': len(agents.data),
            'agent_by_type': agent_by_type,
            'total_automata': len(automata.data),
            'det_transitions': det_trans,
            'prob_transitions': prob_trans,
            'static_events': event_scheduler.get_static_events_count()
        }
    
    def set_final_stats(self, events_processed: int, max_time: float):
        """Collect final simulation stats."""
        self._final_stats = {
            'dynamic_events': events_processed,
            'max_time': max_time
        }
    
    def record_automaton_result(self, automaton_name: str, result: str) -> None:
        """Record the result of an automaton execution."""
        self._automaton_counts[automaton_name][result] += 1
    
    def print_report(self) -> None:
        """Print unified simulation report."""
        stats = self._initial_stats
        final = self._final_stats
        
        total_events = final.get('dynamic_events', 0) + stats.get('static_events', 0)
        
        print("\n" + "="*60)
        print("📊 SIMULATION REPORT STATS")
        print("="*60)
        
        # Distributions
        print(f"\n📈 Distributions: {stats.get('total_distributions', 0)}")
        for fam, cnt in sorted(stats.get('dist_by_family', {}).items()):
            print(f"   └── {fam}: {cnt}")
        
        # Environments
        print(f"\n🌍 Environments: {stats.get('total_environments', 0)}")
        for t, cnt in sorted(stats.get('env_by_type', {}).items()):
            print(f"   └── {t}: {cnt}")
        
        # Agents
        print(f"\n👤 Agents: {stats.get('total_agents', 0)}")
        for t, cnt in sorted(stats.get('agent_by_type', {}).items()):
            print(f"   └── {t}: {cnt}")
        
        # Automata transitions
        print(f"\n🤖 Automata: {stats.get('total_automata', 0)}")
        print(f"   └── deterministic transitions: {stats.get('det_transitions', 0)}")
        print(f"   └── probabilistic transitions: {stats.get('prob_transitions', 0)}")
        
        # Events
        print(f"\n📅 Total Events: {total_events}")
        print(f"   └── static: {stats.get('static_events', 0)}")
        print(f"   └── dynamic: {final.get('dynamic_events', 0)}")
        
        # Automata executions
        if self._automaton_counts:
            print(f"\n🎯 Automata Executions:")
            for aut_name in sorted(self._automaton_counts.keys()):
                for state, count in sorted(self._automaton_counts[aut_name].items()):
                    print(f"   └── {aut_name} → {state}: {count}")
        
        print("\n" + "="*60)