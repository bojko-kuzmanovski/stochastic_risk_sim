from typing import Dict
from collections import defaultdict


class MetricsCollector:
    """
    Collects event and automata metrics and prints unified report.
    """

    def __init__(self):
        # Distribution samples count by distribution_name and family
        self._distribution_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Event count by event_category and signal
        self._event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Automaton execution count by automaton_name and final state
        self._automaton_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Environments executed actions count by environment_type and action
        self._environment_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # --- hooks that Agents / Automata will use ---
    def record_event(self, event_category: str, signal: str) -> None:
        if event_category not in self._event_counts:
            event_category = "dynamic"
        self._event_counts[event_category][signal] += 1

    def record_automaton(self, automaton_name: str, state: str) -> None:
        self._automaton_counts[automaton_name][state] += 1

    # --- report ---
    def print_report(self, distributions, environments, agents, automata, events):
        """Print unified simulation report."""

        # Distributions by family
        dist_by_family = {}
        for _, info in distributions.samplers.items():
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
        det_trans = 0
        prob_trans = 0

        for aut in automata.data:
            for t in aut.get("transitions", []):
                if t.get("type") == "deterministic":
                    det_trans += 1
                elif t.get("type") == "probabilistic":
                    prob_trans += 1

        # Declared events
        static_defined = [
            e for e in events.data
            if e.get("event_category") == "static"
        ]

        # Runtime totals
        total_static = sum(self._event_counts["static"].values())
        total_dynamic = sum(self._event_counts["dynamic"].values())
        total_events = total_static + total_dynamic

        print("\n" + "="*60)
        print("📊 SIMULATION STATISTICS")
        print("="*60)

        print("\n🧱 BUILD TIME METRICS")

        # Distributions
        print(f"\n📈 Distributions: {len(distributions.samplers)}")
        for fam, cnt in sorted(dist_by_family.items()):
            print(f"   └── {fam}: {cnt}")

        # Environments
        print(f"\n🌍 Environments: {len(environments.data)}")
        for t, cnt in sorted(env_by_type.items()):
            print(f"   └── {t}: {cnt}")

        # Agents
        print(f"\n👤 Agents: {len(agents.data)}")
        for t, cnt in sorted(agent_by_type.items()):
            print(f"   └── {t}: {cnt}")

        # Automata structure
        print(f"\n🤖 Automata: {len(automata.data)}")
        print(f"   └── deterministic transitions: {det_trans}")
        print(f"   └── probabilistic transitions: {prob_trans}")

        # Declared events
        print(f"\n🧾 Static Events (defined): {len(static_defined)}")
        for e in sorted(static_defined, key=lambda x: x["signal"]):
            print(f"   └── {e['signal']} (periodicity={e.get('periodicity')})")

        print("\n🚀 RUNTIME METRICS")

        # Runtime events
        print(f"\n📅 Total Events: {total_events}")
        print(f"   └── static: {total_static}")
        print(f"   └── dynamic: {total_dynamic}")

        # Distribution usage
        if self._distribution_counts:
            print("\n🎲 Distribution Usage:")
            for dist_name in sorted(self._distribution_counts.keys()):
                total = sum(self._distribution_counts[dist_name].values())
                print(f"   └── {dist_name}: {total}")

                for fam, cnt in sorted(self._distribution_counts[dist_name].items()):
                    print(f"       └── {fam}: {cnt}")
        
        # Environment actions
        if self._environment_counts:
            print("\n🌍 Environment Actions:")
            for env_type in sorted(self._environment_counts.keys()):
                total = sum(self._environment_counts[env_type].values())
                print(f"   └── {env_type}: {total}")

                for action, cnt in sorted(self._environment_counts[env_type].items()):
                    print(f"       └── {action}: {cnt}")

        # Breakdown static
        if self._event_counts["static"]:
            print("\n   Static by signal:")
            for sig, cnt in sorted(self._event_counts["static"].items()):
                print(f"   └── {sig}: {cnt}")

        # Breakdown dynamic
        if self._event_counts["dynamic"]:
            print("\n   Dynamic by signal:")
            for sig, cnt in sorted(self._event_counts["dynamic"].items()):
                print(f"   └── {sig}: {cnt}")

        # Automata executions
        if self._automaton_counts:
            print("\n🎯 Automata Executions:")
            for aut_name in sorted(self._automaton_counts.keys()):
                for state, count in sorted(self._automaton_counts[aut_name].items()):
                    print(f"   └── {aut_name} → {state}: {count}")

        print("\n" + "=" * 60)