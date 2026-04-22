from typing import Dict
from collections import defaultdict


class MetricsCollector:
    """
    Collects event, automata, distribution and environment metrics.
    """

    def __init__(self):

        # Distribution runtime usage
        self._distribution_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Events runtime usage
        self._event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Automata execution counts
        self._automaton_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Environment actions
        self._environment_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


    # RUNTIME HOOKS
    def record_agent_event(self, event_category: str, signal: str) -> None:
        self._event_counts[event_category][signal] += 1

    def record_automaton(self, automaton_name: str, state: str) -> None:
        self._automaton_counts[automaton_name][state] += 1

    def record_distribution(self, dist_name: str, family: str) -> None:
        self._distribution_counts[dist_name][family] += 1

    def record_environment(self, env_type: str, action: str) -> None:
        self._environment_counts[env_type][action] += 1

    # REPORT
    def print_report(self, distributions, environments, agents, automata, events):

        print("\n" + "=" * 60)
        print("📊 SIMULATION STATISTICS")
        print("=" * 60)

        # BUILDTIME METRICS
        print("\n🧱 BUILDTIME METRICS")

        print(f"\n📈 Distributions: {len(distributions.samplers)}")

        print(f"\n🌍 Environments: {len(environments.data)}")
        for e in environments.data:
            print(f"   └── {e.get('environment_type')}")

        print(f"\n👤 Agents: {len(agents.data)}")
        for a in agents.data:
            print(f"   └── {a.get('agent_type')}")

        print(f"\n🤖 Automata: {len(automata.data)}")

        static_defined = [e for e in events.data if e.get("event_category") == "static"]

        print(f"\n🧾 Static Events (defined): {len(static_defined)}")
        for e in sorted(static_defined, key=lambda x: x["signal"]):
            print(f"   └── {e['signal']}")

        # RUNTIME METRICS
        print("\n🚀 RUNTIME METRICS")

        # Events
        total_static = sum(self._event_counts["static"].values())
        total_dynamic = sum(self._event_counts["dynamic"].values())

        print(f"\n📅 Events:")
        print(f"   └── static: {total_static}")
        print(f"   └── dynamic: {total_dynamic}")

        for cat, signals in self._event_counts.items():
            print(f"\n   {cat.upper()}:")
            for sig, cnt in signals.items():
                print(f"   └── {sig}: {cnt}")

        # Distributions runtime
        print("\n🎲 Distribution Usage:")
        for dist, fams in self._distribution_counts.items():
            total = sum(fams.values())
            print(f"   └── {dist}: {total}")
            for fam, cnt in fams.items():
                print(f"       └── {fam}: {cnt}")

        # Environments runtime
        print("\n🌍 Environment Actions:")
        for env, actions in self._environment_counts.items():
            total = sum(actions.values())
            print(f"   └── {env}: {total}")
            for act, cnt in actions.items():
                print(f"       └── {act}: {cnt}")

        # Automata runtime
        print("\n🎯 Automata Executions:")
        for aut, states in self._automaton_counts.items():
            for state, cnt in states.items():
                print(f"   └── {aut} → {state}: {cnt}")

        print("\n" + "=" * 60)