from typing import Dict
from collections import defaultdict


class MetricsCollector:
    """
    Collects event, automata, distribution and environment metrics.
    """

    def __init__(self):

        # Events runtime usage
        self._agent_event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Automata execution counts
        self._automaton_execution_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Distribution runtime usage
        self._distribution_sample_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Environment actions
        self._environment_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Agents actions
        self._agent_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


    # RUNTIME HOOKS
    def record_agent_event(self, event_category: str, signal: str) -> None:
        self._agent_event_counts[event_category][signal] += 1

    def record_automaton_execution(self, automaton_name: str, state: str) -> None:
        self._automaton_execution_counts[automaton_name][state] += 1

    def record_distribution_sample(self, dist_name: str, family: str) -> None:
        self._distribution_sample_counts[dist_name][family] += 1

    def record_environment_action(self, env_type: str, action: str) -> None:
        self._environment_action_counts[env_type][action] += 1

    def record_agent_action(self, agent_type: str, action: str) -> None:
        self._agent_action_counts[agent_type][action] += 1

    # REPORT
    def print_report(self, distributions, environments, agents, automata, events):
        print("\n" + "=" * 60)
        print("📊 SIMULATION STATISTICS")
        print("=" * 60)

        # BUILDTIME METRICS
        print("\n" + "*" * 60)
        print("🧱 BUILDTIME METRICS")
        print("*" * 60)

        print(f"\n📈 Distributions: {len(distributions.samplers)}")
        for dist_name in sorted(distributions.samplers.keys()):
            print(f"   └── {dist_name}")

        env_count = {}
        for e in environments.data:
            env_type = e.get('environment_type')
            env_count[env_type] = env_count.get(env_type, 0) + 1

        print(f"\n🌍 Environments: {len(environments.data)}")
        for env_type in sorted(env_count.keys()):
            print(f"   └── {env_type}: {env_count[env_type]}")

        agent_count = {}
        for a in agents.data:
            agent_type = a.get('agent_type')
            agent_count[agent_type] = agent_count.get(agent_type, 0) + 1

        print(f"\n👤 Agents: {len(agents.data)}")
        for agent_type in sorted(agent_count.keys()):
            print(f"   └── {agent_type}: {agent_count[agent_type]}")

        print(f"\n🤖 Automata: {len(automata.data)}")
        for aut in sorted(automata.data, key=lambda x: x.get('automaton_name', '')):
            print(f"   └── {aut.get('automaton_name')}")

        static_defined = [e for e in events.data if e.get("event_category") == "static"]
        print(f"\n🧾 Static Events: {len(static_defined)}")
        for e in sorted(static_defined, key=lambda x: x["signal"]):
            print(f"   └── {e['signal']}")

        # RUNTIME METRICS
        print("\n" + "*" * 60)
        print("🚀 RUNTIME METRICS")
        print("*" * 60)

        # Distributions runtime
        print("\n🎲 Distribution Usage:")
        for dist in sorted(self._distribution_sample_counts.keys()):
            fams = self._distribution_sample_counts[dist]
            total = sum(fams.values())
            print(f"   └── {dist}: {total}")
            for fam in sorted(fams.keys()):
                print(f"       └── {fam}: {fams[fam]}")

        # Environments runtime
        print("\n🌍 Environment Actions:")
        for env in sorted(self._environment_action_counts.keys()):
            actions = self._environment_action_counts[env]
            total = sum(actions.values())
            print(f"   └── {env}: {total}")
            for act in sorted(actions.keys()):
                print(f"       └── {act}: {actions[act]}")

        # Agent runtime
        print("\n🤖 Agent Actions:")
        for agent in sorted(self._agent_action_counts.keys()):
            actions = self._agent_action_counts[agent]
            total = sum(actions.values())
            print(f"   └── {agent}: {total}")
            for act in sorted(actions.keys()):
                print(f"       └── {act}: {actions[act]}")

        # Automata runtime
        print("\n🎯 Automata Executions:")
        for aut in sorted(self._automaton_execution_counts.keys()):
            states = self._automaton_execution_counts[aut]
            for state in sorted(states.keys()):
                print(f"   └── {aut} → {state}: {states[state]}")
        
        # Events runtime
        total_static = sum(self._agent_event_counts["static"].values())
        total_dynamic = sum(self._agent_event_counts["dynamic"].values())

        print(f"\n📅 Events Executions:")
        print(f"   └── static: {total_static}")
        print(f"   └── dynamic: {total_dynamic}")

        print("\n" + "=" * 60)