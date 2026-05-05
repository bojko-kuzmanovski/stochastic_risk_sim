from typing import Dict
from collections import defaultdict


class MetricsCollector:
    """
    Collects event, automata, distribution and environment metrics.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

        # Events runtime usage
        self._agent_event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Automata execution counts
        self._automaton_execution_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Distribution runtime usage
        self._distribution_sample_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Environment actions
        self._environment_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Agents actions
        self._agent_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # PATL snapshots
        self._patl_snapshot_counts: Dict[str, int] = defaultdict(int)


    # RUNTIME HOOKS
    def record_agent_event(self, event_category: str, signal: str) -> None:
        if not self.enabled:
            return
        self._agent_event_counts[event_category][signal] += 1


    def record_automaton_execution(self, automaton_name: str, result: str, state: str) -> None:
        if not self.enabled:
            return
        self._automaton_execution_counts[automaton_name][result][state] += 1


    def record_distribution_sample(self, dist_name: str, family: str) -> None:
        if not self.enabled:
            return
        self._distribution_sample_counts[dist_name][family] += 1


    def record_environment_action(self, env_type: str, action: str) -> None:
        if not self.enabled:
            return
        self._environment_action_counts[env_type][action] += 1


    def record_agent_action(self, agent_type: str, action: str) -> None:
        if not self.enabled:
            return
        self._agent_action_counts[agent_type][action] += 1


    def record_patl_sampling(self, automaton_name: str, state: str) -> None:
        if not self.enabled:
            return
        self._patl_snapshot_counts[automaton_name][state] += 1


    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    # REPORT
    def print_report(self, distributions, environments, agents, automata, events, snapshot_manager):
        print("\n" + "=" * 60)
        print("📊 SIMULATION STATISTICS")
        print("=" * 60)

        # BUILDTIME METRICS
        print("\n" + "*" * 60)
        print("🧱 BUILDTIME METRICS")
        print("*" * 60)

        # Distributions buildtime
        family_distributions = {}
        for dist_name in distributions.samplers.keys():
            family = distributions.samplers[dist_name]['family']
            if family not in family_distributions:
                family_distributions[family] = []
            family_distributions[family].append(dist_name)

        print(f"\n📈 Distributions: {len(distributions.samplers)}")
        for family in sorted(family_distributions.keys()):
            count = len(family_distributions[family])
            print(f"   └── {family}: {count}")
            for dist_name in sorted(family_distributions[family]):
                print(f"       └── {dist_name}")

        # Environments buildtime
        env_count = {}
        for e in environments.data:
            env_type = e.get('environment_type')
            env_count[env_type] = env_count.get(env_type, 0) + 1

        print(f"\n🌍 Environments: {len(environments.data)}")
        for env_type in sorted(env_count.keys()):
            print(f"   └── {env_type}: {env_count[env_type]}")

        # Agents buildtime
        agent_count = {}
        for a in agents.data:
            agent_type = a.get('agent_type')
            agent_count[agent_type] = agent_count.get(agent_type, 0) + 1

        print(f"\n👤 Agents: {len(agents.data)}")
        for agent_type in sorted(agent_count.keys()):
            print(f"   └── {agent_type}: {agent_count[agent_type]}")

        # Automata buildtime
        print(f"\n🤖 Automata: {len(automata.data)}")
        for aut in sorted(automata.data, key=lambda x: x.get('automaton_name', '')):
            print(f"   └── {aut.get('automaton_name')}")

        # Declared events buildtime
        static_defined = [e for e in events.data if e.get("event_category") == "static"]
        signal_counts = {}
        signal_agents = {}
        for e in static_defined:
            signal = e["signal"]
            agent_type = e["agent_type"]
            
            if signal not in signal_counts:
                signal_counts[signal] = 0
                signal_agents[signal] = set()
            
            signal_counts[signal] += 1
            signal_agents[signal].add(agent_type)

        print(f"\n🧾 Static Declared Events: {len(static_defined)}")
        for signal in sorted(signal_counts.keys()):
            print(f"   └── {signal}: {signal_counts[signal]}")
            for agent_type in sorted(signal_agents[signal]):
                print(f"       └── {agent_type}")

        # PATL buildtime
        data = snapshot_manager.data
        total_predicates = sum(len(preds) for preds in data.values())
        print(f"\n📸 PATL Predicates: {total_predicates}")

        by_automaton = {}
        for (aut, state), preds in data.items():
            if aut not in by_automaton:
                by_automaton[aut] = {"total_preds": 0, "states": {}}
            by_automaton[aut]["total_preds"] += len(preds)
            by_automaton[aut]["states"][state] = preds

        for aut in sorted(by_automaton.keys()):
            info = by_automaton[aut]
            print(f"   └── {aut}: {info['total_preds']} predicates")
            for state in sorted(info["states"].keys()):
                preds = info["states"][state]
                print(f"       └── {state}: {len(preds)} predicates")
                for pred in preds:
                    pred_id = pred.get("predicate_id", "?")
                    pred_type = pred.get("type", "?")
                    print(f"           └── {pred_id} ({pred_type})")


        # RUNTIME METRICS
        print("\n" + "*" * 60)
        print("🚀 RUNTIME METRICS")
        print("*" * 60)

        # Distributions runtime
        family_usage = {}
        total_all = 0
        for dist_name, fams in self._distribution_sample_counts.items():
            for family, count in fams.items():
                if family not in family_usage:
                    family_usage[family] = {"total": 0, "distributions": []}
                family_usage[family]["total"] += count
                family_usage[family]["distributions"].append((dist_name, count))
                total_all += count

        print(f"\n🎲 Distribution Usage: {total_all}")
        for family in sorted(family_usage.keys()):
            total = family_usage[family]["total"]
            print(f"   └── {family}: {total}")
            for dist_name, count in sorted(family_usage[family]["distributions"]):
                print(f"       └── {dist_name}: {count}")

        # Environments runtime
        total_env = sum(sum(actions.values()) for actions in self._environment_action_counts.values())
        print(f"\n🌍 Environment Actions: {total_env}")
        for env in sorted(self._environment_action_counts.keys()):
            actions = self._environment_action_counts[env]
            total = sum(actions.values())
            print(f"   └── {env}: {total}")
            for act in sorted(actions.keys()):
                print(f"       └── {act}: {actions[act]}")

        # Agent runtime
        total_agent = sum(sum(actions.values()) for actions in self._agent_action_counts.values())
        print(f"\n🤖 Agent Actions: {total_agent}")
        for agent in sorted(self._agent_action_counts.keys()):
            actions = self._agent_action_counts[agent]
            total = sum(actions.values())
            print(f"   └── {agent}: {total}")
            for act in sorted(actions.keys()):
                print(f"       └── {act}: {actions[act]}")

        # Automata runtime
        total_aut = sum(sum(sum(states.values()) for states in result.values()) for result in self._automaton_execution_counts.values())
        print(f"\n🎯 Automata Executions: {total_aut}")
        for aut in sorted(self._automaton_execution_counts.keys()):
            results = self._automaton_execution_counts[aut]
            total = sum(sum(states.values()) for states in results.values())
            print(f"   └── {aut}: {total}")
            for result in sorted(results.keys()):
                states = results[result]
                result_total = sum(states.values())
                print(f"       └── {result}: {result_total}")
                for state in sorted(states.keys()):
                    print(f"           └── {state}: {states[state]}")

        # Events runtime
        total_static = sum(self._agent_event_counts["static"].values())
        total_dynamic = sum(self._agent_event_counts["dynamic"].values())

        print(f"\n📅 Events Generated: {total_static + total_dynamic}")
        print(f"   └── static: {total_static}")
        print(f"   └── dynamic: {total_dynamic}")

        # PATL runtime
        total_patl = sum(
            sum(states.values())
            for states in self._patl_snapshot_counts.values()
        )

        print(f"\n📸 PATL Snapshots Captured: {total_patl}")
        for aut in sorted(self._patl_snapshot_counts.keys()):
            states = self._patl_snapshot_counts[aut]
            total_aut = sum(states.values())
            print(f"   └── {aut}: {total_aut}")
            for state in sorted(states.keys()):
                print(f"       └── {state}: {states[state]}")

        print("\n" + "=" * 60)