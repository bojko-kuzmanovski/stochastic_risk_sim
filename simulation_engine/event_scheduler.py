import time
import threading


class EventScheduler:
    def __init__(self, static_events, agents):
        self._events = static_events
        self._agents = agents

        self._running = False
        self._threads = []

        # event state
        self._event_state = {}

        for event in self._events:
            signal = event["signal"]
            interval = event["periodicity"]  # ya es número

            self._event_state[signal] = {
                "interval": interval,
                "last_execution": 0.0,
                "in_progress": False,
                "agent_type": event.get("agent_type", "*")
            }

    def start(self):
        self._running = True

        def run_event_loop(signal, state):
            while self._running:
                now = time.time()

                if (now - state["last_execution"]) >= state["interval"]:
                    if not state["in_progress"]:
                        state["in_progress"] = True

                        # select agents
                        target_agents = self._agents.get_by_type(state["agent_type"])

                        # emit event to all agents
                        for agent in target_agents:
                            self._agents.receive_event(
                                agent["agent_id"],
                                {"signal": signal}
                            )

                        # mark as completed
                        state["last_execution"] = now
                        state["in_progress"] = False

                time.sleep(2)

        # create one thread per event
        for signal, state in self._event_state.items():
            t = threading.Thread(
                target=run_event_loop,
                args=(signal, state),
                daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False

        for t in self._threads:
            t.join()