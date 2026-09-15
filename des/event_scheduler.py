import heapq
import itertools

from core.utils.evaluator import resolve_value


class EventCalendar:
    """
    Calendario global Q: cola de prioridad sobre (t, e) ordenada por t creciente.
    A igual t se respeta el orden de inserción.
    """

    def __init__(self):
        self._heap = []
        self._seq = itertools.count()

    def push(self, t, event):
        heapq.heappush(self._heap, (t, next(self._seq), event))

    def next_time(self):
        return self._heap[0][0] if self._heap else None

    def pop_instant(self):
        """Extrae todos los eventos con el menor t y devuelve (t, eventos)."""
        t = self._heap[0][0]
        batch = []
        while self._heap and self._heap[0][0] == t:
            batch.append(heapq.heappop(self._heap)[2])
        return t, batch

    def clear(self):
        self._heap.clear()

    def __len__(self):
        return len(self._heap)


class EventScheduler:
    """
    Programa los eventos estáticos. Cada evento estático (signal, tipo, rho) genera un
    evento por agente del tipo, con primera activación en t0 ~ rho, y se reprograma en
    T + dt, dt ~ rho, cada vez que se despacha.
    """

    def __init__(self, static_events, agents, distributions):
        self._events = static_events
        self._agents = agents
        self._distributions = distributions
        self._calendar = None
        self._running = False

    def _sample_periodicity(self, event_def):
        pdef = event_def.get("periodicity_def")
        dt = resolve_value(pdef, self._distributions) if pdef else event_def["periodicity"]
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"periodicity must resolve to a positive number for signal "
                             f"'{event_def['signal']}', got {dt}")
        return float(dt)

    def _push(self, event_def, agent_id, T):
        event = {
            "event_category": "static",
            "signal": event_def["signal"],
            "agent_type": event_def["agent_type"],
            "agent_id": agent_id,
            "periodicity_def": event_def.get("periodicity_def"),
            "periodicity": event_def["periodicity"],
        }
        self._calendar.push(T + self._sample_periodicity(event_def), event)

    def start(self, calendar, T=0.0):
        self._calendar = calendar
        self._running = True
        for event_def in self._events:
            for agent in self._agents.data:
                if agent.get("agent_type") == event_def["agent_type"]:
                    self._push(event_def, agent["agent_id"], T)

    def schedule_agent(self, agent, T):
        if not self._running:
            return
        for event_def in self._events:
            if agent.get("agent_type") == event_def["agent_type"]:
                self._push(event_def, agent["agent_id"], T)

    def reschedule(self, event, T):
        """Paso 3: reinserta el evento estático en T + dt."""
        if not self._running:
            return
        self._calendar.push(T + self._sample_periodicity(event), event)

    def stop(self):
        self._running = False
