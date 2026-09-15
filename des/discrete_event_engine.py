#!/usr/bin/env python3
import sys

from des.event_scheduler import EventCalendar

# Cota de eventos despachados en un mismo instante; superarla indica un ciclo de
# eventos dinámicos con latencia cero (comportamiento de Zenón).
MAX_EVENTS_PER_INSTANT = 1_000_000


class DiscreteEventSimulator:
    """
    Motor DES con calendario global Q y reloj simulado T.

    En cada iteración T salta al menor tiempo de Q, se despachan a sus colas Q_i todos los
    eventos de ese instante (Paso 1), cada agente con eventos pendientes procesa hasta K de
    ellos, una sesión atómica por evento (Paso 2), y los eventos estáticos se reprograman
    (Paso 3). Los eventos dinámicos emitidos durante una sesión entran a Q en T + latencia.
    La simulación termina cuando Q queda vacío o el siguiente evento excede T_max; después
    se vacían las colas pendientes sin despachar eventos nuevos.
    """

    def __init__(self, distributions, environments, automata, agents, events,
                 event_scheduler, snapshot_manager, queue_batch=5, event_latency=0.0):
        self.distributions = distributions
        self.environments = environments
        self.automata = automata
        self.agents = agents
        self.events = events
        self.event_scheduler = event_scheduler
        self.snapshot_manager = snapshot_manager
        self.queue_batch = queue_batch
        self.event_latency = event_latency
        self.calendar = EventCalendar()
        self.T = 0.0
        self._dispatching = False

    def _emit(self, event):
        if self._dispatching:
            self.calendar.push(self.T + self.event_latency, event)

    def _process_ready(self, ready):
        for agent_id in list(ready):
            self.agents.process_queue(agent_id, self.queue_batch)
            if self.agents.pending(agent_id) == 0:
                del ready[agent_id]

    def run_simulation(self, max_time: float = 60.0):
        self.automata.set_event_sink(self._emit)
        self.agents.set_on_agent_added(lambda agent: self.event_scheduler.schedule_agent(agent, self.T))

        # Inicialización (T = 0)
        self.agents.start()
        self.event_scheduler.start(self.calendar, 0.0)
        self._dispatching = True

        ready = {}
        same_instant = 0
        last_T = None

        # Ejecución
        while self.calendar and self.calendar.next_time() <= max_time:
            T, batch = self.calendar.pop_instant()
            self.T = T

            same_instant = same_instant + len(batch) if T == last_T else len(batch)
            last_T = T
            if same_instant > MAX_EVENTS_PER_INSTANT:
                print(f"[FATAL] DES: more than {MAX_EVENTS_PER_INSTANT} events dispatched at T={T}.",
                      file=sys.stderr, flush=True)
                sys.exit(1)

            for event in batch:
                agent_id = event.get("agent_id")
                if self.agents.receive_event(agent_id, event):
                    ready[agent_id] = None
                    if event.get("event_category") == "static":
                        self.event_scheduler.reschedule(event, T)

            self._process_ready(ready)

        # Finalización: cesa el despacho y se vacían las colas.
        self._dispatching = False
        self.event_scheduler.stop()
        while ready:
            self._process_ready(ready)
        self.agents.stop()
        self.calendar.clear()
        return self.T
