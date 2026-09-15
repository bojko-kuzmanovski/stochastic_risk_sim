#!/usr/bin/env python3
import sys

from des.event_scheduler import EventCalendar
from core.trace import tracer

# Cota de eventos despachados en un mismo instante; superarla indica un ciclo de
# eventos dinámicos con latencia cero (comportamiento de Zenón).
MAX_EVENTS_PER_INSTANT = 1_000_000


class DiscreteEventSimulator:
    """
    Motor DES con calendario global Q, reloj simulado T y cota K de eventos procesados por agente
    en cada instante.

    En cada iteración T salta al menor tiempo de Q y el contador de procesados de cada agente vuelve a 0:

    1. Rezago: cada agente con eventos pendientes de instantes anteriores procesa hasta K de ellos,
       en el orden en que su cola dejó de estar vacía.
    2. Despacho en orden de calendario: los eventos con tiempo T se extraen uno por uno (a igual tiempo,
       en orden de inserción). Cada evento se encola en Q_i y, si el agente i procesó menos de K eventos
       en este instante, se procesa de inmediato (una sesión atómica); si no, espera en Q_i. Un evento
       estático se reprograma en T + dt al despacharse. Un evento dirigido a un agente eliminado se
       descarta y no se reprograma.

    Los eventos dinámicos emitidos durante una sesión entran a Q en T + latencia; con latencia cero se
    despachan en este mismo instante, después de los que ya estaban en Q con tiempo T. La simulación
    termina cuando Q queda vacío o el siguiente evento excede T_max; después se vacían las colas
    pendientes sin despachar eventos nuevos, K eventos por agente en cada vuelta.
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

    def _clock(self):
        return self.T

    def _emit(self, event, delay=None, from_channel=False):
        if not self._dispatching:
            return
        if from_channel:
            # Un evento de canal solo llega a los participantes que implementan el autómata de esa señal.
            agent = self.agents._find(event.get("agent_id"))
            if agent is None or event.get("signal") not in agent.get("automata", []):
                return
        delay = self.event_latency if delay is None else delay
        if delay < 0:
            raise ValueError(f"event latency must be non-negative, got {delay}")
        self.calendar.push(self.T + delay, event)
        tracer.emit("des", "schedule_dynamic", at=round(self.T + delay, 9), agent=event.get("agent_id"),
                    signal=event.get("signal"), channel=event.get("channel_id"))

    def _process_ready(self, ready):
        for agent_id in list(ready):
            self.agents.process_queue(agent_id, self.queue_batch)
            if self.agents.pending(agent_id) == 0:
                ready.pop(agent_id, None)

    def run_simulation(self, max_time: float = 60.0):
        self.automata.set_event_sink(self._emit)
        self.environments.event_sink = self._emit
        self.environments.clock = self._clock
        if hasattr(self.snapshot_manager, "set_clock"):
            self.snapshot_manager.set_clock(self._clock)
        self.agents.set_on_agent_added(lambda agent: self.event_scheduler.schedule_agent(agent, self.T))

        # Inicialización (T = 0)
        self.agents.start()
        self.event_scheduler.start(self.calendar, 0.0)
        self._dispatching = True

        K = self.queue_batch
        # Agentes con Q_i no vacía, en el orden en que su cola dejó de estar vacía.
        ready = {}

        # Ejecución
        while self.calendar and self.calendar.next_time() <= max_time:
            T = self.calendar.next_time()
            self.T = T
            tracer.clock = T
            if tracer.on("des"):
                at_T = self.calendar.peek_instant()
                tracer.emit("des", "instant", dispatched=len(at_T),
                            events=[(e.get("agent_id"), e.get("signal"), e.get("event_category")) for e in at_T],
                            calendar_size=len(self.calendar), backlog=list(ready))

            processed = {}

            # 1. Rezago de instantes anteriores, sujeto a K.
            for agent_id in list(ready):
                processed[agent_id] = self.agents.process_queue(agent_id, K)
                if self.agents.pending(agent_id) == 0:
                    ready.pop(agent_id, None)

            # 2. Eventos del instante en orden de calendario.
            same_instant = 0
            while self.calendar and self.calendar.next_time() == T:
                _, event = self.calendar.pop()
                same_instant += 1
                if same_instant > MAX_EVENTS_PER_INSTANT:
                    print(f"[FATAL] DES: more than {MAX_EVENTS_PER_INSTANT} events dispatched at T={T}.",
                          file=sys.stderr, flush=True)
                    sys.exit(1)

                agent_id = event.get("agent_id")
                if not self.agents.receive_event(agent_id, event):
                    ready.pop(agent_id, None)
                    continue
                if event.get("event_category") == "static":
                    self.event_scheduler.reschedule(event, T)
                tracer.emit("des", "dispatch", agent=agent_id, signal=event.get("signal"),
                            category=event.get("event_category"))

                if processed.get(agent_id, 0) < K:
                    processed[agent_id] = processed.get(agent_id, 0) + self.agents.process_queue(agent_id, 1)
                if self.agents.pending(agent_id) > 0:
                    ready[agent_id] = None
                else:
                    ready.pop(agent_id, None)

        # Finalización: cesa el despacho y se vacían las colas.
        self._dispatching = False
        self.event_scheduler.stop()
        tracer.emit("des", "drain", pending_agents=len(ready), discarded_calendar=len(self.calendar))
        while ready:
            self._process_ready(ready)
        self.agents.stop()
        self.calendar.clear()
        tracer.emit("des", "end", final_T=self.T)
        return self.T
