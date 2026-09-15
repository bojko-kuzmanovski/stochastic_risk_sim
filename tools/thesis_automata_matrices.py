"""
Genera las matrices de transición simbólicas de los autómatas del caso de estudio para el Anexo B de la tesis.

Herramienta de autoría (no la importa el motor). Complementa a thesis_automata_diagrams.py: lee automata.json,
agents.json, events.json, patl.json y un archivo de distribuciones, y escribe en el directorio de salida:

    matrices_estilos.tex      colores y glifos (dibujados con TikZ, sin fuentes adicionales)
    matrices_leyenda.tex      leyenda única de marcas de estado, etiquetas y glifos
    matriz_<automata>.tex     un cuadro por autómata: matriz y notas de sus etiquetas

Construcción de la matriz M de un autómata con estados s_1, ..., s_n:
  * Orden de flujo: capas por camino más largo desde el estado inicial (con ruptura de ciclos, igual que en
    los diagramas) y, dentro de cada capa, orden de descubrimiento en profundidad siguiendo los casos en su
    orden de evaluación. Los estados inalcanzables van al final y se atenúan.
  * M[i][j] reúne los casos de la transición de s_i cuyo destino es s_j. Cada caso lleva una etiqueta:
      p_r   caso de una compuerta probabilística: probabilidad de que la muestra cumpla la condición del caso
            y ninguna de las anteriores (regla del primer caso que coincide);
      g_r   caso de una transición que compara una lectura o un cálculo: indicador de la condición;
      1     transición incondicional o muestreo sin bifurcación.
  * Glifos de acción del caso: contexto de sesión, escritura en el propio agente, escritura en otro agente,
    escritura en el entorno, emisión de evento y creación de agente. El fondo de la celda toma el color del
    efecto de mayor alcance.
  * El destinatario de cada escritura en agentes se resuelve con un análisis de flujo de datos hacia adelante
    sobre las variables de sesión: una variable tomada del evento (event_agent_id) es el propio agente; un
    identificador literal lo es si pertenece a un tipo que ejecuta el autómata y ese tipo tiene una sola
    instancia. Los tipos que ejecutan un autómata son los que lo tienen asignado y lo reciben por evento
    estático o por emisión con destinatario resuelto.

Uso:
    python tools/thesis_automata_matrices.py \\
        --config configs/startups/runway-risk \\
        --out ~/Desktop/BK/LaTex/UNAM-MATE/TITULACION-LICENCIATURA/secciones_reporte/anexo_b_diagramas
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_automata_diagrams import (DEFAULT_CONFIG, DEFAULT_OUT, Automaton, break_cycles, case_label,  # noqa: E402
                                      load_abbreviations, longest_path_layers, tex_escape)

LANDSCAPE_FROM = 13      # estados a partir de los cuales la matriz va en página horizontal
NOTE_COLUMNS = 3         # columnas de la lista de notas

UNKNOWN = ("unk",)
EVENT = ("event",)

# Prioridad de efectos para el color de fondo (mayor alcance primero).
EFFECTS = ["add", "emit", "env", "other", "own", "ctx"]
GLYPH = {"ctx": r"\anxmctx", "own": r"\anxmown", "other": r"\anxmother", "env": r"\anxmenv",
         "emit": r"\anxmemit", "add": r"\anxmadd"}
FILL = {"add": "anxmadd", "emit": "anxmemit", "env": "anxmenv", "other": "anxmother", "own": "anxmown",
        "ctx": "anxmctx"}
ENV_WRITES = {"write_env_param", "write_member", "remove_rel", "add_rel", "write_rel", "write_channel"}


def esc(s):
    return tex_escape(s).replace(r"\_", r"\_\allowbreak{}")


# ----------------------------------------------------------------------------------------------------
# Flujo de datos de variables de sesión
# ----------------------------------------------------------------------------------------------------
def var_sources(env, ref):
    if not isinstance(ref, str):
        return {("lit", ref)}
    if not ref.startswith("$"):
        return {("lit", ref)}
    for key in (ref, ref[1:], "$" + ref.lstrip("$")):
        if key in env:
            return set(env[key])
    return {UNKNOWN}


def apply_actions(env, actions, on_write=None, on_emit=None):
    """Aplica en orden las acciones de un caso sobre el entorno simbólico de variables."""
    env = {k: set(v) for k, v in env.items()}
    for act in actions:
        if "update_params" in act:
            for k, v in act["update_params"].items():
                if isinstance(v, dict) and v.get("type") == "deterministic":
                    env[k] = var_sources(env, v.get("value"))
                elif isinstance(v, dict) and v.get("method") == "event_agent_id":
                    env[k] = {EVENT}
                else:
                    env[k] = {UNKNOWN}
        elif "action_required" in act:
            for k, spec in act["action_required"].items():
                if isinstance(spec, dict) and spec.get("method") and on_write:
                    on_write(spec, env)
                env[k] = {UNKNOWN}
        elif "event_emit" in act and on_emit:
            on_emit(act["event_emit"], env)
    return env


def join(a, b):
    out = {k: set(v) for k, v in a.items()}
    changed = False
    for k, v in b.items():
        if k not in out:
            out[k] = set(v)
            changed = True
        elif not v <= out[k]:
            out[k] |= v
            changed = True
    return out, changed


def fixpoint(spec):
    """Entorno simbólico de variables a la entrada de cada estado."""
    IN = {spec["states"]["initial"]: {}}
    for _ in range(60):
        changed = False
        for t in spec["transitions"]:
            u = t["from"]
            if u not in IN:
                continue
            for th in t["thresholds"]:
                out = apply_actions(IN[u], th["actions"])
                v = th["to"]
                if v not in IN:
                    IN[v] = out
                    changed = True
                else:
                    IN[v], ch = join(IN[v], out)
                    changed |= ch
        if not changed:
            break
    return IN


# ----------------------------------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------------------------------
class Model:
    def __init__(self, config, distributions):
        self.automata = json.loads((config / "automata.json").read_text(encoding="utf-8"))
        agents = json.loads((config / "agents.json").read_text(encoding="utf-8"))
        events = json.loads((config / "events.json").read_text(encoding="utf-8"))
        self.patl = json.loads((config / "patl.json").read_text(encoding="utf-8"))
        dists = json.loads((config / distributions).read_text(encoding="utf-8"))
        self.families = {d["distribution_name"]: d["family"] for d in dists}
        self.quantity = {a["agent_type"]: a.get("quantity", 1) for a in agents}
        self.assigned = defaultdict(set)
        for a in agents:
            for name in a.get("automata", []):
                self.assigned[name].add(a["agent_type"])
        self.abbr = load_abbreviations()
        self.triggers = defaultdict(list)
        self.targets = defaultdict(set)
        for o in self.patl["observations"]:
            self.triggers[(o["automaton_name"], o["trigger_state"])].extend(p["predicate_id"] for p in o["predicates"])
            for p in o["predicates"]:
                for grp in p["coalition"]:
                    for aut in grp["automata"]:
                        for s in aut.get("target_states", []):
                            self.targets[(aut["automaton_name"], s)].add(self.abbr[p["predicate_id"]])
        self.envs = {a["automaton_name"]: fixpoint(a) for a in self.automata}
        receivers = defaultdict(set)
        for e in events:
            if e.get("event_category") == "static" and e.get("agent_type"):
                receivers[e["signal"]].add(e["agent_type"])
        for a in self.automata:
            IN = self.envs[a["automaton_name"]]

            def on_emit(args, env, emitter=a["automaton_name"]):
                for s in var_sources(env, args[0]):
                    if s[0] != "lit":
                        continue
                    for g in var_sources(env, args[1]):
                        if g[0] == "lit":
                            receivers[s[1]].add(self.type_of(g[1]))
                        elif g == EVENT:
                            # Destinatario tomado del evento: un agente de algún tipo que ejecuta al emisor.
                            receivers[s[1]].update(self.assigned[emitter])
                        else:
                            receivers[s[1]].add(None)

            for t in a["transitions"]:
                if t["from"] in IN:
                    for th in t["thresholds"]:
                        apply_actions(IN[t["from"]], th["actions"], on_emit=on_emit)
        self.executors = {}
        for a in self.automata:
            name = a["automaton_name"]
            got = receivers.get(name, set())
            known = {x for x in got if x} & self.assigned[name]
            self.executors[name] = known if known and None not in got else set(self.assigned[name])

    def type_of(self, agent_id):
        m = re.match(r"^(.*)_(\d+)$", str(agent_id))
        return m.group(1) if m else None

    def classify_target(self, aut_name, sources):
        kinds = set()
        for s in sources:
            if s == EVENT:
                kinds.add("own")
            elif s[0] == "lit":
                typ = self.type_of(s[1])
                if typ in self.executors[aut_name] and self.quantity.get(typ, 1) == 1:
                    kinds.add("own")
                else:
                    kinds.add("other")
            else:
                kinds.add("other?")
        return kinds


# ----------------------------------------------------------------------------------------------------
# Construcción de una matriz
# ----------------------------------------------------------------------------------------------------
def flow_order(spec):
    initial = spec["states"]["initial"]
    succ = defaultdict(list)
    nodes = [initial]
    for t in spec["transitions"]:
        for th in t["thresholds"]:
            for s in (t["from"], th["to"]):
                if s not in nodes:
                    nodes.append(s)
            if th["to"] not in succ[t["from"]]:
                succ[t["from"]].append(th["to"])
    for f in spec["states"].get("final", []):
        if f not in nodes:
            nodes.append(f)
    seen, disc = set(), {}

    def dfs(u):
        seen.add(u)
        disc[u] = len(disc)
        for v in succ[u]:
            if v not in seen:
                dfs(v)

    dfs(initial)
    reachable = set(seen)
    adj = {n: list(succ[n]) for n in nodes}
    back = break_cycles(nodes, adj, initial)
    dag = [((v, u) if (u, v) in back else (u, v)) for u in nodes for v in succ[u] if u != v]
    layer = longest_path_layers([n for n in nodes if n in reachable], [e for e in dag if e[0] in reachable])
    ordered = sorted((n for n in nodes if n in reachable), key=lambda n: (layer[n], disc[n]))
    unreachable = [n for n in nodes if n not in reachable]
    return ordered + unreachable, reachable


def build(model, spec):
    name = spec["automaton_name"]
    aut = Automaton(spec, model.families, model.triggers, model.abbr)
    order, reachable = flow_order(spec)
    index = {s: i for i, s in enumerate(order)}
    finals = set(spec["states"].get("final", []))
    incoming = defaultdict(int)
    for t in spec["transitions"]:
        for th in t["thresholds"]:
            incoming[th["to"]] += 1
    IN = model.envs[name]

    cells = defaultdict(lambda: {"labels": [], "effects": set()})
    notes = []
    counters = {"p": 0, "g": 0}
    unresolved = []
    for u in order:
        t = next((t for t in spec["transitions"] if t["from"] == u), None)
        if t is None:
            continue
        (var, tv), = t["threshold_value"].items()
        ncases = len(t["thresholds"])
        probabilistic = tv.get("type") == "probabilistic"
        for ci, th in enumerate(t["thresholds"]):
            conds = th["threshold_case"]
            trivial = (ncases == 1 and (tv.get("type") == "deterministic"
                                        or (len(conds) == 1 and conds[0]["operator"] == "!=" and conds[0]["value"] is None)))
            cell = cells[(u, th["to"])]
            if trivial:
                cell["labels"].append("1")
            else:
                kind = "p" if probabilistic else "g"
                counters[kind] += 1
                lab = f"{kind}_{{{counters[kind]}}}"
                cell["labels"].append(lab)
                cond, _ = case_label(conds)
                if probabilistic:
                    subject = r"$X \sim$ \textsf{" + esc(tv["distribution"]) + "}"
                else:
                    det = aut.detail(u)
                    if det and det[1] not in ("cálculo", "relación"):
                        what = det[1]
                    elif det:
                        what = f"{var.lstrip('$')} ({det[1]})"
                    else:
                        what = var.lstrip("$")
                    subject = r"\textsf{" + esc(what) + "}"
                notes.append((f"${lab}$", subject, cond, f"{ci + 1}/{ncases}", u, th["to"]))

            effects = set()

            def on_write(call, env, effects=effects):
                method = call.get("method", "")
                if method == "write_agent_param":
                    ref = (call.get("params") or ["$agent_id"])[0]
                    kinds = model.classify_target(name, var_sources(env, ref))
                    if "other?" in kinds:
                        unresolved.append((u, th["to"], ref))
                        kinds = (kinds - {"other?"}) | {"other"}
                    effects.update(kinds)
                elif method in ENV_WRITES:
                    effects.add("env")
                elif method == "add_agent":
                    effects.add("add")
                else:
                    effects.add("ctx")

            def on_emit(args, env, effects=effects):
                effects.add("emit")

            if any("update_params" in a for a in th["actions"]):
                effects.add("ctx")
            apply_actions(IN.get(u, {}), th["actions"], on_write=on_write, on_emit=on_emit)
            cell["effects"] |= effects

    return {
        "name": name, "order": order, "index": index, "reachable": reachable, "finals": finals,
        "initial": spec["states"]["initial"], "incoming": incoming, "cells": cells, "notes": notes,
        "executors": sorted(model.executors[name]), "unresolved": unresolved,
        "triggers": {s: [model.abbr[p] for p in pids] for (a, s), pids in model.triggers.items() if a == name},
        "targets": {s: sorted(ab) for (a, s), ab in model.targets.items() if a == name},
    }


# ----------------------------------------------------------------------------------------------------
# Escritura LaTeX
# ----------------------------------------------------------------------------------------------------
def state_label(M, s):
    marks = []
    if s == M["initial"]:
        marks.append(r"\anxminit")
    if s in M["finals"]:
        marks.append(r"\anxmfinal")
    if s in M["triggers"]:
        marks.append(r"\anxmtrig")
    if s in M["targets"]:
        marks.append(r"\anxmtarget")
    unused = s not in M["reachable"] or (s != M["initial"] and M["incoming"][s] == 0)
    name = r"\texttt{" + tex_escape(s) + "}"
    if unused:
        name = r"\textcolor{black!40}{" + name + "}"
    tags = []
    if s in M["triggers"]:
        tags.append(r"\anxmtag{" + ", ".join(M["triggers"][s]) + "}")
    if s in M["targets"]:
        tags.append(r"\anxmtagt{" + ", ".join(M["targets"][s]) + "}")
    return r"\,".join(marks), name +("\\," + "\\,".join(tags) if tags else ""), unused


def cell_tex(cell):
    labels = cell["labels"]
    numbered = [l for l in labels if l != "1"]
    if numbered:
        sep = "{+}" if numbered[0].startswith("p") else r"{\lor}"
        lab = "$" + sep.join(numbered) + "$"
        if "1" in labels:
            lab = "$1$"
    else:
        lab = "$1$"
    glyphs = "".join(GLYPH[e] for e in EFFECTS[::-1] if e in cell["effects"])
    top = next((e for e in EFFECTS if e in cell["effects"]), None)
    color = r"\cellcolor{" + FILL[top] + "!18}" if top and top != "ctx" else ""
    return color + lab + (r"\," + glyphs if glyphs else "")


def render(M):
    n = len(M["order"])
    land = n >= LANDSCAPE_FROM
    lines = [
        f"% Matriz de transición simbólica del autómata {M['name']}.",
        "% Generado por tools/thesis_automata_matrices.py (simulador) a partir de automata.json, agents.json,",
        "% events.json y patl.json. No editar a mano: volver a ejecutar el generador.",
    ]
    colspec = "r@{\\hspace{2pt}}l@{\\hspace{3pt}}l" + "|c" * n + "|"
    lines.append(r"\begin{anxmfit}{" + (r"\footnotesize" if land else r"\scriptsize") + "}")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    head = [r"\multicolumn{3}{r|}{\anxmhead{de $\downarrow$ \quad a $\rightarrow$}}"]
    head += [r"\multicolumn{1}{c|}{\anxmhead{" + str(j + 1) + "}}" for j in range(n)]
    lines.append(" & ".join(head) + r" \\ \hline")
    for i, s in enumerate(M["order"]):
        marks, label, unused = state_label(M, s)
        row = [r"\anxmhead{" + str(i + 1) + "}", marks, label]
        for j, v in enumerate(M["order"]):
            cell = M["cells"].get((s, v))
            if cell:
                row.append(cell_tex(cell))
            elif i == j:
                row.append(r"\cellcolor{black!6}")
            else:
                row.append(r"\anxmempty")
        lines.append(" & ".join(row) + r" \\ \hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{anxmfit}")
    notes = M["notes"]
    if notes:
        ncols = NOTE_COLUMNS if land else 2
        per = -(-len(notes) // ncols)
        cols = [notes[k * per:(k + 1) * per] for k in range(ncols)]
        width = "0.325" if land else "0.49"
        lines.append(r"\par\smallskip")
        lines.append(r"\begin{minipage}[t]{\linewidth}\anxmnotes")
        for k, col in enumerate(cols):
            if not col:
                continue
            lines.append(r"\begin{minipage}[t]{" + width + r"\linewidth}\raggedright")
            for lab, subject, cond, case, u, v in col:
                src = (r"\textcolor{black!55}{\texttt{" + f"{M['index'][u] + 1}" + r"}$\to$\texttt{"
                       + f"{M['index'][v] + 1}" + r"}}\ ")
                lines.append(rf"{src}{lab}\ {subject}: {cond} \textcolor{{black!55}}{{(caso {case})}}\par")
            lines.append(r"\end{minipage}" + (r"\hfill" if k < ncols - 1 else ""))
        lines.append(r"\end{minipage}")
    return "\n".join(lines) + "\n", land


STYLES = r"""% Colores y glifos de las matrices de transición simbólicas del Anexo B.
% Generado por tools/thesis_automata_matrices.py (simulador). No editar a mano.
\definecolor{unamazul}{HTML}{002B7A}
\definecolor{unamoro}{HTML}{C5911F}
\colorlet{anxmctx}{black!55}
\colorlet{anxmown}{unamazul}
\definecolor{anxmother}{HTML}{7B2D8E}
\definecolor{anxmenv}{HTML}{1E7B3A}
\colorlet{anxmemit}{unamoro!90!black}
\definecolor{anxmadd}{HTML}{B22222}
\newcommand{\anxmglyph}[1]{\tikz[baseline=-0.55ex, line join=round]{#1}}
\newcommand{\anxmctx}{\anxmglyph{\draw[anxmctx, line width=0.55pt] (0,0) circle (1.25pt);}}
\newcommand{\anxmown}{\anxmglyph{\node[star, star points=5, star point ratio=2.4, fill=anxmown, inner sep=0pt,
  minimum size=4.6pt] {};}}
\newcommand{\anxmother}{\anxmglyph{\node[star, star points=5, star point ratio=2.4, draw=anxmother,
  line width=0.45pt, inner sep=0pt, minimum size=4.6pt] {};}}
\newcommand{\anxmenv}{\anxmglyph{\node[diamond, fill=anxmenv, inner sep=0pt, minimum size=4.2pt] {};}}
\newcommand{\anxmemit}{\anxmglyph{\draw[anxmemit, line width=0.45pt] (-1.9pt,-1.3pt) rectangle (1.9pt,1.3pt);
  \draw[anxmemit, line width=0.45pt] (-1.9pt,1.3pt) -- (0,-0.2pt) -- (1.9pt,1.3pt);}}
\newcommand{\anxmadd}{\anxmglyph{\draw[anxmadd, line width=1.0pt] (-1.8pt,0) -- (1.8pt,0) (0,-1.8pt) -- (0,1.8pt);}}
\newcommand{\anxminit}{\anxmglyph{\fill[unamazul] (-1.3pt,-1.6pt) -- (1.6pt,0) -- (-1.3pt,1.6pt) -- cycle;}}
\newcommand{\anxmfinal}{\anxmglyph{\fill[black!80] (-1.5pt,-1.5pt) rectangle (1.5pt,1.5pt);}}
\newcommand{\anxmtrig}{\anxmglyph{\draw[unamazul, line width=0.5pt] (0,0) circle (1.8pt); \fill[unamazul] (0,0)
  circle (0.8pt);}}
\newcommand{\anxmtarget}{\anxmglyph{\draw[anxmadd, line width=0.5pt] (-1.4pt,-2pt) -- (-1.4pt,2pt);
  \fill[anxmadd] (-1.4pt,2pt) -- (1.8pt,1.1pt) -- (-1.4pt,0.2pt) -- cycle;}}
\newcommand{\anxmtag}[1]{{\sffamily\bfseries\textcolor{unamazul}{#1}}}
\newcommand{\anxmtagt}[1]{{\sffamily\bfseries\textcolor{anxmadd}{#1}}}
\newcommand{\anxmhead}[1]{{\sffamily\bfseries\textcolor{black!70}{#1}}}
\newcommand{\anxmempty}{\textcolor{black!25}{$\cdot$}}
\newcommand{\anxmnotes}{\scriptsize\setlength{\parskip}{0.6pt}}
% Ajusta la matriz al ancho de la línea sin ampliarla nunca.
\newsavebox{\anxmbox}
\newenvironment{anxmfit}[1]{\begin{lrbox}{\anxmbox}#1\arrayrulecolor{black!18}\renewcommand{\arraystretch}{1.35}%
  \setlength{\tabcolsep}{1.6pt}}{\end{lrbox}%
  \ifdim\wd\anxmbox>\linewidth\resizebox{\linewidth}{!}{\usebox{\anxmbox}}\else\usebox{\anxmbox}\fi}
"""

LEGEND = r"""% Leyenda de las matrices de transición simbólicas del Anexo B.
% Generado por tools/thesis_automata_matrices.py (simulador). No editar a mano.
{\scriptsize
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}c l @{\hspace{14pt}} c l @{\hspace{14pt}} c l@{}}
  \multicolumn{2}{@{}l}{\textbf{Estados (filas)}} & \multicolumn{2}{l}{\textbf{Etiquetas de celda}} & \multicolumn{2}{l@{}}{\textbf{Acciones del caso}} \\[1pt]
  \anxminit & inicial & $p_r$ & caso probabilístico & \anxmctx & actualiza el contexto de sesión \\
  \anxmfinal & final & $g_r$ & caso con condición & \anxmown & escribe en el propio agente \\
  \anxmtrig\ \anxmtag{IDV} & disparador PATL & $1$ & incondicional & \anxmother & escribe en otro agente \\
  \anxmtarget\ \anxmtagt{IDV} & objetivo de predicado & \anxmempty & sin transición & \anxmenv & escribe en el entorno \\
  \textcolor{black!40}{\texttt{NOMBRE}} & sin uso & $p_1{+}p_2$ & varios casos al mismo destino & \anxmemit & emite un evento \\
  & & & & \anxmadd & crea un agente \\
\end{tabular}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--distributions", default="distributions_1_base.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    model = Model(args.config, args.distributions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrices_estilos.tex").write_text(STYLES, encoding="utf-8")
    (args.out / "matrices_leyenda.tex").write_text(LEGEND, encoding="utf-8")
    print(f"{'autómata':32s} estados horiz notas ejecutores / sin resolver")
    floats = ["% Cuadros de las matrices de transición simbólicas, en el orden de automata.json.",
              "% Generado por tools/thesis_automata_matrices.py (simulador). No editar a mano.", ""]
    for spec in model.automata:
        M = build(model, spec)
        tex, land = render(M)
        (args.out / f"matriz_{M['name']}.tex").write_text(tex, encoding="utf-8")
        who = ", ".join(r"\texttt{" + t + "}" for t in M["executors"])
        table = [r"\begin{table}[H]", r"  \centering",
                 r"  \caption{\textit{\textbf{Matriz de transición simbólica de \texttt{" + esc(M["name"]) + r"}.}} "
                 + f"Ejecuta: {who}. Diagrama en la Figura~\\ref{{fig:anxb-{M['name']}}}.}}",
                 rf"  \label{{tab:anxb-mat-{M['name']}}}",
                 rf"  \input{{\anxbdir/matriz_{M['name']}.tex}}", r"\end{table}"]
        if land:
            table = [r"\begin{landscape}"] + table + [r"\end{landscape}"]
        floats += table + [""]
        print(f"{M['name']:32s} {len(M['order']):7d} {'sí' if land else 'no':5s} {len(M['notes']):5d} "
              f"{','.join(M['executors'])} / {M['unresolved'] or '-'}")
    (args.out / "matrices_todas.tex").write_text("\n".join(floats), encoding="utf-8")


if __name__ == "__main__":
    main()
