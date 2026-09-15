"""
Genera los diagramas de estados (TikZ) de los autómatas del caso de estudio para el Anexo B de la tesis.

Herramienta de autoría (no la importa el motor). Lee automata.json, patl.json y un archivo de
distribuciones, y escribe en el directorio de salida:

    estilos.tex        colores, estilos TikZ y la macro \\anxbfit (ajuste al ancho y alto disponibles)
    leyenda.tex        leyenda única de los estilos
    <automata>.tex     un tikzpicture por autómata, envuelto en \\anxbfit

La disposición se calcula aquí, no a mano, con el esquema por capas de Sugiyama, Tagawa y Toda (1981):
  1. Ruptura de ciclos: búsqueda en profundidad desde el estado inicial; las aristas de retroceso se
     invierten solo para el cálculo de capas.
  2. Capas: longitud del camino más largo desde las fuentes en el grafo acíclico resultante.
  3. Nodos ficticios en las aristas que cruzan más de una capa, para ordenar y trazar sin atravesar nodos.
  4. Orden dentro de cada capa: heurística de baricentro con barridos descendentes y ascendentes,
     seguida de transposiciones adyacentes; se conserva el orden con menos cruces.
  5. Coordenadas: cada pasada fija la posición deseada de un nodo en el promedio de sus vecinos y
     resuelve una regresión isotónica ponderada (pool adjacent violators) que respeta el orden y la
     separación mínima.
  6. Orientación: de izquierda a derecha si cabe en el ancho de página sin ser más alta que la vertical;
     en otro caso, de arriba abajo.
  7. Etiquetas de aristas: se prueban posiciones a lo largo de cada tramo y se elige la primera que no
     se enciman con nodos, otras etiquetas ni otras aristas.

Uso:
    python tools/thesis_automata_diagrams.py \\
        --config configs/startups/runway-risk \\
        --out ~/Desktop/BK/LaTex/UNAM-MATE/TITULACION-LICENCIATURA/secciones_reporte/anexo_b_diagramas
"""
import argparse
import ast
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "startups" / "runway-risk"
DEFAULT_OUT = (Path.home() / "Desktop" / "BK" / "LaTex" / "UNAM-MATE" / "TITULACION-LICENCIATURA"
               / "secciones_reporte" / "anexo_b_diagramas")

# Dimensiones de página (cm) y métricas aproximadas de la fuente \tiny en un documento de 12 pt.
TEXT_WIDTH = 16.2
TT_CHAR = 0.111      # ancho de un carácter \texttt en \tiny
SF_CHAR = 0.104      # ancho medio de un carácter \sffamily\itshape en \tiny
MATH_CHAR = 0.118    # ancho medio en etiquetas matemáticas \tiny (con espacios de relación)
LINE_H = 0.235       # alto de renglón en \tiny
NODE_PAD = 0.12      # inner sep más grosor de borde, por lado
MAX_TT = 13          # caracteres por renglón en nombres de estado
MAX_SF = 17          # caracteres por renglón en nombres de distribución o parámetro

OPS = {"<": "<", "<=": r"\leq", ">": ">", ">=": r"\geq", "==": "=", "!=": r"\neq"}
EMIT_MARK = r"\anxbemit"
WRITE_MARK = r"\anxbwrite"


# ----------------------------------------------------------------------------------------------------
# Lectura del modelo
# ----------------------------------------------------------------------------------------------------
def load_abbreviations():
    """Abreviaturas de predicados tomadas de thesis_results.py (sin importar sus dependencias)."""
    src = (Path(__file__).resolve().parent / "thesis_results.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PREDICATES" for t in node.targets):
            return {pid: ab for ab, pid, _ in ast.literal_eval(node.value)}
    raise RuntimeError("PREDICATES no encontrado en thesis_results.py")


def tex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("$", r"\$").replace("&", r"\&")


def split_name(name, maxc):
    parts = str(name).split("_")
    lines, cur = [], ""
    for p in parts:
        cand = p if not cur else cur + "_" + p
        if len(cand) <= maxc or not cur:
            cur = cand
        else:
            lines.append(cur + "_")
            cur = p
    lines.append(cur)
    return lines


def fmt_value(v):
    if isinstance(v, bool):
        return r"\texttt{" + ("true" if v else "false") + "}"
    if v is None:
        return r"\texttt{null}"
    if isinstance(v, str):
        return r"\texttt{" + tex_escape(v) + "}"
    return json.dumps(v)


def value_len(v):
    if isinstance(v, bool):
        return 5 * TT_CHAR / MATH_CHAR
    if isinstance(v, str):
        return len(v) * TT_CHAR / MATH_CHAR
    return len(json.dumps(v))


def case_label(conds):
    """Condición breve de un caso y su longitud aproximada en caracteres."""
    if len(conds) == 2 and all(isinstance(c["value"], (int, float)) and not isinstance(c["value"], bool)
                               for c in conds):
        lo = [c for c in conds if c["operator"] in (">", ">=")]
        hi = [c for c in conds if c["operator"] in ("<", "<=")]
        if len(lo) == 1 and len(hi) == 1:
            a, b = lo[0], hi[0]
            left = "(" if a["operator"] == ">" else "["
            right = ")" if b["operator"] == "<" else "]"
            txt = f"{left}{json.dumps(a['value'])}, {json.dumps(b['value'])}{right}"
            return f"${txt}$", len(txt)
    pieces, n = [], 0
    for c in conds:
        op, v = c["operator"], c["value"]
        if isinstance(v, str):
            pieces.append(f"${OPS[op]}$ {fmt_value(v)}")
        else:
            pieces.append(f"${OPS[op]} {fmt_value(v)}$")
        n += 2 + value_len(v)
    return r" $\land$ ".join(pieces), n + 2 * (len(conds) - 1)


class Automaton:
    def __init__(self, spec, families, triggers, abbr):
        self.name = spec["automaton_name"]
        self.initial = spec["states"]["initial"]
        self.finals = list(spec["states"]["final"])
        self.order = [self.initial]
        self.trans = {}
        for t in spec["transitions"]:
            self._add(t["from"])
            self.trans[t["from"]] = t
            for th in t["thresholds"]:
                self._add(th["to"])
        for f in self.finals:
            self._add(f)
        self.families = families
        self.trigger_tags = {}
        for (aut, state), pids in triggers.items():
            if aut == self.name:
                self.trigger_tags[state] = [abbr[p] for p in pids]
        self._build_edges()

    def _add(self, s):
        if s not in self.order:
            self.order.append(s)

    # Clasificación de estados --------------------------------------------------------------------
    def kind(self, s):
        if s not in self.trans:
            return "final"
        t = self.trans[s]
        (spec,) = t["threshold_value"].values()
        ncases = len(t["thresholds"])
        if spec.get("type") == "probabilistic":
            return "gate" if ncases >= 2 else "sample"
        return "det"

    def detail(self, s):
        """Segundo renglón del nodo: distribución muestreada o parámetro leído."""
        if s not in self.trans:
            return None
        t = self.trans[s]
        (spec,) = t["threshold_value"].values()
        if spec.get("type") == "probabilistic":
            return ("dist", spec["distribution"])
        method = spec.get("method")
        if method == "read_rel":
            return ("read", "relación")
        if method == "math_pipeline":
            return ("read", "cálculo")
        if method in ("read_agent_param", "read_env_param"):
            key = self._param_key_into(s)
            return ("read", key) if key else None
        return None

    def _param_key_into(self, s, depth=0, seen=None):
        seen = seen or set()
        if depth > 6 or s in seen:
            return None
        seen.add(s)
        found = set()
        preds = []
        for u, t in self.trans.items():
            for th in t["thresholds"]:
                if th["to"] != s:
                    continue
                key = None
                for act in th["actions"]:
                    up = act.get("update_params", {})
                    if "$param_key" in up and up["$param_key"].get("type") == "deterministic":
                        key = up["$param_key"]["value"]
                if key is not None:
                    found.add(key)
                else:
                    preds.append(u)
        if not found:
            for u in preds:
                k = self._param_key_into(u, depth + 1, seen)
                if k:
                    found.add(k)
        return found.pop() if len(found) == 1 else None

    # Aristas -------------------------------------------------------------------------------------
    def _build_edges(self):
        self.edges = {}
        for u, t in self.trans.items():
            (var, spec), = t["threshold_value"].items()
            unconditional = spec.get("type") == "deterministic" and len(t["thresholds"]) == 1
            for th in t["thresholds"]:
                v = th["to"]
                conds = th["threshold_case"]
                if unconditional or (len(conds) == 1 and conds[0]["operator"] == "!=" and conds[0]["value"] is None):
                    lab, n = "", 0
                else:
                    lab, n = case_label(conds)
                emit = any("event_emit" in a for a in th["actions"])
                write = False
                for a in th["actions"]:
                    for call in a.get("action_required", {}).values():
                        m = call.get("method", "")
                        if not m.startswith("read") and m != "event_agent_id":
                            write = True
                e = self.edges.setdefault((u, v), {"labels": [], "emit": False, "write": False})
                if lab:
                    e["labels"].append((lab, n))
                e["emit"] |= emit
                e["write"] |= write


# ----------------------------------------------------------------------------------------------------
# Geometría de nodos y etiquetas
# ----------------------------------------------------------------------------------------------------
def node_box(aut, s):
    lines = [(r"\texttt{" + tex_escape(l) + "}", len(l) * TT_CHAR) for l in split_name(s, MAX_TT)]
    det = aut.detail(s)
    if det:
        kind, name = det
        prefix = r"$\sim$\," if kind == "dist" else ""
        chunks = split_name(name, MAX_SF)
        for i, c in enumerate(chunks):
            txt = (prefix if i == 0 else "") + tex_escape(c)
            w = len(c) * SF_CHAR + (0.12 if (i == 0 and prefix) else 0)
            lines.append((r"\textsf{\itshape " + txt + "}", w))
    w = max(l[1] for l in lines) + 2 * NODE_PAD
    h = len(lines) * LINE_H + 2 * NODE_PAD - 0.02
    text = r"\\".join(l[0] for l in lines)
    return text, max(w, 0.9), h


def edge_label(e):
    labels = e["labels"]
    marks = ""
    if e["write"]:
        marks += WRITE_MARK
    if e["emit"]:
        marks += EMIT_MARK
    nmark = 1.6 * (int(e["write"]) + int(e["emit"]))
    if not labels and not marks:
        return None
    rows = [l for l, _ in labels] or [""]
    lens = [n for _, n in labels] or [0]
    rows[-1] = (rows[-1] + ("\\," if rows[-1] and marks else "") + marks)
    lens[-1] += nmark
    w = max(lens) * MATH_CHAR + 0.14
    h = len(rows) * (LINE_H - 0.01) + 0.06
    return r"\\".join(rows), w, h


# ----------------------------------------------------------------------------------------------------
# Disposición por capas
# ----------------------------------------------------------------------------------------------------
def break_cycles(nodes, adj, start):
    back, state = set(), {}

    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if v == u:
                continue
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2

    dfs(start)
    for n in nodes:
        if n not in state:
            dfs(n)
    return back


def longest_path_layers(nodes, dag_edges):
    indeg = {n: 0 for n in nodes}
    succ = {n: [] for n in nodes}
    for u, v in dag_edges:
        succ[u].append(v)
        indeg[v] += 1
    layer = {n: 0 for n in nodes}
    queue = [n for n in nodes if indeg[n] == 0]
    while queue:
        u = queue.pop(0)
        for v in succ[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return layer


def count_crossings(layers, links):
    total = 0
    pos = {}
    for lay in layers:
        for i, n in enumerate(lay):
            pos[n] = i
    for i in range(len(layers) - 1):
        es = [(pos[a], pos[b]) for a, b in links if a in layers[i] and b in layers[i + 1]]
        for x in range(len(es)):
            for y in range(x + 1, len(es)):
                if (es[x][0] - es[y][0]) * (es[x][1] - es[y][1]) < 0:
                    total += 1
    return total


def order_layers(layers, links, passes=24):
    up = {}
    down = {}
    for a, b in links:
        down.setdefault(a, []).append(b)
        up.setdefault(b, []).append(a)
    best = [list(l) for l in layers]
    best_c = count_crossings(best, links)
    cur = [list(l) for l in layers]

    def sweep(ref_map, rng, ref_offset):
        for i in rng:
            ref = {n: k for k, n in enumerate(cur[i + ref_offset])}
            keyed = []
            for k, n in enumerate(cur[i]):
                nb = [ref[m] for m in ref_map.get(n, []) if m in ref]
                keyed.append((sum(nb) / len(nb) if nb else k, k, n))
            keyed.sort()
            cur[i] = [n for _, _, n in keyed]

    for it in range(passes):
        sweep(up, range(1, len(cur)), -1)
        sweep(down, range(len(cur) - 2, -1, -1), +1)
        c = count_crossings(cur, links)
        if c < best_c:
            best, best_c = [list(l) for l in cur], c
    # Transposiciones adyacentes.
    cur = [list(l) for l in best]
    improved = True
    while improved:
        improved = False
        for i, lay in enumerate(cur):
            for k in range(len(lay) - 1):
                lay[k], lay[k + 1] = lay[k + 1], lay[k]
                c = count_crossings(cur, links)
                if c < best_c:
                    best_c, improved = c, True
                    best = [list(l) for l in cur]
                else:
                    lay[k], lay[k + 1] = lay[k + 1], lay[k]
    return best, best_c


def isotonic_place(desired, weights, seps):
    """Minimiza sum w (x - d)^2 con x[i+1] - x[i] >= seps[i]."""
    n = len(desired)
    off = [0.0] * n
    for i in range(1, n):
        off[i] = off[i - 1] + seps[i - 1]
    target = [desired[i] - off[i] for i in range(n)]
    blocks = []  # [valor, peso, cuenta]
    for i in range(n):
        blocks.append([target[i], weights[i], 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    y = []
    for v, _, c in blocks:
        y.extend([v] * c)
    return [y[i] + off[i] for i in range(n)]


def layout(aut, orientation):
    nodes = list(aut.order)
    adj = {n: [] for n in nodes}
    for (u, v) in aut.edges:
        adj[u].append(v)
    back = break_cycles(nodes, adj, aut.initial)
    dag = []
    for (u, v) in aut.edges:
        if u == v:
            continue
        dag.append((v, u) if (u, v) in back else (u, v))
    layer = longest_path_layers(nodes, dag)
    nlayers = max(layer.values()) + 1

    boxes = {n: node_box(aut, n) for n in nodes}
    labels = {k: edge_label(e) for k, e in aut.edges.items()}

    # Nodos ficticios.
    # Etiqueta de predicados: a la derecha del nodo (vertical) o encima (horizontal); se reserva su sitio.
    tag_dims = {}
    for s, abbrs in aut.trigger_tags.items():
        txt = ", ".join(abbrs)
        tag_dims[s] = (txt, len(txt) * 0.095 + 0.1, 0.22)
    size = {}
    for n in nodes:
        _, w, h = boxes[n]
        size[n] = (w, h)
        if n in tag_dims:
            _, tw, th = tag_dims[n]
            size[n] = (w + 2 * (tw + 0.06), h) if orientation == "TB" else (w, h + 2 * (th + 0.02))
    layers = [[] for _ in range(nlayers)]
    for n in nodes:
        layers[layer[n]].append(n)
    links = []
    paths = {}
    dummy_id = 0
    for (u, v) in aut.edges:
        if u == v:
            paths[(u, v)] = [u]
            continue
        a, b = ((v, u) if (u, v) in back else (u, v))
        chain = [a]
        for L in range(layer[a] + 1, layer[b]):
            d = f"__d{dummy_id}"
            dummy_id += 1
            layer[d] = L
            size[d] = (0.18, 0.18)
            layers[L].append(d)
            chain.append(d)
        chain.append(b)
        for x, y in zip(chain, chain[1:]):
            links.append((x, y))
        paths[(u, v)] = chain if (u, v) not in back else list(reversed(chain))

    layers, crossings = order_layers(layers, links)

    # Separaciones.
    def breadth(n):
        w, h = size[n]
        return w if orientation == "TB" else h

    def depth(n):
        w, h = size[n]
        return h if orientation == "TB" else w

    gap_same = 0.32 if orientation == "TB" else 0.26
    pos = {}
    for lay in layers:
        x = 0.0
        for i, n in enumerate(lay):
            if i:
                x += (breadth(lay[i - 1]) + breadth(n)) / 2 + gap_same
            pos[n] = x
        shift = x / 2
        for n in lay:
            pos[n] -= shift

    nbr_up, nbr_down = {}, {}
    for a, b in links:
        nbr_down.setdefault(a, []).append(b)
        nbr_up.setdefault(b, []).append(a)

    def relax(mode):
        rng = range(len(layers)) if mode != "up" else range(len(layers) - 1, -1, -1)
        for i in rng:
            lay = layers[i]
            if not lay:
                continue
            desired, weights = [], []
            for n in lay:
                nb = []
                if mode in ("down", "both"):
                    nb += nbr_up.get(n, [])
                if mode in ("up", "both"):
                    nb += nbr_down.get(n, [])
                if nb:
                    desired.append(sum(pos[m] for m in nb) / len(nb))
                    weights.append(3.0 if n.startswith("__d") else 1.0 + 0.2 * len(nb))
                else:
                    desired.append(pos[n])
                    weights.append(0.2)
            seps = [(breadth(lay[k]) + breadth(lay[k + 1])) / 2 + gap_same for k in range(len(lay) - 1)]
            new = isotonic_place(desired, weights, seps)
            for n, x in zip(lay, new):
                pos[n] = x

    for _ in range(10):
        relax("down")
        relax("up")
    for _ in range(6):
        relax("both")

    # Profundidad de capas: el hueco entre capas deja sitio a las etiquetas.
    layer_pos = []
    acc = 0.0
    for i, lay in enumerate(layers):
        dmax = max((depth(n) for n in lay if not n.startswith("__d")), default=0.2)
        if i == 0:
            acc = dmax / 2
        else:
            prev = layers[i - 1]
            pmax = max((depth(n) for n in prev if not n.startswith("__d")), default=0.2)
            lab_extent = 0.0
            for k, lb in labels.items():
                if lb is None or k[0] == k[1]:
                    continue
                ch = paths[k]
                for x, y in zip(ch, ch[1:]):
                    if {layer[x], layer[y]} == {i - 1, i}:
                        lab_extent = max(lab_extent, lb[2] if orientation == "TB" else lb[1])
            if lab_extent == 0:
                gap = 0.5 if orientation == "TB" else 0.6
            else:
                gap = max(0.8 if orientation == "TB" else 0.85, lab_extent + (0.6 if orientation == "TB" else 0.5))
            acc += pmax / 2 + gap + dmax / 2
        layer_pos.append(acc)

    xy = {}
    minb = min(pos[n] - breadth(n) / 2 for n in pos)
    for i, lay in enumerate(layers):
        for n in lay:
            b = pos[n] - minb
            d = layer_pos[i]
            xy[n] = (b, -d) if orientation == "TB" else (d, -b)

    # Rectángulos de nodos reales.
    rects = {n: (xy[n][0], xy[n][1], boxes[n][1], boxes[n][2]) for n in nodes}

    # Marcador de estado inicial.
    ix, iy, iw, ih = rects[aut.initial]
    if orientation == "TB":
        start = (ix, iy + ih / 2 + 0.34)
    else:
        start = (ix - iw / 2 - 0.34, iy)

    # Etiqueta de predicados (esquina superior derecha).
    tags = {}
    for s, (txt, tw, th) in tag_dims.items():
        x, y, w, h = rects[s]
        if orientation == "TB":
            center = (x + w / 2 + tw / 2 + 0.06, y)
        else:
            center = (x + w / 2 - tw / 2, y + h / 2 + th / 2 + 0.02)
        tags[s] = (txt, center, tw, th)

    # Segmentos de cada arista (recortados en los bordes de los nodos).
    def clip_point(p, q, rect):
        x, y, w, h = rect
        dx, dy = q[0] - p[0], q[1] - p[1]
        if dx == 0 and dy == 0:
            return p
        t = min((w / 2) / abs(dx) if dx else math.inf, (h / 2) / abs(dy) if dy else math.inf)
        return (p[0] + dx * t, p[1] + dy * t)

    segs = {}
    for k, ch in paths.items():
        if len(ch) == 1:
            segs[k] = []
            continue
        pts = [xy[n] for n in ch]
        pts[0] = clip_point(pts[0], pts[1], rects[ch[0]])
        pts[-1] = clip_point(pts[-1], pts[-2], rects[ch[-1]])
        segs[k] = list(zip(pts, pts[1:]))

    # Aristas en ambos sentidos entre los mismos nodos: se curvan.
    bidir = {k for k in aut.edges if (k[1], k[0]) in aut.edges and k[0] != k[1]}

    # Colocación de etiquetas.
    def overlap(r1, r2, pad=0.03):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        ox = (w1 + w2) / 2 + pad - abs(x1 - x2)
        oy = (h1 + h2) / 2 + pad - abs(y1 - y2)
        return ox * oy if ox > 0 and oy > 0 else 0.0

    def seg_hits_rect(p, q, rect):
        x, y, w, h = rect
        xmin, xmax, ymin, ymax = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        t0, t1 = 0.0, 1.0
        dx, dy = q[0] - p[0], q[1] - p[1]
        for pp, qq in ((-dx, p[0] - xmin), (dx, xmax - p[0]), (-dy, p[1] - ymin), (dy, ymax - p[1])):
            if pp == 0:
                if qq < 0:
                    return False
            else:
                r = qq / pp
                if pp < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
                if t0 > t1:
                    return False
        return True

    obstacles = list(rects.values()) + [(start[0], start[1], 0.14, 0.14)]
    obstacles += [(c[0], c[1], tw, th) for _, c, tw, th in tags.values()]
    placed = {}
    order = sorted([k for k in aut.edges if labels[k] is not None], key=lambda k: -labels[k][1] * labels[k][2])
    for k in order:
        _, lw, lh = labels[k]
        cands = []
        if k[0] == k[1]:
            x, y, w, h = rects[k[0]]
            cands.append((x + w / 2 + 0.45 + lw / 2, y))
        else:
            sg = segs[k]
            fracs = [0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82]
            for si, (p, q) in enumerate(sg):
                for f in fracs:
                    cands.append((p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f))
        best, best_score = None, None
        for ci, c in enumerate(cands):
            r = (c[0], c[1], lw, lh)
            node_ov = sum(overlap(r, o) for o in obstacles)
            lab_ov = sum(overlap(r, o, pad=0.08) for o in placed.values())
            hits = 0
            for k2, sg2 in segs.items():
                if k2 == k:
                    continue
                for p, q in sg2:
                    if seg_hits_rect(p, q, r):
                        hits += 1
            score = (round(node_ov + lab_ov, 4) > 0, node_ov + lab_ov, hits, ci)
            if best_score is None or score < best_score:
                best, best_score = c, score
        placed[k] = (best[0], best[1], lw, lh)

    # Caja envolvente.
    xs, ys = [], []
    for x, y, w, h in list(rects.values()) + list(placed.values()) + [(start[0], start[1], 0.14, 0.14)]:
        xs += [x - w / 2, x + w / 2]
        ys += [y - h / 2, y + h / 2]
    for _, c, tw, th in tags.values():
        xs += [c[0] - tw / 2, c[0] + tw / 2]
        ys += [c[1] - th / 2, c[1] + th / 2]
    width, height = max(xs) - min(xs), max(ys) - min(ys)

    return {
        "orientation": orientation, "xy": xy, "rects": rects, "boxes": boxes, "paths": paths,
        "back": back, "bidir": bidir, "labels": labels, "placed": placed, "start": start, "tags": tags,
        "width": width, "height": height, "crossings": crossings, "nlayers": nlayers,
    }


def choose_layout(aut):
    lr = layout(aut, "LR")
    tb = layout(aut, "TB")
    if lr["width"] <= TEXT_WIDTH and lr["height"] <= tb["height"]:
        return lr
    if tb["width"] <= TEXT_WIDTH:
        return tb
    return lr if lr["width"] / TEXT_WIDTH < tb["width"] / TEXT_WIDTH else tb


# ----------------------------------------------------------------------------------------------------
# Escritura TikZ
# ----------------------------------------------------------------------------------------------------
def c(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"


def render(aut, L):
    ids = {n: f"s{i}" for i, n in enumerate(aut.order)}
    out = [
        f"% Diagrama de estados del autómata {aut.name}.",
        "% Generado por tools/thesis_automata_diagrams.py (simulador) a partir de automata.json y patl.json.",
        "% No editar a mano: volver a ejecutar el generador.",
        r"\anxbfit{%",
        r"\begin{tikzpicture}[anxb]",
    ]
    for n in aut.order:
        text, w, h = L["boxes"][n]
        style = {"gate": "anxb gate", "sample": "anxb sample", "det": "anxb det", "final": "anxb final"}[aut.kind(n)]
        if n in aut.trigger_tags:
            style += ", anxb trigger"
        x, y = L["xy"][n]
        out.append(rf"  \node[{style}, minimum width={c(w)}cm] ({ids[n]}) at ({c(x)},{c(y)}) {{{text}}};")
    # Las etiquetas de predicados se anclan al borde real del nodo (su posición calculada solo reserva sitio).
    for n, (txt, _, _, _) in L["tags"].items():
        if L["orientation"] == "TB":
            out.append(rf"  \node[anxb tag, anchor=west] at ([xshift=1.2pt]{ids[n]}.east) {{{txt}}};")
        else:
            out.append(rf"  \node[anxb tag, anchor=south east] at ([yshift=1pt]{ids[n]}.north east) {{{txt}}};")
    sx, sy = L["start"]
    out.append(rf"  \node[anxb start] (start) at ({c(sx)},{c(sy)}) {{}};")
    out.append(rf"  \draw[anxb edge] (start) -- ({ids[aut.initial]});")
    for k in aut.edges:
        u, v = k
        if u == v:
            loop = "loop right" if L["orientation"] == "TB" else "loop above"
            out.append(rf"  \draw[anxb edge] ({ids[u]}) edge[{loop}] ({ids[u]});")
            continue
        ch = L["paths"][k]
        if k in L["bidir"] and len(ch) == 2:
            out.append(rf"  \draw[anxb edge] ({ids[u]}) to[bend left=18] ({ids[v]});")
            continue
        parts = [f"({ids[ch[0]]})"]
        for d in ch[1:-1]:
            x, y = L["xy"][d]
            parts.append(f"({c(x)},{c(y)})")
        parts.append(f"({ids[ch[-1]]})")
        out.append(r"  \draw[anxb edge] " + " -- ".join(parts) + ";")
    for k, (x, y, _, _) in L["placed"].items():
        out.append(rf"  \node[anxb lbl] at ({c(x)},{c(y)}) {{{L['labels'][k][0]}}};")
    out.append(r"\end{tikzpicture}%")
    out.append("}")
    return "\n".join(out) + "\n"


STYLES = r"""% Estilos de los diagramas de estados del Anexo B.
% Generado por tools/thesis_automata_diagrams.py (simulador). No editar a mano.
\definecolor{unamazul}{HTML}{002B7A}
\definecolor{unamoro}{HTML}{C5911F}
\tikzset{
  anxb/.style={font=\tiny, line join=round},
  anxb state/.style={draw, rounded corners=2pt, align=center, inner sep=1.4pt, line width=0.45pt,
    minimum height=0.44cm},
  anxb det/.style={anxb state, fill=unamazul!9, draw=unamazul!60},
  anxb gate/.style={anxb state, fill=unamoro!30, draw=unamoro!80!black},
  anxb sample/.style={anxb state, fill=unamazul!9, draw=unamoro!80!black, dashed},
  anxb final/.style={anxb state, fill=black!3, draw=black!75, double, double distance=0.8pt},
  anxb trigger/.style={line width=1.3pt, draw=unamazul, solid},
  anxb tag/.style={font=\tiny\sffamily\bfseries, text=white, fill=unamazul, rounded corners=1pt,
    inner sep=0.9pt},
  anxb edge/.style={-{Stealth[length=3.2pt,width=2.4pt]}, draw=black!62, line width=0.42pt,
    rounded corners=3pt},
  anxb lbl/.style={font=\tiny, fill=white, inner sep=0.5pt, text=black!88, align=center},
  anxb start/.style={circle, fill=unamazul, inner sep=0pt, minimum size=3.4pt},
}
\providecommand{\anxbwrite}{\textcolor{unamazul}{$\scriptscriptstyle\blacksquare$}}
\providecommand{\anxbemit}{\textcolor{red!70!black}{$\scriptscriptstyle\blacktriangleright$}}
% Ajusta un diagrama al ancho de la línea y a una altura máxima, sin ampliarlo nunca.
\newsavebox{\anxbbox}
\providecommand{\anxbmaxheight}{0.88\textheight}
\providecommand{\anxbfit}[1]{%
  \sbox{\anxbbox}{#1}%
  \ifdim\wd\anxbbox>\linewidth
    \sbox{\anxbbox}{\resizebox{\linewidth}{!}{\usebox{\anxbbox}}}%
  \fi
  \ifdim\ht\anxbbox>\anxbmaxheight
    \sbox{\anxbbox}{\resizebox{!}{\anxbmaxheight}{\usebox{\anxbbox}}}%
  \fi
  \usebox{\anxbbox}}
"""

LEGEND = r"""% Leyenda de los diagramas de estados del Anexo B.
% Generado por tools/thesis_automata_diagrams.py (simulador). No editar a mano.
\begin{tikzpicture}[anxb, every node/.append style={font=\scriptsize}]
  \begin{scope}[every node/.style={font=\tiny}]
    \node[anxb start] (st) at (0,0) {};
    \node[anxb det, minimum width=1.15cm] (ini) at (1.05,0) {\texttt{LOAD\_}\\\texttt{PARAMS}};
    \draw[anxb edge] (st) -- (ini);
    \node[anxb gate, minimum width=1.55cm] at (0.9,-1.05) {\texttt{DORMANT}\\\textsf{\itshape $\sim$\,shock\_arrival}};
    \node[anxb det, minimum width=1.55cm] at (0.9,-2.05) {\texttt{IDLE}\\\textsf{\itshape cash}};
    \node[anxb sample, minimum width=1.55cm] at (0.9,-3.05) {\texttt{FETCH\_}\\\textsf{\itshape $\sim$\,growth\_rate}};
  \end{scope}
  \node[anchor=west, text width=5.6cm] at (2.0,0) {Estado inicial (el punto marca la entrada al autómata).};
  \node[anchor=west, text width=5.6cm] at (2.0,-1.05) {Compuerta probabilística: muestrea la distribución indicada y bifurca según el caso que coincide.};
  \node[anchor=west, text width=5.6cm] at (2.0,-2.05) {Paso determinista: transición incondicional, cálculo o lectura del parámetro indicado.};
  \node[anchor=west, text width=5.6cm] at (2.0,-3.05) {Muestreo sin bifurcación: toma una muestra y continúa por un único caso.};
  \begin{scope}[every node/.style={font=\tiny}]
    \node[anxb final, minimum width=1.3cm] at (8.9,0) {\texttt{DONE}};
    \node[anxb det, anxb trigger, minimum width=1.3cm] (tr) at (8.9,-1.05) {\texttt{IDLE}};
    \node[anxb tag] at ($(tr.north east)+(-0.05,0)$) {SLP};
    \coordinate (a) at (8.2,-2.05); \coordinate (b) at (9.6,-2.05);
    \draw[anxb edge] (a) -- (b);
    \node[anxb lbl] at (8.9,-2.05) {$(0.3, 0.45]$};
    \node at (8.9,-3.05) {\anxbwrite\quad\anxbemit};
  \end{scope}
  \node[anchor=west, text width=5.3cm] at (10.0,0) {Estado final (doble borde).};
  \node[anchor=west, text width=5.3cm] at (10.0,-1.05) {Estado disparador: borde grueso y abreviatura de los predicados PATL que se verifican al alcanzarlo.};
  \node[anchor=west, text width=5.3cm] at (10.0,-2.05) {Arista con la condición del caso sobre el valor evaluado; sin etiqueta, la transición es incondicional.};
  \node[anchor=west, text width=5.3cm] at (10.0,-3.05) {Acciones de la transición: \anxbwrite\ escritura en agentes o entorno; \anxbemit\ emisión de una señal.};
\end{tikzpicture}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--distributions", default="distributions_1_base.json",
                    help="archivo de distribuciones del que se toman las familias (iguales en los cuatro regímenes)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    automata = json.loads((args.config / "automata.json").read_text(encoding="utf-8"))
    patl = json.loads((args.config / "patl.json").read_text(encoding="utf-8"))
    dists = json.loads((args.config / args.distributions).read_text(encoding="utf-8"))
    families = {d["distribution_name"]: d["family"] for d in dists}
    triggers = {}
    for o in patl["observations"]:
        triggers.setdefault((o["automaton_name"], o["trigger_state"]), []).extend(
            p["predicate_id"] for p in o["predicates"])
    abbr = load_abbreviations()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "estilos.tex").write_text(STYLES, encoding="utf-8")
    (args.out / "leyenda.tex").write_text(LEGEND, encoding="utf-8")
    print(f"{'autómata':32s} orient capas cruces retro  ancho   alto")
    for spec in automata:
        aut = Automaton(spec, families, triggers, abbr)
        L = choose_layout(aut)
        (args.out / f"{aut.name}.tex").write_text(render(aut, L), encoding="utf-8")
        print(f"{aut.name:32s} {L['orientation']:6s} {L['nlayers']:5d} {L['crossings']:6d} {len(L['back']):5d}"
              f" {L['width']:6.2f} {L['height']:6.2f}")


if __name__ == "__main__":
    main()
