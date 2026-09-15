"""
Reporte legible de una traza JSONL generada con main.py --trace.

Ejemplos:
  python3 analytics/trace_report.py data/traces/<salida>_run1.jsonl
  python3 analytics/trace_report.py data/traces/<salida>_run1.jsonl --automaton <nombre_del_autómata>
  python3 analytics/trace_report.py data/traces/<salida>_run1.jsonl --predicate <predicate_id> --limit 2
"""
import argparse
import json
from collections import Counter, defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_des(records):
    section("DES: calendario")
    instants = [r for r in records if r["c"] == "des" and r["e"] == "instant"]
    start = next((r for r in records if r["e"] == "run_start"), None)
    end = next((r for r in records if r["e"] == "end"), None)
    if start:
        print(f"semilla={start['seed']}  T_max={start['max_time']}  K={start['queue_batch']}  memoria={start['memory']}")
    by_cat = Counter(cat for r in instants for _, _, cat in r["events"])
    print(f"instantes={len(instants)}  eventos despachados={sum(r['dispatched'] for r in instants)}  por categoría={dict(by_cat)}")
    if end:
        print(f"T final={end['final_T']}")
    by_signal = Counter(sig for r in instants for _, sig, _ in r["events"])
    for sig, n in by_signal.most_common():
        print(f"   {sig:32s} {n}")


def report_agents(records):
    section("Agentes: sesiones atómicas")
    ends = [r for r in records if r["c"] == "agents" and r["e"] == "session_end"]
    table = defaultdict(Counter)
    for r in ends:
        table[(r["agent"].rsplit("_", 1)[0], r["automaton"])][r["final"]] += 1
    for (atype, aut), finals in sorted(table.items()):
        print(f"   {atype:14s} {aut:30s} {dict(finals)}")


def report_automata(records, automaton=None):
    section("Autómatas: reparto de casos por transición" + (f" ({automaton})" if automaton else ""))
    trans = [r for r in records if r["c"] == "automata" and r["e"] == "transition"
             and (automaton is None or r["automaton"] == automaton)]
    table = defaultdict(Counter)
    values = defaultdict(list)
    for r in trans:
        key = (r["automaton"], r["frm"], r["source"])
        table[key][r["to"]] += 1
        if isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool):
            values[key].append(r["value"])
    for (aut, frm, source), tos in sorted(table.items()):
        total = sum(tos.values())
        shares = ", ".join(f"{to} {n / total:.0%}" for to, n in tos.most_common())
        vals = values[(aut, frm, source)]
        rng = f"  valores [{min(vals):.4f}, {max(vals):.4f}]" if vals else ""
        print(f"   {aut}::{frm} <{source}> n={total}: {shares}{rng}")
    if automaton:
        calls = [r for r in records if r["c"] == "automata" and r["e"] == "call" and r["automaton"] == automaton]
        writes = Counter((r["method"], tuple(r["args"][:2]) if isinstance(r["args"], list) else None)
                         for r in calls if str(r["method"]).startswith("write"))
        if writes:
            print("   escrituras:")
            for (method, target), n in writes.most_common():
                print(f"      {method} {target} x{n}")


def report_snapshots(records):
    section("Instantáneas por estado disparador")
    caps = Counter((r["automaton"], r["state"]) for r in records if r["c"] == "snapshots")
    for (aut, state), n in caps.most_common():
        print(f"   {aut}::{state}  {n}")


def report_patl(records, predicate=None, limit=3):
    section("PATL: predicados verificados" + (f" ({predicate})" if predicate else ""))
    results = [r for r in records if r["c"] == "patl" and r["e"] in ("predicate_result", "predicate_error")
               and (predicate is None or r["predicate"] == predicate)]
    table = defaultdict(list)
    errors = Counter()
    for r in results:
        if r["e"] == "predicate_error":
            errors[(r["predicate"], r["reason"])] += 1
        else:
            table[(r["predicate"], r["memory_k"])].append(r)
    for (pid, k), rows in sorted(table.items()):
        vals = [r["value"] for r in rows]
        sat = sum(r["result"] == "SATISFIED" for r in rows) / len(rows)
        degenerate = sum(v <= 1e-9 or v >= 1 - 1e-9 for v in vals) / len(vals)
        print(f"   {pid:44s} k={k} n={len(rows):4d} min={min(vals):.4f} max={max(vals):.4f} "
              f"media={sum(vals) / len(vals):.4f} SAT={sat:.0%} degenerados={degenerate:.0%}")
    for (pid, reason), n in errors.items():
        print(f"   {pid:44s} ERROR x{n}: {reason}")

    if predicate is None:
        return
    section(f"Detalle de los primeros {limit} juegos de {predicate}")
    shown = 0
    current = None
    for r in records:
        if r["c"] == "patl" and r["e"] == "predicate_start" and r["predicate"] == predicate:
            if shown >= limit:
                break
            shown += 1
            current = True
            print(f"\n-- instantánea: {r['snapshot_agent']} en {r['snapshot_automaton']}::{r['snapshot_state']}  k={r['memory_k']}")
            print(f"   {r['type']} delta={r['depth']} {r['operator']} {r['bound']} cuantificador={r['quantifier']}")
            print(f"   coalición={r['coalition']}  ({r['coalition_ext']})")
            print(f"   adversarios={r['adversaries']}  ({r['adversary_ext']})")
            print(f"   sesión en curso={r['pending']}  secuencias={r['sequences']}")
        elif current and r["c"] == "patl_rounds":
            if r["e"] == "round":
                outs = "; ".join(f"p={o['p']:.4f} -> {o['last']}" for o in r["outcomes"][:6])
                more = f" (+{len(r['outcomes']) - 6})" if len(r["outcomes"]) > 6 else ""
                print(f"      ronda {r['round']} acciones={r['actions']} valor={r['value']:.4f}: {outs}{more}")
            elif r["e"] == "adversary_choice":
                print(f"      ronda {r['round']} adversario {r['extremum']} sobre {r['choices']} -> {r['values']} = {r['value']:.4f}")
        elif current and r["c"] == "patl" and r["e"] == "sequence_value" and r["predicate"] == predicate:
            print(f"   secuencia {r['sequence']} -> {r['value']:.4f} ({r['nodes']} nodos)")
        elif current and r["c"] == "patl" and r["e"] == "predicate_result" and r["predicate"] == predicate:
            print(f"   RESULTADO {r['value']:.4f} {r['operator']} {r['bound']} -> {r['result']}")
            current = None


def main():
    parser = argparse.ArgumentParser(description="Reporte de una traza JSONL de simulación y verificación")
    parser.add_argument("trace")
    parser.add_argument("--automaton", default=None, help="Detalla transiciones y escrituras de un autómata")
    parser.add_argument("--predicate", default=None, help="Detalla los juegos PATL de un predicado")
    parser.add_argument("--limit", type=int, default=3, help="Juegos a detallar con --predicate")
    args = parser.parse_args()

    records = load(args.trace)
    components = Counter(r["c"] for r in records)
    print(f"{args.trace}: {len(records)} registros {dict(components)}")
    if components.get("des"):
        report_des(records)
    if components.get("agents"):
        report_agents(records)
    if components.get("automata"):
        report_automata(records, args.automaton)
    if components.get("snapshots"):
        report_snapshots(records)
    if components.get("patl"):
        report_patl(records, args.predicate, args.limit)


if __name__ == "__main__":
    main()
