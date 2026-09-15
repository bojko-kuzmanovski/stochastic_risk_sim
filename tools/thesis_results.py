"""
Genera los cuadros, figuras y macros de resultados de la tesis a partir de las salidas de main.py.

Herramienta de autoría del caso de estudio (no la importa el motor). Lee <prefijo>_<régimen>_patl.csv,
_des.csv y _summary.csv y escribe archivos LaTeX listos para \\input. Todo número se escribe con cuatro
decimales.

Uso:
    python tools/thesis_results.py --data data --prefix tesis --out <directorio>
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = [("1_base", "Base"), ("2_bull", "Bull"), ("3_bear", "Bear"), ("4_crisis", "Crisis")]
LETTER = {"Base": "Base", "Bull": "Bull", "Bear": "Bear", "Crisis": "Crisis"}

# Abreviatura, identificador y grupo temático, en el orden de presentación.
PREDICATES = [
    ("DSM", "PR_STOCHASTIC_DEAD_VIA_SENTIMENT", "Supervivencia"),
    ("SMD", "PR_STOCHASTIC_MORTALITY_VIA_DRIFT", "Supervivencia"),
    ("LPP", "PR_STOCHASTIC_LAYOFFS_VIA_PANIC", "Supervivencia"),
    ("ISM", "PR_STOCHASTIC_INVESTMENT_VIA_SENTIMENT", "Financiamiento"),
    ("CCA", "PR_STOCHASTIC_CAPITAL_CRUNCH_VIA_APPETITE", "Financiamiento"),
    ("SIP", "PR_STOCHASTIC_EXIT_VIA_PATIENCE", "Financiamiento"),
    ("IDV", "PR_STOCHASTIC_DEMAND_STABILITY_VIA_PROFILE", "Clientes"),
    ("HCV", "PR_STOCHASTIC_HYPER_CHURN_VIA_PROFILE", "Clientes"),
    ("SCA", "PR_STOCHASTIC_SHOCK_CASCADE_VIA_ARRIVAL", "Choques y regulación"),
    ("SLP", "PR_STOCHASTIC_LAYOFFS_VIA_SHOCK", "Choques y regulación"),
    ("IAA", "PR_STOCHASTIC_IMMUNITY_VIA_AUDIT", "Choques y regulación"),
    ("VBB", "PR_STOCHASTIC_VULNERABILITY_VIA_BURDEN", "Choques y regulación"),
]
ABBR = {pid: ab for ab, pid, _ in PREDICATES}

# Observaciones: (autómata, estado disparador) con su etiqueta.
OBSERVATIONS = [
    ("shock_lifecycle", "DORMANT", "Choque latente"),
    ("runway_lifecycle", "EVALUATE_MARKET_SURVIVAL", "Evaluación de supervivencia"),
    ("runway_critical", "ASSESS_SURVIVAL_CHANCE", "Runway crítico"),
    ("compliance_monitor", "READ_REGULATOR_CHANNEL", "Monitoreo regulatorio"),
    ("client_retention_cycle", "STOCHASTIC_RETENTION_FILTER", "Revisión de cliente"),
    ("client_retention_cycle", "PREPARE_CHURN_EMIT", "Salida de cliente"),
    ("investment_decision", "STOCHASTIC_SENTIMENT_GATE", "Decisión de inversión"),
    ("headcount_controller", "IDLE", "Revisión de plantilla"),
    ("investor_exit", "DETERMINE_EXIT_REASON", "Salida de inversor"),
]

# Métricas operativas: (etiqueta, autómata, estados finales sumados).
OPERATIONS = [
    ("Muertes con refundación", "runway_lifecycle", ["DEAD"]),
    ("Ciclos de runway cerrados", "runway_lifecycle", ["DONE_CYCLE"]),
    ("Despidos de emergencia", "runway_critical", ["LAYOFFS_EXECUTED"]),
    ("Inversiones concretadas", "investment_decision", ["INVESTED"]),
    ("Inversiones declinadas", "investment_decision", ["PASSED"]),
    ("Salidas de inversores", "investor_exit", ["EXITED"]),
    ("Clientes perdidos", "client_retention_cycle", ["CHURNED"]),
    ("Clientes repuestos", "client_replacement", ["REPLACED"]),
    ("Penalizaciones regulatorias", "compliance_monitor", ["PENALIZED"]),
    ("Cascadas de choques", "shock_lifecycle", ["MARKET_CASCADE", "REGULATORY_CASCADE", "COMPOUND_CASCADE"]),
]


def f4(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:.4f}"


def prop(x):
    return f"{x:.4f}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def load(data, prefix):
    patl, des, summ = [], [], []
    for key, lab in REGIMES:
        p = data / f"{prefix}_{key}_patl.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"value": str})
        df["regime"] = lab
        patl.append(df)
        d = pd.read_csv(data / f"{prefix}_{key}_des.csv")
        d["regime"] = lab
        des.append(d)
        s = pd.read_csv(data / f"{prefix}_{key}_summary.csv")
        s["regime"] = lab
        summ.append(s)
    patl = pd.concat(patl, ignore_index=True)
    patl["abbr"] = patl.predicate_id.map(ABBR)
    patl["v"] = pd.to_numeric(patl["value"], errors="coerce")
    return patl, pd.concat(des, ignore_index=True), pd.concat(summ, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--prefix", default="tesis")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patl, des, summ = load(Path(args.data), args.prefix)
    regimes = [lab for _, lab in REGIMES if lab in set(patl.regime)]
    valid = patl[patl.result != "ERROR"].copy()
    macros = [f"% Generado por tools/thesis_results.py. Regímenes: {', '.join(regimes)}."]

    runs = summ.groupby("regime").run_id.nunique()
    errors = patl[patl.result == "ERROR"].groupby("regime").size()
    macros.append(macro("resErroresTotales", int((patl.result == "ERROR").sum())))
    macros.append(macro("resFilasPATL", int(len(patl))))
    for r in regimes:
        macros.append(macro(f"resCorridas{LETTER[r]}", int(runs.get(r, 0))))

    # ------------------------------------------------------------------ estadísticos por predicado
    base = valid[valid.memory_k == valid.memory_k.min()]
    spec = patl.groupby("abbr").agg(op=("operator", "first"), bound=("bound", "first"))
    # Media por corrida para el error estándar entre corridas.
    per_run = base.groupby(["regime", "abbr", "run_id"]).v.mean().reset_index()
    se = per_run.groupby(["regime", "abbr"]).v.agg(lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan"))
    stats = base.groupby(["regime", "abbr"]).agg(
        n=("v", "size"), mean=("v", "mean"), sd=("v", "std"), vmin=("v", "min"), med=("v", "median"),
        vmax=("v", "max"), sat=("result", lambda x: (x == "SATISFIED").mean()))
    max_se = float(np.nanmax(se.values)) if len(se) else float("nan")
    macros.append(macro("resErrorEstandarMaximo", f4(max_se)))

    for (r, ab), row in stats.iterrows():
        tag = f"{ab}{LETTER[r]}"
        macros += [macro(f"res{tag}Media", f4(row["mean"])), macro(f"res{tag}Sat", prop(row["sat"])),
                   macro(f"res{tag}N", int(row["n"]))]

    # Cuadro: valor medio (porcentaje satisfecho) por predicado y régimen.
    lines = [r"\begin{tabular}{l l c " + " ".join(["r r"] * len(regimes)) + "}", r"\toprule",
             r"& & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{r}}}}}" for r in regimes) + r" \\",
             " ".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(regimes))),
             r"\textbf{Grupo} & \textbf{Pred.} & \textbf{Cota} & " + " & ".join([r"\textbf{Valor} & \textbf{Sat.}"] * len(regimes)) + r" \\",
             r"\midrule"]
    last_group = None
    for ab, pid, group in PREDICATES:
        if ab not in spec.index:
            continue
        op = spec.loc[ab, "op"].replace(">=", r"$\geq$").replace("<=", r"$\leq$")
        cells = []
        for r in regimes:
            if (r, ab) in stats.index:
                s = stats.loc[(r, ab)]
                cells.append(rf"{f4(s['mean'])} & {prop(s['sat'])}")
            else:
                cells.append("--- & ---")
        gcell = group if group != last_group else ""
        if last_group is not None and group != last_group:
            lines.append(r"\addlinespace[2pt]")
        last_group = group
        lines.append(rf"{gcell} & {ab} & {op} {float(spec.loc[ab, 'bound']):.4f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "res_valores.tex").write_text("\n".join(lines) + "\n")

    # Mapa de calor en TikZ: celdas coloreadas por el valor medio.
    order = [ab for ab, _, _ in PREDICATES if ab in spec.index]
    tik = [r"\begin{tikzpicture}[x=1.9cm, y=0.55cm]"]
    for j, ab in enumerate(order):
        y = -j
        tik.append(rf"\node[anchor=east, font=\scriptsize] at (-0.05, {y}) {{{ab}}};")
        for i, r in enumerate(regimes):
            if (r, ab) not in stats.index:
                continue
            v = float(stats.loc[(r, ab), "mean"])
            shade = int(round(100 * v))
            txt = "white" if v > 0.55 else "black"
            tik.append(rf"\fill[unamazul!{shade}!white] ({i}, {y - 0.5}) rectangle ({i + 1}, {y + 0.5});")
            tik.append(rf"\node[font=\scriptsize, text={txt}] at ({i + 0.5}, {y}) {{{f4(v)}}};")
    for i, r in enumerate(regimes):
        tik.append(rf"\node[font=\scriptsize\bfseries] at ({i + 0.5}, 1) {{{r}}};")
    tik.append(rf"\draw[black!40] (0, 0.5) rectangle ({len(regimes)}, {-len(order) + 0.5});")
    tik.append(r"\end{tikzpicture}")
    (out / "res_mapa.tex").write_text("\n".join(tik) + "\n")

    # Figura de puntos: valor medio por régimen frente a la cota, un panel por predicado.
    gp = [r"\begin{tikzpicture}",
          r"\begin{groupplot}[group style={group size=4 by 3, horizontal sep=1.1cm, vertical sep=1.35cm},",
          r"  width=4.2cm, height=3.4cm, ymin=0, ymax=1, xmin=0.5, xmax=%d," % len(regimes) + (0.5 > 0) * "",
          r"  xtick={%s}, xticklabels={%s}, x tick label style={font=\tiny, rotate=35, anchor=east}," % (
              ",".join(str(i + 1) for i in range(len(regimes))), ",".join(regimes)),
          r"  ytick={0,0.5,1}, y tick label style={font=\tiny}, title style={font=\scriptsize\bfseries}]"]
    gp[2] = r"  width=4.2cm, height=3.4cm, ymin=0, ymax=1, xmin=0.5, xmax=%.1f," % (len(regimes) + 0.5)
    for ab in order:
        pts = " ".join(f"({i + 1},{float(stats.loc[(r, ab), 'mean']):.4f})" for i, r in enumerate(regimes) if (r, ab) in stats.index)
        b = float(spec.loc[ab, "bound"])
        gp.append(rf"\nextgroupplot[title={{{ab} ({spec.loc[ab, 'op'].replace('>=', '$\\geq$').replace('<=', '$\\leq$')} {b:.4f})}}]")
        gp.append(rf"\addplot[unamoro, dashed, thick, domain=0.5:{len(regimes) + 0.5}] {{{b:.4f}}};")
        gp.append(rf"\addplot[unamazul, thick, mark=*, mark size=1.6pt] coordinates {{{pts}}};")
    gp += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (out / "res_puntos.tex").write_text("\n".join(gp) + "\n")

    # ------------------------------------------------------------------ memoria
    kmin, kmax = int(valid.memory_k.min()), int(valid.memory_k.max())
    mem_lines = []
    if kmax > kmin:
        keys = ["regime", "run_id", "automaton_name", "trigger_state", "agent_id", "abbr"]
        tmp = valid.copy()
        tmp["occ"] = tmp.groupby(keys + ["memory_k"]).cumcount()
        wide = tmp.pivot_table(index=keys + ["occ"], columns="memory_k", values="v").dropna()
        wide["gain"] = wide[kmax] - wide[kmin]
        eff = wide.groupby(["abbr", "regime"]).agg(k1=(kmin, "mean"), kk=(kmax, "mean"),
                                                   share=("gain", lambda g: (g > 1e-9).mean()),
                                                   gain=("gain", "mean"), gmax=("gain", "max"))
        with_gain = sorted({ab for (ab, _), row in eff.iterrows() if row.gmax > 1e-9})
        no_gain = [ab for ab in order if ab not in with_gain]
        macros.append(macro("resPredicadosSinGananciaMemoria", len(no_gain)))
        mem_lines = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
                     rf"\textbf{{Pred.}} & \textbf{{Régimen}} & \textbf{{$k={kmin}$}} & \textbf{{$k={kmax}$}} & \textbf{{Diferencia}} & \textbf{{Prop. con mejora}} & \textbf{{Mejora máx.}} \\",
                     r"\midrule"]
        for ab in with_gain:
            for r in regimes:
                if (ab, r) not in eff.index:
                    continue
                row = eff.loc[(ab, r)]
                mem_lines.append(rf"{ab} & {r} & {f4(row.k1)} & {f4(row.kk)} & {f4(row.gain)} & {prop(row.share)} & {f4(row.gmax)} \\")
                tag = f"{ab}{LETTER[r]}"
                macros += [macro(f"res{tag}MemUno", f4(row.k1)), macro(f"res{tag}MemDos", f4(row.kk)),
                           macro(f"res{tag}MemGanancia", f4(row.gain)), macro(f"res{tag}MemMejora", prop(row.share)),
                           macro(f"res{tag}MemMax", f4(row.gmax))]
        mem_lines += [r"\bottomrule", r"\end{tabular}"]
        # Estrategias aleatorizadas: fracción de filas cuyo valor lo alcanzó una mezcla.
        mixed = valid.groupby("abbr").strategy.apply(lambda s: (s == "mixed").mean())
        for ab, share in mixed.items():
            macros.append(macro(f"res{ab}Mezcla", prop(share)))
    (out / "res_memoria.tex").write_text("\n".join(mem_lines) + "\n")

    # ------------------------------------------------------------------ cobertura y operación
    snaps = des[des.runtime_section == "snapshot_captured"]
    cov = [r"\begin{tabular}{l l " + " ".join(["r"] * len(regimes)) + "}", r"\toprule",
           r"\textbf{Observación} & \textbf{Predicados} & " + " & ".join(rf"\textbf{{{r}}}" for r in regimes) + r" \\",
           r"\midrule"]
    obs_preds = patl.groupby(["automaton_name", "trigger_state"]).abbr.agg(lambda s: ", ".join(sorted(set(s))))
    for aut, st, label in OBSERVATIONS:
        cells = []
        for r in regimes:
            sel = snaps[(snaps.regime == r) & (snaps.entity_key == aut) & (snaps.metric_subkey == st)]
            total = sel.execution_value.sum()
            cells.append(f4(total / runs[r]))
        preds = obs_preds.get((aut, st), "")
        cov.append(rf"{label} & {preds} & " + " & ".join(cells) + r" \\")
    cov += [r"\bottomrule", r"\end{tabular}"]
    (out / "res_cobertura.tex").write_text("\n".join(cov) + "\n")

    autex = des[des.runtime_section == "automaton_execution"].copy()
    autex["state"] = autex.metric_subkey.str.split(":").str[-1]
    op_lines = [r"\begin{tabular}{l " + " ".join(["r"] * len(regimes)) + "}", r"\toprule",
                r"\textbf{Media por corrida} & " + " & ".join(rf"\textbf{{{r}}}" for r in regimes) + r" \\", r"\midrule"]
    for label, aut, states in OPERATIONS:
        cells = []
        for r in regimes:
            sel = autex[(autex.regime == r) & (autex.entity_key == aut) & (autex.state.isin(states))]
            m = sel.execution_value.sum() / runs[r]
            cells.append(f4(m))
            macros.append(macro("resOp" + "".join(w.capitalize() for w in label.split()).replace("ó", "o").replace("é", "e") + LETTER[r], f4(m)))
        op_lines.append(rf"{label} & " + " & ".join(cells) + r" \\")
    op_lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "res_operacion.tex").write_text("\n".join(op_lines) + "\n")

    perf = summ[summ.category == "performance"].copy()
    perf["value"] = pd.to_numeric(perf.value)
    ev = des[des.runtime_section == "event_generated"]
    run_lines = [r"\begin{tabular}{l r r r r r}", r"\toprule",
                 r"\textbf{Régimen} & \textbf{Corridas} & \textbf{Instantáneas} & \textbf{Filas PATL} & \textbf{DES (s)} & \textbf{PATL (s)} \\",
                 r"\midrule"]
    for r in regimes:
        n_snap = snaps[snaps.regime == r].execution_value.sum() / runs[r]
        rows_r = len(patl[patl.regime == r])
        t = perf[perf.regime == r].groupby("key").value.mean()
        run_lines.append(rf"{r} & {int(runs[r])} & {f4(n_snap)} & {rows_r} & {f4(t.get('elapsed_des_sec', float('nan')))} & {f4(t.get('elapsed_patl_sec', float('nan')))} \\")
        macros += [macro(f"resInstantaneas{LETTER[r]}", f4(n_snap)), macro(f"resTiempoPATL{LETTER[r]}", f4(t.get('elapsed_patl_sec', float('nan'))))]
    run_lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "res_corridas.tex").write_text("\n".join(run_lines) + "\n")

    # ------------------------------------------------------------------ anexo C: detalle completo
    anx = [r"\bigskip", r"\section{Resultados numéricos por régimen}\label{anx:resultados}", "",
           r"\medskip \noindent",
           r"Este anexo reporta, para cada régimen y cota de memoria, los estadísticos de los valores de los doce predicados sobre todas las instantáneas verificadas: número de instantáneas ($n$), media, desviación estándar, mínimo, mediana, máximo y proporción de instantáneas en que se satisface la cota. La media del Cuadro de resultados principal corresponde a la columna de media con $k = %d$. Las corridas de un régimen son réplicas independientes con semillas distintas, por lo que sus estadísticos describen la variabilidad entre trayectorias en el sentido del análisis de salidas por réplicas \\parencite{banks2005, law2015simulation}; las instantáneas de una misma corrida, en cambio, están correlacionadas en el tiempo, y la dispersión entre instantáneas no debe leerse como error de estimación." % kmin, ""]
    full = valid.groupby(["regime", "memory_k", "abbr"]).agg(
        n=("v", "size"), mean=("v", "mean"), sd=("v", "std"), vmin=("v", "min"), med=("v", "median"),
        vmax=("v", "max"), sat=("result", lambda x: (x == "SATISFIED").mean()))
    for r in regimes:
        anx += [r"\begin{table}[H]", r"\centering",
                rf"\caption{{\textit{{\textbf{{Estadísticos de los predicados en el régimen {r}.}}}} Errores de verificación: {int(errors.get(r, 0))}.}}",
                rf"\label{{tab:anxc-{r.lower()}}}", r"\scriptsize",
                r"\begin{tabular}{l c r r r r r r r}", r"\toprule",
                r"\textbf{Pred.} & \textbf{$k$} & \textbf{$n$} & \textbf{Media} & \textbf{Desv.} & \textbf{Mín.} & \textbf{Mediana} & \textbf{Máx.} & \textbf{Prop. S} \\",
                r"\midrule"]
        for ab in order:
            for k in sorted(valid.memory_k.unique()):
                if (r, k, ab) not in full.index:
                    continue
                s = full.loc[(r, k, ab)]
                sd = 0.0 if pd.isna(s.sd) else s.sd
                anx.append(rf"{ab if k == kmin else ''} & {k} & {int(s.n)} & {f4(s['mean'])} & {f4(sd)} & {f4(s.vmin)} & {f4(s.med)} & {f4(s.vmax)} & {prop(s.sat)} \\")
        anx += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "anexo_c_resultados.tex").write_text("\n".join(anx) + "\n")

    (out / "res_macros.tex").write_text("\n".join(macros) + "\n")
    print("regímenes:", regimes, "| corridas:", dict(runs), "| errores:", int((patl.result == 'ERROR').sum()))
    print("archivos:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
