import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_summary(file_path: Path):
    """Processes the *_summary.csv file containing performance and buildtime metrics."""
    print("\n" + "=" * 60)
    print(f"📊 [1/3] DESCRIPTIVE ANALYSIS OF SUMMARY: {file_path.name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"❌ File not found at path: {file_path}")
        return

    df = pd.read_csv(file_path)

    # 1. Simulator Performance (DES vs PATL)
    print("\n⏱️ Performance and Simulation Duration (Seconds):")
    perf_df = df[df["category"] == "performance"]
    if not perf_df.empty:
        perf_pivot = perf_df.pivot(
            index="run_id", columns="key", values="value"
        ).astype(float)
        stats = perf_pivot.describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ]
        print(stats.to_string())
    else:
        print("  No performance metrics were found.")

    # 2. Static Configuration Summary (Buildtime)
    print("\n🧱 Initialization Structure (Buildtime - Modes/Constants):")
    build_df = df[df["category"] == "buildtime_summary"]
    if not build_df.empty:
        build_summary = (
            build_df.groupby("key")["value"]
            .agg(["min", "max", "count"])
            .rename(columns={"min": "Value", "count": "Total Runs"})
        )
        print(build_summary.to_string())
    else:
        print("  No buildtime metrics were found.")


def analyze_des(file_path: Path):
    """Processes the *_des.csv file that records transactional telemetry of events and actions."""
    print("\n" + "=" * 60)
    print(f"🎲 [2/3] DESCRIPTIVE ANALYSIS OF DES (RUNTIME): {file_path.name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"❌ File not found at path: {file_path}")
        return

    df = pd.read_csv(file_path)
    sections = df["runtime_section"].unique()

    for section in sections:
        print(f"\n🔹 Execution Section: {section.upper()}")
        sec_df = df[df["runtime_section"] == section].copy()
        sec_df["metric_identity"] = (
            sec_df["entity_key"] + " [" + sec_df["metric_subkey"] + "]"
        )

        pivot_df = sec_df.pivot_table(
            index="run_id",
            columns="metric_identity",
            values="execution_value",
            aggfunc="sum",
            fill_value=0,
        )

        stats = pivot_df.describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ]
        stats["total_accumulated"] = pivot_df.sum()
        print(stats.to_string())


def analyze_patl(file_path: Path):
    """Processes the *_patl.csv file with the results of the temporal logic verification."""
    print("\n" + "=" * 60)
    print(f"🔍 [3/3] DESCRIPTIVE ANALYSIS OF PATL VERIFIER: {file_path.name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"❌ File not found at path: {file_path}")
        return

    df = pd.read_csv(file_path)

    # 1. Distribution of Qualitative Results
    print("\n📈 Frequency of Semantic Verification States:")
    result_counts = df.groupby(["automaton_name", "predicate_id", "result"])[
        "run_id"
    ].count()
    result_pct = df.groupby(["automaton_name", "predicate_id", "result"])[
        "run_id"
    ].count() / df.groupby(["automaton_name", "predicate_id"])["run_id"].count()

    summary_qual = pd.DataFrame(
        {"Frequency (Cases)": result_counts, "Proportion (%)": result_pct * 100}
    )
    print(summary_qual.to_string())

    # 2. Quantitative analysis of the reached probability values (p_value)
    print("\n🎲 Statistical Distribution of Probability Values (p_value):")
    p_stats = df.groupby(["automaton_name", "predicate_id"])["p_value"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    print(p_stats.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Descriptive Analyzer of DES/PATL Telemetry for Adjusted Project Structure."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario prefix (e.g., runway_risk_base, runway_risk_crisis)",
    )
    # By default, we calculate the relative path going up out of analytics/ and entering data/
    default_data_dir = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument(
        "--dir",
        default=str(default_data_dir),
        help=f"Directory where CSVs are located (Default: {default_data_dir})",
    )

    args = parser.parse_args()
    base_path = Path(args.dir)
    scenario = args.scenario

    summary_file = base_path / f"{scenario}_summary.csv"
    des_file = base_path / f"{scenario}_des.csv"
    patl_file = base_path / f"{scenario}_patl.csv"

    print(f"🚀 Launching deep processing for scenario: {scenario}")
    print(f"📂 Searching files in: {base_path.resolve()}\n")

    analyze_summary(summary_file)
    analyze_des(des_file)
    analyze_patl(patl_file)

    print("\n" + "=" * 60)
    print("🏁 ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()