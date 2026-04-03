"""
Geopolitical Intelligence Pipeline
====================================
Main entry point. Orchestrates:
  1. Data collection (GDELT or mock)
  2. Preprocessing
  3. Graph construction
  4. Network analysis
  5. Event classification (ML)
  6. LLM narrative summarization
  7. Report export

Usage:
  # Full pipeline with mock data (no API keys needed)
  python main.py

  # With GDELT data
  python main.py --source gdelt --start 2024-01-01 --end 2024-03-31

  # With LLM summaries
  python main.py --llm

  # Launch dashboard only
  python main.py --dashboard
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.gdelt_collector import collect_gdelt_range, generate_mock_data, preprocess
from analysis.graph_builder import (
    build_graph, compute_metrics, compute_network_stats,
    build_temporal_graphs, get_bilateral_summary
)
from analysis.narrator import GeopoliticalNarrator


def parse_args():
    parser = argparse.ArgumentParser(description="Geopolitical Intelligence Pipeline")
    parser.add_argument("--source", choices=["gdelt", "mock"], default="mock",
                        help="Data source")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 90 days ago)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--events", type=int, default=5000,
                        help="Number of mock events to generate")
    parser.add_argument("--llm", action="store_true",
                        help="Enable LLM narrative summarization")
    parser.add_argument("--llm-provider", default="auto",
                        choices=["auto", "anthropic", "openai", "offline"],
                        help="LLM provider for narratives")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Streamlit dashboard")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    parser.add_argument("--bilateral", nargs=2, metavar=("COUNTRY_A", "COUNTRY_B"),
                        help="Analyze bilateral relationship (e.g., --bilateral USA CHN)")
    parser.add_argument("--temporal", choices=["month", "quarter", "year"],
                        default="month", help="Temporal granularity")
    return parser.parse_args()



def safe_save_csv(df: pd.DataFrame, path, **kwargs):
    """
    Save DataFrame to CSV with a fallback filename if the target is locked.
    On Windows, Excel or another process may hold an exclusive lock on a CSV
    that is currently open, causing PermissionError. We try the original path
    first, then append a timestamp suffix to avoid the lock.
    """
    import time
    path = Path(path)
    try:
        df.to_csv(path, **kwargs)
        print(f"✓ Saved: {path}")
    except PermissionError:
        alt = path.with_stem(path.stem + "_" + str(int(time.time())))
        print(f"  ⚠ PermissionError on {path.name} (is it open in Excel?)")
        print(f"  → Saving to {alt.name} instead")
        df.to_csv(alt, **kwargs)
        print(f"✓ Saved (alt): {alt}")


def run_pipeline(args):
    print("\n" + "="*60)
    print("  🌍 GEOPOLITICAL INTELLIGENCE PIPELINE")
    print("="*60 + "\n")

    # ─── Setup ───────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else end - timedelta(days=90)

    print(f"📅 Date range: {start.date()} → {end.date()}")
    print(f"📡 Data source: {args.source}")
    print(f"📁 Output: {out_dir.resolve()}\n")

    # ─── Step 1: Data Collection ──────────────────────────────────────
    print("─" * 40)
    print("STEP 1: Data Collection")
    print("─" * 40)

    if args.source == "gdelt":
        raw_df = collect_gdelt_range(start, end)
    else:
        raw_df = generate_mock_data(start, end, n_events=args.events)

    # ─── Step 2: Preprocessing ────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 2: Preprocessing")
    print("─" * 40)

    df = preprocess(raw_df)

    # ── Guard: abort early if preprocessing yielded 0 events ─────────────
    if len(df) == 0:
        print("\n❌ ERROR: Preprocessing returned 0 valid events.")
        print("   This usually means the GDELT files were fetched but contain no")
        print("   rows with valid 3-letter country codes in both Actor fields.")
        print("   Try a smaller date range or use --source mock to test the pipeline.")
        return

    safe_save_csv(df, out_dir / "events_clean.csv")

    # ─── Step 3: Graph Construction ───────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 3: Graph Construction")
    print("─" * 40)

    G = build_graph(df)

    # Save edge list
    edge_records = []
    for u, v, d in G.edges(data=True):
        edge_records.append({"source": u, "target": v, **{
            k: v for k, v in d.items() if k != "event_types"
        }})
    safe_save_csv(pd.DataFrame(edge_records), out_dir / "edges.csv")

    # ─── Step 4: Network Analysis ─────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 4: Network Analysis")
    print("─" * 40)

    metrics_df = compute_metrics(G)
    stats = compute_network_stats(G)

    safe_save_csv(metrics_df, out_dir / "country_metrics.csv")

    # Save stats (excluding non-serializable community_map for JSON)
    stats_serializable = {k: v for k, v in stats.items() if k != "community_map"}
    with open(out_dir / "network_stats.json", "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"✓ Stats saved: {out_dir / 'network_stats.json'}")

    # ─── Step 5: Temporal Analysis ────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 5: Temporal Analysis")
    print("─" * 40)

    temporal_graphs = build_temporal_graphs(df, period=args.temporal)
    temporal_records = []
    for period, G_t in temporal_graphs.items():
        s = compute_network_stats(G_t)
        temporal_records.append({
            "period": period,
            "nodes": s["nodes"],
            "edges": s["edges"],
            "avg_tone": s["avg_tone"],
            "neg_ratio": s["negative_edge_ratio"],
            "ggpi": s["ggpi"],
            "modularity": s["modularity"],
        })

    df_temporal = pd.DataFrame(temporal_records).sort_values("period")
    safe_save_csv(df_temporal, out_dir / "temporal_metrics.csv")
    print(df_temporal[["period", "avg_tone", "ggpi", "modularity"]].to_string(index=False))

    # ─── Step 6: Bilateral Analysis ───────────────────────────────────
    if args.bilateral:
        ca, cb = args.bilateral
        print(f"\n{'─' * 40}")
        print(f"STEP 6: Bilateral Analysis — {ca} ↔ {cb}")
        print("─" * 40)

        bil = get_bilateral_summary(G, ca, cb)
        print(json.dumps({k: v for k, v in bil.items()
                           if not isinstance(v, dict) or k in ["relationship_type"]},
                          indent=2))

    # ─── Step 7: LLM Narratives ───────────────────────────────────────
    if args.llm:
        print("\n" + "─" * 40)
        print("STEP 7: LLM Narrative Summarization")
        print("─" * 40)

        narrator = GeopoliticalNarrator(provider=args.llm_provider)
        summaries = {}

        print("Generating network summary...")
        summaries["network"] = narrator.summarize_network(G, stats, metrics_df)
        print(f"\n📝 NETWORK SUMMARY:\n{summaries['network']}\n")

        print("Generating top-10 country profiles...")
        country_summaries = narrator.batch_summarize_countries(G, metrics_df, top_n=10)
        summaries["countries"] = country_summaries

        if args.bilateral:
            ca, cb = args.bilateral
            print(f"Generating bilateral summary: {ca} ↔ {cb}")
            summaries[f"bilateral_{ca}_{cb}"] = narrator.summarize_bilateral(G, ca, cb)
            print(f"\n📝 {ca}–{cb}:\n{summaries[f'bilateral_{ca}_{cb}']}\n")

        narrator.save_summaries(summaries, str(out_dir / "summaries.json"))

    # ─── Final Report ─────────────────────────────────────────────────
    generate_report(df, G, metrics_df, stats, out_dir)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print(f"   Outputs in: {out_dir.resolve()}")
    print("="*60)

    return G, metrics_df, stats, df


def generate_report(df, G, metrics_df, stats, out_dir):
    """Generate a Markdown summary report."""
    top10 = metrics_df.head(10)

    report = f"""# Geopolitical Intelligence Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Summary
- **Total events**: {len(df):,}
- **Countries**: {df['Actor1CountryCode'].nunique()}
- **Date range**: {df['date'].min()} to {df['date'].max()}

## Network Statistics
| Metric | Value |
|--------|-------|
| Nodes | {stats['nodes']} |
| Edges | {stats['edges']} |
| Density | {stats['density']:.4f} |
| Avg Sentiment | {stats['avg_tone']:+.4f} |
| Conflict Rate | {stats['negative_edge_ratio']:.1%} |
| Modularity | {stats['modularity']:.4f} |
| Geopolitical Blocs | {stats['num_communities']} |
| **GGPI** | **{stats['ggpi']:.4f}** |

## Top 10 Most Influential Countries
| Country | PageRank | Betweenness | Conflict% | Events |
|---------|----------|-------------|-----------|--------|
"""
    for country, row in top10.iterrows():
        report += (
            f"| {country} | {row['pagerank']:.6f} | "
            f"{row['betweenness']:.4f} | "
            f"{row['conflict_ratio']:.1%} | "
            f"{int(row['total_events'])} |\n"
        )

    report += f"""
## Event Type Distribution
"""
    for etype, count in df["event_type"].value_counts().items():
        pct = count / len(df) * 100
        report += f"- **{etype}**: {count:,} ({pct:.1f}%)\n"

    report += f"""
## Global Geopolitical Polarization Index (GGPI)

**GGPI = {stats['ggpi']:.4f}** (scale 0–1, higher = more polarized)

Computed as:
- 40% × Network Modularity ({stats['modularity']:.4f})
- 40% × Negative Edge Ratio ({stats['negative_edge_ratio']:.4f})
- 20% × Negative Average Tone ({max(0, -stats['avg_tone']):.4f})

---
*Generated by GeoIntel Pipeline*
"""

    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Report saved: {report_path}")


def launch_dashboard():
    """
    Launch the Streamlit dashboard.
    Uses python -m streamlit so it works even when the streamlit
    command is not on PATH (common on Windows Store Python installs).
    """
    import subprocess
    dashboard_path = ROOT / "dashboard" / "app.py"
    print(f"\n🚀 Launching dashboard: {dashboard_path}")
    print("   Open http://localhost:8501 in your browser\n")

    # Try bare streamlit first, fall back to python -m streamlit
    for cmd in [
        ["streamlit", "run", str(dashboard_path)],
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
    ]:
        try:
            subprocess.run(cmd, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"Dashboard exited with code {e.returncode}")
            return

    print("\n\u274c Could not launch Streamlit. Make sure it is installed:")
    print("   pip install streamlit")
    print("\nThen run manually:")
    print(f"   streamlit run {dashboard_path}")
    print("   -- or --")
    print(f"   python -m streamlit run {dashboard_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.dashboard:
        launch_dashboard()
    else:
        run_pipeline(args)
