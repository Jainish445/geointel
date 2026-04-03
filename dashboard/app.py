"""
Geopolitical Intelligence Dashboard
=====================================
Interactive Streamlit dashboard for analyzing and visualizing
the global geopolitical relationship network.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gdelt_collector import generate_mock_data, preprocess
from analysis.graph_builder import build_graph, compute_metrics, compute_network_stats, build_temporal_graphs
from analysis.narrator import GeopoliticalNarrator


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoIntel — Geopolitical Network Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1e293b;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --green: #22c55e;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
  }

  .stApp { background: var(--bg); color: var(--text); }

  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
  body, p, span, div { font-family: 'Space Mono', monospace !important; font-size: 0.85rem; }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 16px;
    margin: 4px 0;
  }

  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Syne', sans-serif !important;
  }

  .metric-label {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .narrative-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent2);
    border-radius: 6px;
    padding: 20px;
    line-height: 1.7;
    color: var(--text);
  }

  .ggpi-bar-container {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    margin-top: 6px;
  }

  .tag {
    display: inline-block;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: var(--accent);
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px;
  }

  .tag-red {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.3);
    color: var(--red);
  }

  .tag-green {
    background: rgba(34, 197, 94, 0.12);
    border-color: rgba(34, 197, 94, 0.3);
    color: var(--green);
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  .stSelectbox > div > div { background: var(--surface) !important; }
  .stSlider > div { color: var(--accent) !important; }

  div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading (cached) ────────────────────────────────────────────────────

# Path to the pipeline output directory (sibling of dashboard/)
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


@st.cache_data(show_spinner=False, ttl=300)
def load_data(source: str, n_events: int = 5000, days_back: int = 90,
              gdelt_start: str = "", gdelt_end: str = "") -> tuple:
    """
    Load event data from one of three sources (in priority order):

    1. pipeline_output  — reads output/events_clean.csv produced by main.py
                          This is REAL data (GDELT or whatever was last run).
    2. gdelt_live       — fetches directly from GDELT right now for given dates.
    3. mock             — generates synthetic data (for offline testing only).

    Returns (df, source_label) so the UI can show which source is active.
    """
    from data.gdelt_collector import (
        collect_gdelt_range, generate_mock_data, preprocess
    )

    if source == "pipeline_output":
        csv_path = os.path.join(_OUTPUT_DIR, "events_clean.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Validate it has the columns we need
                required = {"Actor1CountryCode", "Actor2CountryCode",
                            "tone_norm", "event_type", "date"}
                if required.issubset(set(df.columns)):
                    return df, f"Pipeline output ({len(df):,} events from {csv_path})"
            except Exception as e:
                pass
        return pd.DataFrame(), "pipeline_output — file not found (run main.py first)"

    elif source == "gdelt_live":
        try:
            if gdelt_start and gdelt_end:
                start = datetime.strptime(gdelt_start, "%Y-%m-%d")
                end   = datetime.strptime(gdelt_end,   "%Y-%m-%d")
            else:
                end   = datetime.now()
                start = end - timedelta(days=min(days_back, 7))
            raw = collect_gdelt_range(start, end, target_rows_per_day=2000)
            df  = preprocess(raw)
            if len(df) > 0:
                return df, f"GDELT live ({len(df):,} events, {start.date()} – {end.date()})"
        except Exception as e:
            pass
        return pd.DataFrame(), "GDELT live — fetch failed (check network)"

    else:  # mock
        end   = datetime.now()
        start = end - timedelta(days=days_back)
        raw   = generate_mock_data(start, end, n_events)
        df    = preprocess(raw)
        return df, f"Mock simulated ({len(df):,} synthetic events)"


@st.cache_resource(show_spinner=False)
def load_graph(df_json: str):
    df = pd.read_json(df_json)
    G = build_graph(df)
    metrics_df = compute_metrics(G)
    stats = compute_network_stats(G)
    return G, metrics_df, stats


@st.cache_resource(show_spinner=False)
def load_temporal(df_json: str):
    df = pd.read_json(df_json)
    return build_temporal_graphs(df, period="month")


# ─── App State ────────────────────────────────────────────────────────────────

def main():
    # Header
    col_logo, col_title = st.columns([1, 8])
    with col_title:
        st.markdown("""
        <h1 style='margin:0; font-size:2rem; color:#00d4ff; letter-spacing:-0.02em;'>
          🌐 GeoIntel
        </h1>
        <p style='color:#64748b; margin:0; font-size:0.8rem;'>
          AI-Powered Geopolitical Relationship Network Analysis
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Data Source")

        source = st.radio(
            "Source",
            options=["pipeline_output", "gdelt_live", "mock"],
            format_func=lambda s: {
                "pipeline_output": "📁 Pipeline output (main.py)",
                "gdelt_live":      "🌐 GDELT live fetch",
                "mock":            "🎲 Mock / simulated",
            }[s],
            index=0,
            help="pipeline_output reads output/events_clean.csv generated by main.py"
        )

        # Source-specific controls
        n_events, days_back, gdelt_start, gdelt_end = 5000, 90, "", ""
        if source == "mock":
            n_events  = st.slider("Events to simulate", 1000, 10000, 5000, 500)
            days_back = st.slider("Date range (days)",  30, 365, 90, 30)
        elif source == "gdelt_live":
            st.markdown("**GDELT date range**")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                gdelt_start = st.date_input(
                    "From", value=datetime.now().date() - timedelta(days=7),
                    key="gdelt_from"
                ).strftime("%Y-%m-%d")
            with col_d2:
                gdelt_end = st.date_input(
                    "To", value=datetime.now().date(), key="gdelt_to"
                ).strftime("%Y-%m-%d")
            st.caption("⚠️ Fetches full daily files — allow ~5s per day")
        else:  # pipeline_output
            csv_path = os.path.join(_OUTPUT_DIR, "events_clean.csv")
            if os.path.exists(csv_path):
                mtime = datetime.fromtimestamp(os.path.getmtime(csv_path))
                st.success(f"✓ Found events_clean.csv\n{mtime.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.error(
                    "events_clean.csv not found.\n\n"
                    "Run the pipeline first:\n"
                    "`python main.py`\n"
                    "or\n"
                    "`python main.py --source gdelt --start 2024-01-01 --end 2024-03-31`"
                )

        if st.button("🔄 Reload data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        with st.spinner("Loading data..."):
            df, data_source_label = load_data(
                source, n_events, days_back, gdelt_start, gdelt_end
            )

        if len(df) == 0:
            st.error("No data loaded. Check the source above.")
            st.stop()

        st.caption(f"📊 {data_source_label}")

        event_types = ["All"] + sorted(df["event_type"].unique().tolist())
        selected_type = st.selectbox("Event Type", event_types)

        st.markdown("---")
        st.markdown("### 🤖 LLM Narratives")
        use_llm = st.toggle("Enable AI Summaries", value=False,
                            help="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY env variable")

        st.markdown("---")
        st.markdown(
            f"<p style='color:#64748b; font-size:0.7rem;'>"
            f"Model: NetworkX + DistilBERT<br>v1.0 — GeoIntel Pipeline</p>",
            unsafe_allow_html=True
        )

    # Filter dataframe
    df_filtered = df.copy()
    if selected_type != "All":
        df_filtered = df_filtered[df_filtered["event_type"] == selected_type]

    # Build graph
    with st.spinner("Building geopolitical network..."):
        G, metrics_df, stats = load_graph(df_filtered.to_json())

    # ─── Top KPI Strip ────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    def kpi_card(col, value, label, color="var(--accent)"):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{color}'>{value}</div>
          <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi_card(kpi1, stats["nodes"], "Countries")
    kpi_card(kpi2, f"{stats['edges']:,}", "Interactions")
    kpi_card(kpi3, f"{stats['avg_tone']:+.3f}", "Avg Sentiment",
             "#22c55e" if stats['avg_tone'] > 0 else "#ef4444")
    kpi_card(kpi4, f"{stats['negative_edge_ratio']:.1%}", "Conflict Rate", "#ef4444")
    kpi_card(kpi5, f"{stats['num_communities']}", "Blocs Detected")
    kpi_card(kpi6, f"{stats['ggpi']:.3f}", "GGPI", "#ff6b35")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ Network Graph",
        "📊 Influence Rankings",
        "🔗 Bilateral Analysis",
        "📈 Temporal Trends",
        "🌐 Network Overview"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Network Graph
    # ═══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### 🗺️ Geopolitical Network — World Map")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            min_weight = st.slider("Min edge weight", 1, 20, 3, 1,
                                   key="net_weight")
        with col_ctrl2:
            max_nodes = st.slider("Max countries shown", 10, max(30, len(G.nodes())), min(30, len(G.nodes())), 5,
                                  key="net_nodes")
        with col_ctrl3:
            color_by = st.selectbox("Color nodes by",
                                    ["PageRank", "Conflict Ratio", "Community"],
                                    key="net_color")

        # Build geo map
        fig_net = build_network_figure(G, metrics_df, stats, min_weight, max_nodes, color_by)
        st.plotly_chart(fig_net, use_container_width=True)

        # Legend
        st.markdown("""
        <div style='color:#64748b; font-size:0.75rem; display:flex; gap:12px; flex-wrap:wrap;'>
          <span class='tag'>Node size = GDP (PPP 2023)</span>
          <span class='tag-green'>Green edges = cooperative</span>
          <span class='tag-red'>Red edges = conflictual</span>
          <span class='tag'>Edge width = interaction volume</span>
        </div>
        """, unsafe_allow_html=True)

        # ─── Geopolitical Bloc Analyzer ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧭 Geopolitical Bloc Analyzer")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Select 2–4 power poles. Each country is scored by bilateral network tone "
            "toward each pole and assigned to the bloc it leans toward most."
            "</span>", unsafe_allow_html=True
        )

        all_iso = sorted(metrics_df.index.tolist())
        POLE_PRESETS = {
            "USA vs CHN":           ["USA", "CHN"],
            "USA vs CHN vs RUS":    ["USA", "CHN", "RUS"],
            "USA vs CHN vs EU":     ["USA", "CHN", "DEU"],
            "USA vs CHN vs RUS vs IND": ["USA", "CHN", "RUS", "IND"],
            "Custom":               [],
        }

        col_pre, col_poles = st.columns([1, 2])
        with col_pre:
            preset = st.selectbox("Preset", list(POLE_PRESETS.keys()), key="bloc_preset")
        with col_poles:
            poles = st.multiselect(
                "Power poles (2–4 countries)",
                options=all_iso,
                default=[p for p in POLE_PRESETS[preset] if p in all_iso],
                max_selections=4,
                key="bloc_poles",
            )

        if len(poles) < 2:
            st.info("⬆ Select at least 2 poles above to compute blocs.")
        else:
            bloc_assignments, affinity_table = compute_blocs(G, poles)
            fig_bloc = build_bloc_figure(G, metrics_df, bloc_assignments,
                                         affinity_table, poles, min_weight)
            st.plotly_chart(fig_bloc, use_container_width=True)

            BLOC_COLORS = ["#00d4ff", "#ff6b35", "#22c55e", "#f59e0b", "#a78bfa", "#ec4899"]
            bloc_members: dict = {}
            for c, b in bloc_assignments.items():
                bloc_members.setdefault(b, []).append(c)

            cols_b = st.columns(len(poles))
            for i, pole in enumerate(poles):
                members = sorted(bloc_members.get(pole, []))
                col_c = BLOC_COLORS[i % len(BLOC_COLORS)]
                with cols_b[i]:
                    pills = "".join(
                        f"<span style='display:inline-block;background:{col_c}18;"
                        f"border:1px solid {col_c}40;color:{col_c};"
                        f"border-radius:3px;padding:1px 7px;margin:2px;"
                        f"font-size:0.72rem'>{m}</span>"
                        for m in members
                    )
                    st.markdown(
                        f"<div style='border-left:3px solid {col_c};padding:10px 14px;"
                        f"background:rgba(17,24,39,0.8);border-radius:4px'>"
                        f"<div style='color:{col_c};font-weight:700;font-size:1rem;"
                        f"margin-bottom:4px'>🏳 {pole} bloc</div>"
                        f"<div style='color:#94a3b8;font-size:0.73rem;margin-bottom:8px'>"
                        f"{len(members)} countries</div>"
                        f"{pills}</div>",
                        unsafe_allow_html=True,
                    )

            with st.expander("📊 Full affinity scores", expanded=False):
                aff_df = pd.DataFrame(affinity_table).T.round(4)
                aff_df.index.name = "Country"
                aff_df["Assigned Bloc"] = aff_df.index.map(lambda c: bloc_assignments.get(c, "—"))
                st.dataframe(aff_df.style.background_gradient(
                    cmap="RdYlGn", subset=[p for p in poles if p in aff_df.columns]
                ), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: Influence Rankings
    # ═══════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Country Influence Rankings")

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Top N table
            n_show = st.slider("Show top N countries", 5, 30, 15, key="rank_n")
            top_df = metrics_df.head(n_show).copy()
            top_df.index.name = "Country"
            top_df_display = top_df[[
                "pagerank", "betweenness", "eigenvector",
                "conflict_ratio", "total_events"
            ]].rename(columns={
                "pagerank": "PageRank",
                "betweenness": "Betweenness",
                "eigenvector": "Eigenvector",
                "conflict_ratio": "Conflict%",
                "total_events": "Events"
            })
            top_df_display["Conflict%"] = (top_df_display["Conflict%"] * 100).round(1)
            st.dataframe(top_df_display.style.background_gradient(
                subset=["PageRank"], cmap="Blues"
            ).background_gradient(
                subset=["Conflict%"], cmap="Reds"
            ), use_container_width=True)

        with col_r:
            # Radar chart for a selected country
            selected_country = st.selectbox(
                "Country radar chart", metrics_df.head(20).index.tolist(), key="radar_c"
            )
            fig_radar = build_radar_chart(metrics_df, selected_country)
            st.plotly_chart(fig_radar, use_container_width=True)

        # Bar chart: PageRank
        st.markdown("#### 📊 PageRank Influence Score")
        fig_bar = px.bar(
            metrics_df.head(20).reset_index(),
            x="country", y="pagerank",
            color="conflict_ratio",
            color_continuous_scale="RdYlGn_r",
            labels={"pagerank": "PageRank Score", "country": "Country",
                    "conflict_ratio": "Conflict%"},
            template="plotly_dark"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Bilateral Analysis
    # ═══════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Bilateral Relationship Analysis")

        _bil_countries = sorted(G.nodes())
        col_a, col_b = st.columns(2)
        with col_a:
            ca = st.selectbox("Country A", _bil_countries, index=0, key="bil_a")
        with col_b:
            cb_opts = [c for c in _bil_countries if c != ca]
            cb = st.selectbox("Country B", cb_opts, index=1, key="bil_b")

        if st.button("🔍 Analyze Relationship", type="primary"):
            col_stats, col_chart = st.columns([2, 3])

            with col_stats:
                # Edge data
                a_to_b = G[ca][cb] if G.has_edge(ca, cb) else {}
                b_to_a = G[cb][ca] if G.has_edge(cb, ca) else {}

                tones = [x.get("tone", 0) for x in [a_to_b, b_to_a] if x]
                avg_tone = np.mean(tones) if tones else 0.0
                tone_color = "#22c55e" if avg_tone > 0.05 else "#ef4444" if avg_tone < -0.05 else "#f59e0b"

                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value' style='color:{tone_color}'>{avg_tone:+.4f}</div>
                  <div class='metric-label'>Average Sentiment Tone</div>
                </div>
                """, unsafe_allow_html=True)

                rel = "🤝 Cooperative" if avg_tone > 0.1 else \
                      "⚔️ Conflictual" if avg_tone < -0.1 else "⚖️ Mixed/Neutral"
                st.markdown(f"**Relationship Status:** {rel}")

                if a_to_b:
                    st.markdown(f"**{ca} → {cb}:** {a_to_b.get('num_events', 0)} events, "
                                f"dominant: `{a_to_b.get('dominant_type', 'N/A')}`")
                if b_to_a:
                    st.markdown(f"**{cb} → {ca}:** {b_to_a.get('num_events', 0)} events, "
                                f"dominant: `{b_to_a.get('dominant_type', 'N/A')}`")

            with col_chart:
                fig_bil = build_bilateral_chart(G, ca, cb)
                st.plotly_chart(fig_bil, use_container_width=True)

            # LLM Narrative
            if use_llm:
                st.markdown("#### 🤖 AI Intelligence Summary")
                with st.spinner("Generating analysis..."):
                    narrator = GeopoliticalNarrator()
                    summary = narrator.summarize_bilateral(G, ca, cb)
                st.markdown(f"<div class='narrative-box'>{summary}</div>",
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: Temporal Trends
    # ═══════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Temporal Network Evolution")

        with st.spinner("Building temporal graphs..."):
            temporal_graphs = load_temporal(df_filtered.to_json())

        if len(temporal_graphs) < 2:
            st.warning("Not enough time periods to show trends. Increase date range.")
        else:
            periods = sorted(temporal_graphs.keys())

            # Compute per-period metrics
            period_stats = []
            for p in periods:
                G_t = temporal_graphs[p]
                m_t = compute_metrics(G_t)
                s_t = compute_network_stats(G_t)
                period_stats.append({
                    "period": p,
                    "nodes": G_t.number_of_nodes(),
                    "edges": G_t.number_of_edges(),
                    "avg_tone": s_t["avg_tone"],
                    "conflict_rate": s_t["negative_edge_ratio"],
                    "ggpi": s_t["ggpi"],
                    "modularity": s_t["modularity"],
                })

            df_ts = pd.DataFrame(period_stats)

            # Line charts
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                fig_tone = px.line(df_ts, x="period", y="avg_tone",
                                   title="Average Sentiment Over Time",
                                   template="plotly_dark",
                                   markers=True,
                                   color_discrete_sequence=["#00d4ff"])
                fig_tone.add_hline(y=0, line_dash="dash", line_color="#64748b")
                fig_tone.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_tone, use_container_width=True)

            with col_t2:
                fig_ggpi = px.line(df_ts, x="period", y="ggpi",
                                   title="GGPI (Polarization Index) Over Time",
                                   template="plotly_dark",
                                   markers=True,
                                   color_discrete_sequence=["#ff6b35"])
                fig_ggpi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_ggpi, use_container_width=True)

            # Dual-axis: nodes + edges
            fig_activity = go.Figure()
            fig_activity.add_trace(go.Bar(
                x=df_ts["period"], y=df_ts["edges"],
                name="Interactions", marker_color="#00d4ff", opacity=0.7
            ))
            fig_activity.add_trace(go.Scatter(
                x=df_ts["period"], y=df_ts["conflict_rate"],
                name="Conflict Rate", yaxis="y2",
                line=dict(color="#ef4444", width=2),
                mode="lines+markers"
            ))
            fig_activity.update_layout(
                title="Network Activity & Conflict Rate Over Time",
                yaxis=dict(title="Interactions"),
                yaxis2=dict(title="Conflict Rate", overlaying="y", side="right",
                            tickformat=".0%"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_activity, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: Network Overview
    # ═══════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Global Network Overview")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### 📡 Network Statistics")
            stats_display = {k: v for k, v in stats.items() if k != "community_map"}
            for k, v in stats_display.items():
                label = k.replace("_", " ").title()
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:6px 0;
                            border-bottom:1px solid #1e293b;'>
                  <span style='color:#64748b; font-size:0.8rem;'>{label}</span>
                  <span style='color:#00d4ff; font-weight:bold;'>{val}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("#### 🏘️ Community / Bloc Detection")
            community_map = stats.get("community_map", {})
            if community_map:
                from collections import defaultdict
                blocs = defaultdict(list)
                for country, bloc_id in community_map.items():
                    blocs[bloc_id].append(country)

                for bloc_id, members in sorted(blocs.items()):
                    members_sorted = sorted(members)
                    color = ["#00d4ff", "#ff6b35", "#22c55e", "#f59e0b",
                             "#a78bfa", "#ec4899"][bloc_id % 6]
                    tags = "".join([f"<span class='tag' style='border-color:{color}; color:{color}'>{m}</span>"
                                    for m in members_sorted])
                    st.markdown(f"""
                    <div style='margin:8px 0;'>
                      <div style='color:#64748b; font-size:0.7rem; margin-bottom:4px;'>
                        BLOC {bloc_id + 1} ({len(members)} countries)
                      </div>
                      {tags}
                    </div>
                    """, unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown("#### 🌡️ Tone Matrix (Top Countries)")
        fig_heat = build_tone_heatmap(G, metrics_df, top_n=15)
        st.plotly_chart(fig_heat, use_container_width=True)

        # LLM network summary
        if use_llm:
            st.markdown("#### 🤖 Global Intelligence Summary")
            with st.spinner("Generating executive summary..."):
                narrator = GeopoliticalNarrator()
                network_summary = narrator.summarize_network(G, stats, metrics_df)
            st.markdown(f"<div class='narrative-box'>{network_summary}</div>",
                        unsafe_allow_html=True)


# ─── Visualization Helpers ────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BLOC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# ── Real-world geopolitical prior affinities ─────────────────────────────────
# Baseline scores reflecting established alliances, treaties, and historical
# relationships. These anchor the computation so mock/sparse data can't flip
# known alignments (e.g. ISR and UKR firmly in Western/USA orbit).
# Format: {country: {pole_or_bloc_anchor: prior_weight}}
# Priors are blended with network-derived scores (weight 0.35 prior, 0.65 data).
GEOPOLITICAL_PRIORS = {
    # Strong US allies / Western bloc
    "GBR": {"USA": +0.80, "CHN": -0.35, "RUS": -0.50},
    "CAN": {"USA": +0.85, "CHN": -0.25, "RUS": -0.40},
    "AUS": {"USA": +0.80, "CHN": -0.30, "RUS": -0.35},
    "DEU": {"USA": +0.65, "CHN": -0.15, "RUS": -0.55},
    "FRA": {"USA": +0.60, "CHN": -0.10, "RUS": -0.40},
    "JPN": {"USA": +0.75, "CHN": -0.40, "RUS": -0.35},
    "KOR": {"USA": +0.70, "CHN": -0.20, "RUS": -0.30},
    "NLD": {"USA": +0.65, "CHN": -0.10, "RUS": -0.45},
    "NOR": {"USA": +0.65, "CHN": -0.10, "RUS": -0.45},
    "SWE": {"USA": +0.65, "CHN": -0.10, "RUS": -0.50},
    "POL": {"USA": +0.70, "CHN": -0.15, "RUS": -0.70},
    "ITA": {"USA": +0.55, "CHN": -0.05, "RUS": -0.30},
    # Firmly US-aligned despite geography
    "ISR": {"USA": +0.85, "CHN": -0.10, "RUS": -0.10, "IRN": -0.90},
    "UKR": {"USA": +0.80, "CHN": -0.10, "RUS": -0.95},
    # Contested / swing states
    "IND": {"USA": +0.20, "CHN": -0.30, "RUS": +0.15},
    "TUR": {"USA": +0.10, "CHN": -0.05, "RUS": -0.15},
    "SAU": {"USA": +0.35, "CHN": +0.15, "RUS": +0.10},
    "EGY": {"USA": +0.25, "CHN": +0.10, "RUS": +0.05},
    "ZAF": {"USA": +0.05, "CHN": +0.10, "RUS": +0.05},
    "NGA": {"USA": +0.10, "CHN": +0.05, "RUS": -0.05},
    "IDN": {"USA": +0.15, "CHN": +0.05, "RUS": -0.05},
    "MEX": {"USA": +0.40, "CHN": -0.10, "RUS": -0.10},
    "BRA": {"USA": +0.10, "CHN": +0.15, "RUS": +0.05},
    "ARG": {"USA": +0.05, "CHN": +0.20, "RUS": +0.10},
    "CHL": {"USA": +0.25, "CHN": +0.10, "RUS": -0.05},
    # China/Russia aligned
    "RUS": {"USA": -0.80, "CHN": +0.65},
    "IRN": {"USA": -0.85, "CHN": +0.40, "RUS": +0.35},
    "PAK": {"USA": -0.10, "CHN": +0.65, "RUS": +0.10},
    "CHN": {"USA": -0.65, "RUS": +0.50},
}
PRIOR_WEIGHT = 0.40   # blend: 40% prior knowledge, 60% network data


def compute_blocs(G, poles):
    """
    Compute affinity of every country toward each pole using a hybrid approach:

    1. NETWORK score  — bilateral edge tone between country and pole
                        (direct edges only; 0 if no edge exists)
    2. FRIEND score   — weighted average tone of country's neighbours
                        toward the pole (1-hop propagation, weight 0.3)
    3. PRIOR score    — real-world geopolitical baseline from GEOPOLITICAL_PRIORS
                        (covers poles that may not appear verbatim, e.g. "EU"
                         mapped to DEU+FRA+GBR average)

    Final score = 0.60 * (network + friend) + 0.40 * prior
    The country is assigned to the pole with the highest final score.
    """
    # Helper: get bilateral tone between two nodes
    def bilateral_tone(a, b):
        tones = []
        if G.has_edge(a, b): tones.append(G[a][b].get("tone", 0))
        if G.has_edge(b, a): tones.append(G[b][a].get("tone", 0))
        return float(sum(tones)/len(tones)) if tones else 0.0

    # Expand poles that might be composite (future: EU → DEU+FRA+GBR)
    # For now each pole is a single country; keep as-is
    affinity = {}

    for node in G.nodes():
        if node in poles:
            continue

        scores = {}
        for pole in poles:
            # ── 1. Direct network score ───────────────────────────────
            net_score = bilateral_tone(node, pole)

            # ── 2. Friend-of-friend propagation ───────────────────────
            # neighbours' average tone toward the pole, weighted by edge weight
            nbr_scores, total_w = [], 0.0
            for nbr in set(list(G.successors(node)) + list(G.predecessors(node))):
                if nbr == pole or nbr == node: continue
                edge_w = (G[node][nbr].get("weight", 1) if G.has_edge(node, nbr)
                          else G[nbr][node].get("weight", 1) if G.has_edge(nbr, node)
                          else 1)
                nbr_t = bilateral_tone(nbr, pole)
                nbr_scores.append(nbr_t * edge_w)
                total_w += edge_w
            friend_score = (sum(nbr_scores)/total_w) if total_w > 0 else 0.0

            data_score = 0.70 * net_score + 0.30 * friend_score

            # ── 3. Prior knowledge ────────────────────────────────────
            node_priors = GEOPOLITICAL_PRIORS.get(node, {})
            prior_score = node_priors.get(pole, 0.0)
            # If prior doesn't mention this pole, interpolate from nearby:
            if pole not in node_priors and node_priors:
                # Use the mean of known priors as a neutral baseline
                prior_score = 0.0

            # ── Blend ─────────────────────────────────────────────────
            scores[pole] = (1 - PRIOR_WEIGHT) * data_score + PRIOR_WEIGHT * prior_score

        affinity[node] = scores

    # Assign each country to highest-scoring pole
    assignments = {}
    for node, scores in affinity.items():
        assignments[node] = max(scores, key=scores.get)
    for pole in poles:
        assignments[pole] = pole

    return assignments, affinity


BLOC_PALETTE = {
    0: ("#00d4ff", "#003d4d"),   # cyan
    1: ("#ff6b35", "#4d1f0a"),   # orange
    2: ("#22c55e", "#0a3319"),   # green
    3: ("#f59e0b", "#3d2700"),   # amber
    4: ("#a78bfa", "#2d2050"),   # violet
}


def build_bloc_figure(G, metrics_df, bloc_assignments, affinity_table, poles, min_weight):
    """
    World map with nodes coloured by geopolitical bloc.
    Edges are coloured by the pole of the SOURCE country.
    Pole nodes are drawn larger with a ring highlight.
    """
    pole_idx = {pole: i for i, pole in enumerate(poles)}
    max_gdp  = max(v[3] for v in COUNTRY_GEO.values())

    # Build one Scattergeo trace per bloc so the legend shows bloc names
    traces = []

    # ── Edge traces (drawn first, behind nodes) ───────────────────────────
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0) < min_weight:
            continue
        if u not in COUNTRY_GEO or v not in COUNTRY_GEO:
            continue
        src_bloc  = bloc_assignments.get(u)
        src_pole  = pole_idx.get(src_bloc, 0)
        fill_c, _ = BLOC_PALETTE.get(src_pole, ("#64748b", "#1e293b"))
        tone  = d.get("tone", 0)
        alpha = min(0.55, max(0.06, abs(tone) * 1.8))
        # Tint edges slightly toward pole color but keep conflict/coop signal
        if tone >= 0:
            edge_col = fill_c.replace("#", "rgba(").rstrip(")")  # crude; use explicit
            r, g, b = int(fill_c[1:3],16), int(fill_c[3:5],16), int(fill_c[5:7],16)
            edge_col = f"rgba({r},{g},{b},{alpha:.2f})"
        else:
            edge_col = f"rgba(239,68,68,{alpha:.2f})"
        traces.append(go.Scattergeo(
            lat=[COUNTRY_GEO[u][0], COUNTRY_GEO[v][0], None],
            lon=[COUNTRY_GEO[u][1], COUNTRY_GEO[v][1], None],
            mode="lines",
            line=dict(width=max(0.3, min(d["weight"]/18, 3.0)), color=edge_col),
            hoverinfo="none", showlegend=False,
        ))

    # ── Node traces — one per bloc so legend works ────────────────────────
    bloc_lats = {p: [] for p in poles}
    bloc_lons = {p: [] for p in poles}
    bloc_sizes = {p: [] for p in poles}
    bloc_hover = {p: [] for p in poles}
    bloc_labels = {p: [] for p in poles}

    for node, bloc in bloc_assignments.items():
        if node not in COUNTRY_GEO or bloc not in poles:
            continue
        lat, lon, fullname, gdp = COUNTRY_GEO[node]
        m  = metrics_df.loc[node] if node in metrics_df.index else {}
        pr = float(m.get("pagerank", 0.01)) if hasattr(m, "get") else 0.01
        cr = float(m.get("conflict_ratio", 0.0)) if hasattr(m, "get") else 0.0
        ev = int(m.get("total_events", 0))     if hasattr(m, "get") else 0

        # Poles themselves get a bigger fixed ring
        if node == bloc:
            size = 22 + (gdp / max_gdp) ** 0.5 * 52
        else:
            size = 8  + (gdp / max_gdp) ** 0.5 * 42

        # Affinity bar in tooltip
        aff_lines = ""
        if node in affinity_table:
            for p, sc in sorted(affinity_table[node].items(), key=lambda x: -x[1]):
                bar_len = int(max(0, (sc + 1) / 2 * 12))
                aff_lines += f"{p}: {'█'*bar_len} {sc:+.3f}<br>"

        bloc_lats[bloc].append(lat)
        bloc_lons[bloc].append(lon)
        bloc_sizes[bloc].append(size)
        bloc_labels[bloc].append(node)
        bloc_hover[bloc].append(
            f"<b>{fullname}</b> — {bloc} bloc<br>"
            f"GDP (PPP): ${gdp:.1f}T<br>"
            f"PageRank: {pr:.5f}<br>"
            f"Conflict: {cr:.1%}<br>"
            f"Events: {ev}<br>"
            + ("<br><b>Affinity scores</b><br>" + aff_lines if aff_lines else "")
        )

    for i, pole in enumerate(poles):
        fill_c, bg_c = BLOC_PALETTE.get(i, ("#64748b", "#1e293b"))
        traces.append(go.Scattergeo(
            lat=bloc_lats[pole],
            lon=bloc_lons[pole],
            mode="markers+text",
            name=f"{pole} bloc",
            text=bloc_labels[pole],
            textposition="top center",
            textfont=dict(size=9, color=fill_c),
            hovertext=bloc_hover[pole],
            hoverinfo="text",
            marker=dict(
                size=bloc_sizes[pole],
                sizemode="diameter",
                color=fill_c,
                opacity=0.88,
                line=dict(
                    width=[3 if lbl == pole else 1 for lbl in bloc_labels[pole]],
                    color=["white" if lbl == pole else bg_c
                           for lbl in bloc_labels[pole]],
                ),
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#1a2235",
        showocean=True,      oceancolor="#0d1421",
        showlakes=True,      lakecolor="#0d1421",
        showrivers=False,
        showcountries=True,  countrycolor="#253248", countrywidth=0.5,
        showcoastlines=True, coastlinecolor="#2d3f5a", coastlinewidth=0.6,
        showframe=False,
        bgcolor="#0a0e1a",
    )
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(bgcolor="#0a0e1a"),
        legend=dict(
            orientation="h", y=-0.02, x=0.5, xanchor="center",
            font=dict(color="#e2e8f0", size=11),
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#253248", borderwidth=1,
        ),
    )
    return fig


## ── Country geo data: coordinates + GDP-PPP (USD billions, IMF 2023) ──────────
COUNTRY_GEO = {
    # ISO3: (lat, lon, display_name, gdp_ppp_billions)
    # Major powers
    "USA": (38.0,-97.0,"United States",26900), "CHN": (35.0,103.0,"China",33000),
    "RUS": (62.0,90.0,"Russia",5300),           "DEU": (51.0,10.0,"Germany",5500),
    "GBR": (54.0,-2.0,"United Kingdom",3700),   "FRA": (46.0,2.0,"France",3900),
    "IND": (21.0,78.0,"India",13100),            "BRA": (-10.0,-55.0,"Brazil",3900),
    "JPN": (36.0,138.0,"Japan",6500),            "KOR": (36.0,128.0,"South Korea",2700),
    "ISR": (31.5,35.0,"Israel",520),             "IRN": (32.0,53.0,"Iran",1600),
    "SAU": (24.0,45.0,"Saudi Arabia",2000),      "TUR": (39.0,35.0,"Turkey",3600),
    "PAK": (30.0,70.0,"Pakistan",1500),          "NGA": (9.0,8.0,"Nigeria",1300),
    "ZAF": (-29.0,25.0,"South Africa",900),      "EGY": (27.0,30.0,"Egypt",1900),
    "MEX": (24.0,-102.0,"Mexico",3100),          "ARG": (-34.0,-64.0,"Argentina",1200),
    "IDN": (-2.0,118.0,"Indonesia",4400),        "AUS": (-25.0,134.0,"Australia",1700),
    "CAN": (56.0,-96.0,"Canada",2400),           "ITA": (42.5,12.5,"Italy",3200),
    "UKR": (49.0,31.0,"Ukraine",600),            "POL": (52.0,20.0,"Poland",1700),
    "NLD": (52.3,5.3,"Netherlands",1200),        "SWE": (60.0,15.0,"Sweden",700),
    "NOR": (64.0,13.0,"Norway",600),             "CHL": (-30.0,-71.0,"Chile",600),
    # Europe
    "ESP": (40.0,-4.0,"Spain",2400),             "PRT": (39.5,-8.0,"Portugal",410),
    "BEL": (50.8,4.4,"Belgium",760),             "AUT": (47.5,14.5,"Austria",620),
    "CHE": (47.0,8.0,"Switzerland",720),         "DNK": (56.0,10.0,"Denmark",440),
    "FIN": (64.0,26.0,"Finland",360),            "GRC": (39.0,22.0,"Greece",380),
    "CZE": (49.8,15.5,"Czech Republic",560),     "HUN": (47.0,19.0,"Hungary",420),
    "ROU": (46.0,25.0,"Romania",730),            "BGR": (42.7,25.5,"Bulgaria",230),
    "HRV": (45.0,16.0,"Croatia",160),            "SVK": (48.7,19.5,"Slovakia",260),
    "SVN": (46.1,14.8,"Slovenia",110),           "SRB": (44.0,21.0,"Serbia",190),
    "BLR": (53.5,28.0,"Belarus",220),            "MDA": (47.0,28.5,"Moldova",40),
    "ALB": (41.0,20.0,"Albania",55),             "LTU": (55.9,23.9,"Lithuania",160),
    "LVA": (57.0,25.0,"Latvia",80),              "EST": (59.0,25.0,"Estonia",60),
    "BIH": (44.0,17.5,"Bosnia-Herzegovina",75),  "MKD": (41.6,21.7,"N. Macedonia",45),
    "MNE": (42.7,19.4,"Montenegro",18),          "IRL": (53.0,-8.0,"Ireland",620),
    "LUX": (49.8,6.1,"Luxembourg",105),
    # Middle East & Central Asia
    "IRQ": (33.0,44.0,"Iraq",520),               "SYR": (35.0,38.0,"Syria",60),
    "JOR": (31.0,36.0,"Jordan",120),             "LBN": (33.9,35.5,"Lebanon",70),
    "YEM": (15.5,47.5,"Yemen",55),               "OMN": (22.0,58.0,"Oman",220),
    "ARE": (24.0,54.0,"UAE",760),                "QAT": (25.3,51.2,"Qatar",280),
    "KWT": (29.5,47.8,"Kuwait",240),             "BHR": (26.0,50.5,"Bahrain",92),
    "AFG": (33.0,65.0,"Afghanistan",75),         "KAZ": (48.0,68.0,"Kazakhstan",680),
    "UZB": (41.0,64.0,"Uzbekistan",280),         "TKM": (39.0,59.0,"Turkmenistan",110),
    "TJK": (39.0,71.0,"Tajikistan",45),          "KGZ": (41.0,75.0,"Kyrgyzstan",38),
    "AZE": (40.5,47.5,"Azerbaijan",200),         "ARM": (40.0,45.0,"Armenia",65),
    "GEO": (42.0,43.5,"Georgia",80),
    # Asia-Pacific
    "VNM": (16.0,108.0,"Vietnam",1300),          "THA": (15.0,101.0,"Thailand",1500),
    "MYS": (4.0,109.0,"Malaysia",1200),          "PHL": (13.0,122.0,"Philippines",1100),
    "SGP": (1.35,103.8,"Singapore",720),         "BGD": (24.0,90.0,"Bangladesh",1200),
    "LKA": (7.5,80.7,"Sri Lanka",280),           "NPL": (28.0,84.0,"Nepal",140),
    "MMR": (17.0,96.0,"Myanmar",280),            "KHM": (12.5,105.0,"Cambodia",95),
    "LAO": (18.0,103.0,"Laos",70),               "MNG": (46.0,105.0,"Mongolia",55),
    "NZL": (-41.0,174.0,"New Zealand",260),      "PNG": (-6.0,147.0,"Papua New Guinea",45),
    "TWN": (23.7,121.0,"Taiwan",1700),           "PRK": (40.0,127.0,"North Korea",40),
    # Africa
    "ETH": (9.0,40.0,"Ethiopia",380),            "TZA": (-6.0,35.0,"Tanzania",220),
    "KEN": (1.0,38.0,"Kenya",320),               "GHA": (8.0,-2.0,"Ghana",200),
    "CIV": (7.5,-5.5,"Côte d'Ivoire",180),      "AGO": (-12.0,18.5,"Angola",280),
    "CMR": (6.0,12.0,"Cameroon",130),            "MOZ": (-18.0,35.0,"Mozambique",55),
    "MDG": (-20.0,47.0,"Madagascar",50),         "ZMB": (-15.0,28.0,"Zambia",75),
    "ZWE": (-20.0,30.0,"Zimbabwe",45),           "SEN": (14.5,-14.0,"Senegal",85),
    "MLI": (17.0,-4.0,"Mali",60),                "BFA": (12.0,-2.0,"Burkina Faso",55),
    "NER": (17.0,8.0,"Niger",35),                "TCD": (15.0,19.0,"Chad",30),
    "SDN": (16.0,30.0,"Sudan",220),              "LBY": (27.0,17.0,"Libya",95),
    "TUN": (34.0,9.0,"Tunisia",170),             "DZA": (28.0,3.0,"Algeria",650),
    "MAR": (32.0,-5.0,"Morocco",400),            "COD": (-2.0,23.5,"DR Congo",120),
    "UGA": (1.0,32.0,"Uganda",150),              "RWA": (-2.0,30.0,"Rwanda",40),
    "SOM": (6.0,46.0,"Somalia",15),              "DJI": (11.8,42.5,"Djibouti",5),
    "GAB": (-1.0,11.7,"Gabon",45),               "COG": (-1.0,15.0,"Rep. Congo",25),
    "TGO": (8.0,1.0,"Togo",22),                  "BEN": (9.3,2.3,"Benin",55),
    "LBR": (6.5,-9.5,"Liberia",8),               "SLE": (8.5,-11.8,"Sierra Leone",14),
    "GIN": (11.0,-10.7,"Guinea",32),             "MRT": (21.0,-11.0,"Mauritania",28),
    "NAM": (-22.0,17.0,"Namibia",30),            "BWA": (-22.0,24.0,"Botswana",55),
    "MWI": (-13.5,34.0,"Malawi",35),             "SEN": (14.5,-14.0,"Senegal",85),
    # Americas
    "COL": (4.0,-73.0,"Colombia",950),           "VEN": (8.0,-66.0,"Venezuela",220),
    "PER": (-10.0,-76.0,"Peru",550),             "ECU": (-2.0,-77.5,"Ecuador",250),
    "BOL": (-17.0,-65.0,"Bolivia",130),          "PRY": (-23.0,-58.0,"Paraguay",110),
    "URY": (-33.0,-56.0,"Uruguay",120),          "GTM": (15.5,-90.0,"Guatemala",180),
    "HND": (15.0,-86.5,"Honduras",80),           "SLV": (13.7,-88.9,"El Salvador",70),
    "NIC": (13.0,-85.0,"Nicaragua",45),          "CRI": (10.0,-84.0,"Costa Rica",130),
    "PAN": (9.0,-80.0,"Panama",170),             "CUB": (22.0,-80.0,"Cuba",140),
    "DOM": (19.0,-70.7,"Dominican Republic",270), "HTI": (19.0,-72.3,"Haiti",35),
    "JAM": (18.2,-77.4,"Jamaica",30),            "TTO": (10.7,-61.4,"Trinidad & Tobago",50),
    "GUY": (5.0,-59.0,"Guyana",40),              "SUR": (4.0,-56.0,"Suriname",12),
    # Oceania
    "FJI": (-18.0,178.0,"Fiji",11),
}


def build_network_figure(G, metrics_df, stats, min_weight, max_nodes, color_by):
    """
    Geopolitical network overlaid on a world map.
    - Node position  = real geographic coordinates
    - Node SIZE      = GDP (PPP, 2023) — larger economy → bigger bubble
    - Node COLOR     = PageRank / Conflict Ratio / Community (user choice)
    - Edge color     = green (cooperative tone) / red (conflictual tone)
    - Edge width     = interaction frequency
    """
    community_map = stats.get("community_map", {})

    # Top N by pagerank, limited to countries we have geo for
    top_nodes = [
        n for n in metrics_df.head(max_nodes).index
        if n in COUNTRY_GEO
    ]
    H = G.subgraph(top_nodes).copy()

    # ── Edge traces (geo lines) ───────────────────────────────────────────
    edge_traces = []
    for u, v, d in H.edges(data=True):
        if d.get("weight", 0) < min_weight:
            continue
        if u not in COUNTRY_GEO or v not in COUNTRY_GEO:
            continue

        lat0, lon0 = COUNTRY_GEO[u][0], COUNTRY_GEO[u][1]
        lat1, lon1 = COUNTRY_GEO[v][0], COUNTRY_GEO[v][1]
        tone  = d.get("tone", 0)
        alpha = min(0.75, max(0.08, abs(tone) * 2.0))
        color = f"rgba(34,197,94,{alpha:.2f})"  if tone >= 0 else \
                f"rgba(239,68,68,{alpha:.2f})"
        width = max(0.4, min(d["weight"] / 18.0, 3.5))

        edge_traces.append(go.Scattergeo(
            lat=[lat0, lat1, None],
            lon=[lon0, lon1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Node trace ────────────────────────────────────────────────────────
    lats, lons, sizes, colors, labels, hover = [], [], [], [], [], []
    max_gdp = max(v[3] for v in COUNTRY_GEO.values())

    for node in H.nodes():
        if node not in COUNTRY_GEO:
            continue
        lat, lon, fullname, gdp = COUNTRY_GEO[node]
        lats.append(lat)
        lons.append(lon)

        # Size = GDP PPP (sqrt-scaled so small economies stay visible)
        sizes.append(8 + (gdp / max_gdp) ** 0.5 * 52)

        m  = metrics_df.loc[node] if node in metrics_df.index else {}
        pr = float(m.get("pagerank",       0.01)) if hasattr(m, "get") else 0.01
        cr = float(m.get("conflict_ratio", 0.0))  if hasattr(m, "get") else 0.0
        ev = int(m.get("total_events",     0))     if hasattr(m, "get") else 0

        if color_by == "PageRank":
            colors.append(pr)
        elif color_by == "Conflict Ratio":
            colors.append(cr)
        else:
            colors.append(float(community_map.get(node, 0)))

        labels.append(node)
        hover.append(
            f"<b>{fullname}</b><br>"
            f"GDP (PPP): ${gdp:.1f}T<br>"
            f"PageRank: {pr:.5f}<br>"
            f"Conflict ratio: {cr:.1%}<br>"
            f"Total events: {ev}"
        )

    cscale = "RdYlGn_r" if color_by == "Conflict Ratio" else "Viridis"

    node_trace = go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color="#e2e8f0"),
        hovertext=hover,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            sizemode="diameter",
            color=colors,
            colorscale=cscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=color_by, font=dict(color="#e2e8f0", size=11)),
                thickness=12, len=0.45,
                tickfont=dict(color="#94a3b8", size=9),
                bgcolor="rgba(17,24,39,0.8)",
                bordercolor="#1e293b",
            ),
            opacity=0.92,
            line=dict(width=1.2, color="#0f172a"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#1a2235",
        showocean=True,      oceancolor="#0d1421",
        showlakes=True,      lakecolor="#0d1421",
        showrivers=False,
        showcountries=True,  countrycolor="#253248",  countrywidth=0.5,
        showcoastlines=True, coastlinecolor="#2d3f5a", coastlinewidth=0.6,
        showframe=False,
        bgcolor="#0a0e1a",
    )

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(bgcolor="#0a0e1a"),
    )
    return fig


def build_radar_chart(metrics_df, country):
    """Build radar chart for a country's metrics."""
    if country not in metrics_df.index:
        return go.Figure()

    m = metrics_df.loc[country]

    # Normalize each metric to 0-1 range across all countries
    def norm(col):
        mn, mx = metrics_df[col].min(), metrics_df[col].max()
        return (float(m[col]) - mn) / (mx - mn) if mx > mn else 0.0

    categories = ["PageRank", "Betweenness", "Eigenvector", "In-Degree", "Out-Degree"]
    values = [
        norm("pagerank"),
        norm("betweenness"),
        norm("eigenvector"),
        norm("in_degree_weighted"),
        norm("out_degree_weighted"),
    ]
    values.append(values[0])  # Close polygon
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(0,212,255,0.15)",
        line=dict(color="#00d4ff", width=2),
        name=country,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title=dict(text=f"{country} — Influence Profile", font=dict(color="#00d4ff")),
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False,
    )
    return fig


def build_bilateral_chart(G, ca, cb):
    """Build bilateral event type breakdown chart."""
    categories = list(set(
        list(G[ca][cb].get("event_types", {}).keys()) +
        list(G[cb][ca].get("event_types", {}).keys())
    )) if G.has_edge(ca, cb) or G.has_edge(cb, ca) else []

    if not categories:
        return go.Figure()

    a_to_b_vals = [G[ca][cb]["event_types"].get(c, 0) if G.has_edge(ca, cb) else 0
                   for c in categories]
    b_to_a_vals = [G[cb][ca]["event_types"].get(c, 0) if G.has_edge(cb, ca) else 0
                   for c in categories]

    fig = go.Figure(data=[
        go.Bar(name=f"{ca} → {cb}", x=categories, y=a_to_b_vals,
               marker_color="#00d4ff", opacity=0.85),
        go.Bar(name=f"{cb} → {ca}", x=categories, y=b_to_a_vals,
               marker_color="#ff6b35", opacity=0.85),
    ])
    fig.update_layout(
        barmode="group",
        title=f"{ca} ↔ {cb} Event Breakdown",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=320,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def build_tone_heatmap(G, metrics_df, top_n=15):
    """Build country-to-country tone heatmap."""
    top_countries = metrics_df.head(top_n).index.tolist()
    n = len(top_countries)
    matrix = np.zeros((n, n))

    for i, c1 in enumerate(top_countries):
        for j, c2 in enumerate(top_countries):
            if G.has_edge(c1, c2):
                matrix[i][j] = G[c1][c2]["tone"]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=top_countries,
        y=top_countries,
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(
            title=dict(text="Tone", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0")
        ),
        hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Bilateral Tone Matrix (Green=Cooperative, Red=Conflictual)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=500,
    )
    return fig


if __name__ == "__main__":
    main()
