"""
Geopolitical Graph Builder
===========================
Constructs directed weighted graphs from preprocessed event data.
Supports static and temporal (snapshot) graph generation.
"""

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def build_graph(df: pd.DataFrame, weight_by: str = "frequency") -> nx.DiGraph:
    """
    Build a directed weighted graph from event data.

    Parameters
    ----------
    df : preprocessed event DataFrame
    weight_by : 'frequency' | 'tone' | 'mentions'

    Returns
    -------
    nx.DiGraph with edge attributes: weight, tone, event_types, num_events
    """
    G = nx.DiGraph()

    # Aggregate edges
    edge_data = defaultdict(lambda: {
        "events": 0,
        "tone_sum": 0.0,
        "mentions": 0,
        "event_types": defaultdict(int),
        "conflict_count": 0,
        "coop_count": 0,
    })

    for _, row in df.iterrows():
        src = row["Actor1CountryCode"]
        tgt = row["Actor2CountryCode"]
        key = (src, tgt)

        edge_data[key]["events"] += 1
        edge_data[key]["tone_sum"] += row["tone_norm"]
        edge_data[key]["mentions"] += int(row.get("NumMentions", 1))
        edge_data[key]["event_types"][row["event_type"]] += 1

        if row["event_type"] == "Conflict" or row["event_type"] == "Military/Conflict":
            edge_data[key]["conflict_count"] += 1
        elif row["event_type"] == "Cooperation" or row["event_type"] == "Trade/Aid":
            edge_data[key]["coop_count"] += 1

    # Add edges to graph
    for (src, tgt), attrs in edge_data.items():
        n = attrs["events"]
        avg_tone = attrs["tone_sum"] / n if n > 0 else 0.0

        if weight_by == "frequency":
            weight = n
        elif weight_by == "tone":
            weight = abs(avg_tone) * n
        else:  # mentions
            weight = attrs["mentions"]

        dominant_type = max(attrs["event_types"], key=attrs["event_types"].get, default="Unknown")

        G.add_edge(src, tgt,
                   weight=weight,
                   tone=round(avg_tone, 4),
                   num_events=n,
                   mentions=attrs["mentions"],
                   dominant_type=dominant_type,
                   event_types=dict(attrs["event_types"]),
                   conflict_count=attrs["conflict_count"],
                   coop_count=attrs["coop_count"])

    # Add node metadata
    for node in G.nodes():
        G.nodes[node]["label"] = node
        G.nodes[node]["total_events"] = (
            sum(d["num_events"] for _, _, d in G.out_edges(node, data=True)) +
            sum(d["num_events"] for _, _, d in G.in_edges(node, data=True))
        )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def build_temporal_graphs(
    df: pd.DataFrame,
    period: str = "month"
) -> Dict[str, nx.DiGraph]:
    """
    Build a dict of time-sliced graphs.

    Parameters
    ----------
    df : preprocessed DataFrame with 'date' column
    period : 'month' | 'quarter' | 'year'

    Returns
    -------
    dict of {period_label -> nx.DiGraph}
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if period == "month":
        df["period"] = df["date"].dt.to_period("M").astype(str)
    elif period == "quarter":
        df["period"] = df["date"].dt.to_period("Q").astype(str)
    else:
        df["period"] = df["date"].dt.year.astype(str)

    graphs = {}
    for p, group in df.groupby("period"):
        graphs[p] = build_graph(group)

    periods = sorted(graphs.keys())
    print(f"Built {len(graphs)} temporal graphs ({period}): {periods[0]} to {periods[-1]}")
    return graphs


def get_undirected(G: nx.DiGraph) -> nx.Graph:
    """Convert to undirected, combining edge weights."""
    UG = nx.Graph()
    for u, v, d in G.edges(data=True):
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += d["weight"]
            tones = [UG[u][v].get("tone", 0), d["tone"]]
            UG[u][v]["tone"] = sum(tones) / len(tones)
        else:
            UG.add_edge(u, v, **d)
    return UG


def compute_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute all graph centrality and network metrics.

    Returns DataFrame indexed by country with metric columns.
    """
    UG = get_undirected(G)

    metrics = {}

    # Centrality metrics
    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    deg_cent = nx.degree_centrality(G)
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))

    try:
        btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        btw = {n: 0.0 for n in G.nodes()}

    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    for node in G.nodes():
        # Edge tone aggregates
        out_tones = [d["tone"] for _, _, d in G.out_edges(node, data=True)]
        in_tones = [d["tone"] for _, _, d in G.in_edges(node, data=True)]

        avg_out_tone = np.mean(out_tones) if out_tones else 0.0
        avg_in_tone = np.mean(in_tones) if in_tones else 0.0

        # Conflict vs cooperation ratio
        total_conflict = sum(d["conflict_count"] for _, _, d in G.out_edges(node, data=True))
        total_coop = sum(d["coop_count"] for _, _, d in G.out_edges(node, data=True))
        total = total_conflict + total_coop
        conflict_ratio = total_conflict / total if total > 0 else 0.0

        metrics[node] = {
            "pagerank": round(pr.get(node, 0), 6),
            "degree_centrality": round(deg_cent.get(node, 0), 4),
            "betweenness": round(btw.get(node, 0), 6),
            "eigenvector": round(eig.get(node, 0), 6),
            "in_degree_weighted": round(in_deg.get(node, 0), 2),
            "out_degree_weighted": round(out_deg.get(node, 0), 2),
            "avg_out_tone": round(avg_out_tone, 4),
            "avg_in_tone": round(avg_in_tone, 4),
            "conflict_ratio": round(conflict_ratio, 4),
            "total_events": G.nodes[node].get("total_events", 0),
        }

    df_metrics = pd.DataFrame(metrics).T.sort_values("pagerank", ascending=False)
    df_metrics.index.name = "country"

    print(f"\nTop 10 countries by PageRank:")
    print(df_metrics[["pagerank", "betweenness", "conflict_ratio"]].head(10).to_string())

    return df_metrics


def compute_network_stats(G: nx.DiGraph) -> Dict:
    """Compute global network statistics."""
    UG = get_undirected(G)

    # Modularity via Louvain-style greedy community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(UG, weight="weight"))
        modularity = nx.algorithms.community.quality.modularity(
            UG, communities, weight="weight"
        )
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
    except Exception:
        modularity = 0.0
        community_map = {n: 0 for n in G.nodes()}

    # Negative edge ratio
    all_tones = [d["tone"] for _, _, d in G.edges(data=True)]
    neg_ratio = sum(1 for t in all_tones if t < 0) / len(all_tones) if all_tones else 0.0
    avg_tone = np.mean(all_tones) if all_tones else 0.0

    # Density
    density = nx.density(G)

    # Reciprocity
    reciprocity = nx.overall_reciprocity(G)

    # GGPI: Global Geopolitical Polarization Index
    # Higher = more polarized. Combines modularity, neg_ratio, and low avg_tone.
    ggpi = (
        0.4 * modularity +
        0.4 * neg_ratio +
        0.2 * max(0, -avg_tone)
    )

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(density, 4),
        "reciprocity": round(reciprocity, 4),
        "avg_tone": round(avg_tone, 4),
        "negative_edge_ratio": round(neg_ratio, 4),
        "modularity": round(modularity, 4),
        "num_communities": len(set(community_map.values())),
        "ggpi": round(ggpi, 4),
        "community_map": community_map,
    }

    print(f"\n📊 Network Statistics:")
    for k, v in stats.items():
        if k != "community_map":
            print(f"  {k:30s} {v}")

    return stats


def get_bilateral_summary(G: nx.DiGraph, country_a: str, country_b: str) -> Dict:
    """
    Extract bilateral relationship data between two countries.
    Returns dict with edge data in both directions.
    """
    result = {
        "country_a": country_a,
        "country_b": country_b,
        "a_to_b": None,
        "b_to_a": None,
        "relationship_type": "Neutral",
        "dominant_tone": 0.0,
    }

    if G.has_edge(country_a, country_b):
        result["a_to_b"] = G[country_a][country_b]

    if G.has_edge(country_b, country_a):
        result["b_to_a"] = G[country_b][country_a]

    tones = []
    if result["a_to_b"]:
        tones.append(result["a_to_b"]["tone"])
    if result["b_to_a"]:
        tones.append(result["b_to_a"]["tone"])

    if tones:
        avg = np.mean(tones)
        result["dominant_tone"] = round(avg, 4)
        if avg > 0.1:
            result["relationship_type"] = "Cooperative"
        elif avg < -0.1:
            result["relationship_type"] = "Conflictual"
        else:
            result["relationship_type"] = "Neutral/Mixed"

    return result
