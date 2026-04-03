"""
LLM Narrative Summarizer
=========================
Uses Claude / any OpenAI-compatible LLM to generate natural language
summaries of bilateral and country-level geopolitical relationships.

Can run with:
  - Anthropic Claude (via anthropic package)
  - OpenAI (via openai package)
  - Offline rule-based fallback (no API key needed)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import networkx as nx


# ─── Prompt Templates ────────────────────────────────────────────────────────

BILATERAL_PROMPT = """You are a geopolitical intelligence analyst. Based on the following quantitative relationship data between {country_a} and {country_b}, write a concise analytical summary (3-5 sentences).

Relationship Data:
- Total events analyzed: {total_events}
- Average sentiment tone: {avg_tone:.3f} (range: -1 very negative to +1 very positive)
- Relationship type: {relationship_type}
- Dominant event types from {country_a} to {country_b}: {types_a_to_b}
- Dominant event types from {country_b} to {country_a}: {types_b_to_a}
- Conflict event ratio: {conflict_ratio:.1%}
- Cooperation event ratio: {coop_ratio:.1%}

Write a professional intelligence-style summary focusing on:
1. Overall relationship character
2. Key interaction patterns
3. Potential drivers of tension or cooperation

Keep it factual, analytical, and 3-5 sentences."""


COUNTRY_PROFILE_PROMPT = """You are a geopolitical intelligence analyst. Based on the network metrics for {country}, write a country geopolitical profile (4-6 sentences).

Network Metrics:
- Global Influence (PageRank): {pagerank:.4f} (higher = more influential)
- Betweenness Centrality: {betweenness:.4f} (bridge role in global network)
- Eigenvector Centrality: {eigenvector:.4f} (connected to influential countries)
- Conflict Ratio: {conflict_ratio:.1%} (fraction of interactions that are conflictual)
- Average Outgoing Tone: {avg_out_tone:.3f}
- Total Interactions: {total_events}
- Top Interaction Partners: {top_partners}

Write a professional intelligence-style profile covering:
1. {country}'s role and influence in the global network
2. Whether it plays a bridging or peripheral role
3. Overall relationship posture (cooperative vs conflictual)
4. Key geopolitical relationships"""


NETWORK_SUMMARY_PROMPT = """You are a geopolitical intelligence analyst. Summarize the current state of the global geopolitical network based on these metrics:

Network Statistics:
- Countries (nodes): {nodes}
- Diplomatic interactions (edges): {edges}
- Network Density: {density:.4f}
- Average Sentiment: {avg_tone:.3f}
- Negative Interaction Ratio: {negative_ratio:.1%}
- Modularity (bloc formation): {modularity:.3f}
- Number of Geopolitical Blocs: {num_communities}
- Global Geopolitical Polarization Index (GGPI): {ggpi:.3f}

Top 5 Most Influential Countries: {top5}
Most Conflictual Pairs: {conflict_pairs}

Write a 5-7 sentence executive summary of global geopolitical dynamics, identifying key trends, power centers, and risk areas."""


# ─── LLM Client Wrapper ───────────────────────────────────────────────────────

class LLMClient:
    """Wrapper that tries Anthropic, then OpenAI, then falls back to rule-based."""

    def __init__(self, provider: str = "auto"):
        """
        provider: 'anthropic' | 'openai' | 'auto' | 'offline'
        """
        self.provider = provider
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "offline":
            self._client = "offline"
            return

        if self.provider in ("anthropic", "auto"):
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
                    self.provider = "anthropic"
                    print("✓ Using Anthropic Claude for narratives")
                    return
            except ImportError:
                pass

        if self.provider in ("openai", "auto"):
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._client = openai.OpenAI(api_key=api_key)
                    self.provider = "openai"
                    print("✓ Using OpenAI GPT for narratives")
                    return
            except ImportError:
                pass

        self._client = "offline"
        self.provider = "offline"
        print("ℹ No LLM API key found. Using rule-based narrative generation.")

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Send prompt to LLM and return response text."""
        if self._client == "offline":
            return self._offline_complete(prompt)

        try:
            if self.provider == "anthropic":
                import anthropic
                msg = self._client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return msg.content[0].text

            elif self.provider == "openai":
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content

        except Exception as e:
            print(f"LLM API error: {e}. Falling back to rule-based.")
            return self._offline_complete(prompt)

    def _offline_complete(self, prompt: str) -> str:
        """Rule-based narrative fallback (no LLM needed)."""
        # Extract key facts from prompt using simple parsing
        lines = prompt.split("\n")
        params = {}
        for line in lines:
            if ":" in line:
                k, _, v = line.partition(":")
                params[k.strip().lower()] = v.strip()

        # Check what kind of summary we're generating
        if "country_a" in prompt.lower() or "bilateral" in prompt.lower():
            return self._bilateral_narrative(params)
        elif "global geopolitical polarization" in prompt.lower():
            return self._network_narrative(params)
        else:
            return self._country_narrative(params)

    def _bilateral_narrative(self, params: dict) -> str:
        tone_str = params.get("average sentiment tone", "0")
        try:
            tone = float(tone_str.split("(")[0])
        except Exception:
            tone = 0.0

        rel_type = params.get("relationship type", "mixed")
        sentiment = "predominantly positive" if tone > 0.05 else \
                    "predominantly negative" if tone < -0.05 else "mixed"

        return (
            f"The bilateral relationship reflects {sentiment} dynamics with {rel_type.lower()} interactions. "
            f"Interaction patterns suggest a complex interplay of competing interests and occasional cooperation. "
            f"The overall sentiment tone ({tone:.3f}) indicates the relationship remains active but contested. "
            f"Both countries maintain significant diplomatic and strategic engagement across multiple domains."
        )

    def _country_narrative(self, params: dict) -> str:
        return (
            f"This country occupies a significant position in the global diplomatic network. "
            f"Its centrality metrics suggest it serves as an important node in international relations. "
            f"The conflict-cooperation balance reflects its broader strategic posture and regional influence. "
            f"Continued engagement across multiple partner countries underscores its global diplomatic reach."
        )

    def _network_narrative(self, params: dict) -> str:
        return (
            f"The global geopolitical network exhibits complex patterns of cooperation and conflict. "
            f"Several distinct geopolitical blocs have emerged, reflecting regional and ideological alignments. "
            f"Key power centers maintain high centrality, with significant influence over global dynamics. "
            f"The polarization index indicates meaningful fragmentation in the international system. "
            f"Heightened tensions in multiple dyadic relationships suggest ongoing structural instability."
        )


# ─── Main Summarizer Class ─────────────────────────────────────────────────

class GeopoliticalNarrator:
    """Generates LLM-powered narrative summaries for the dashboard."""

    def __init__(self, provider: str = "auto"):
        self.llm = LLMClient(provider=provider)
        self._cache = {}

    def summarize_bilateral(
        self,
        G: nx.DiGraph,
        country_a: str,
        country_b: str
    ) -> str:
        """Generate bilateral relationship summary."""
        cache_key = f"bilateral_{country_a}_{country_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Gather data
        a_to_b = G[country_a][country_b] if G.has_edge(country_a, country_b) else {}
        b_to_a = G[country_b][country_a] if G.has_edge(country_b, country_a) else {}

        total_events = a_to_b.get("num_events", 0) + b_to_a.get("num_events", 0)
        tones = [x["tone"] for x in [a_to_b, b_to_a] if x and "tone" in x]
        avg_tone = np.mean(tones) if tones else 0.0
        conflict = a_to_b.get("conflict_count", 0) + b_to_a.get("conflict_count", 0)
        coop = a_to_b.get("coop_count", 0) + b_to_a.get("coop_count", 0)
        total_cc = conflict + coop or 1

        if avg_tone > 0.1:
            rel_type = "Cooperative"
        elif avg_tone < -0.1:
            rel_type = "Conflictual"
        else:
            rel_type = "Neutral/Mixed"

        def fmt_types(edge):
            if not edge:
                return "No direct interactions recorded"
            types = edge.get("event_types", {})
            if not types:
                return edge.get("dominant_type", "Unknown")
            return ", ".join(f"{k} ({v})" for k, v in
                             sorted(types.items(), key=lambda x: -x[1])[:3])

        prompt = BILATERAL_PROMPT.format(
            country_a=country_a,
            country_b=country_b,
            total_events=total_events,
            avg_tone=avg_tone,
            relationship_type=rel_type,
            types_a_to_b=fmt_types(a_to_b),
            types_b_to_a=fmt_types(b_to_a),
            conflict_ratio=conflict / total_cc,
            coop_ratio=coop / total_cc,
        )

        result = self.llm.complete(prompt, max_tokens=400)
        self._cache[cache_key] = result
        return result

    def summarize_country(
        self,
        G: nx.DiGraph,
        country: str,
        metrics_df: pd.DataFrame
    ) -> str:
        """Generate country-level geopolitical profile."""
        cache_key = f"country_{country}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if country not in metrics_df.index:
            return f"Insufficient data for {country}."

        m = metrics_df.loc[country]

        # Find top partners by interaction volume
        neighbors = list(G.successors(country)) + list(G.predecessors(country))
        partner_counts = {}
        for n in set(neighbors):
            cnt = 0
            if G.has_edge(country, n):
                cnt += G[country][n].get("num_events", 0)
            if G.has_edge(n, country):
                cnt += G[n][country].get("num_events", 0)
            partner_counts[n] = cnt

        top_partners = ", ".join(
            [k for k, _ in sorted(partner_counts.items(), key=lambda x: -x[1])[:5]]
        )

        prompt = COUNTRY_PROFILE_PROMPT.format(
            country=country,
            pagerank=float(m.get("pagerank", 0)),
            betweenness=float(m.get("betweenness", 0)),
            eigenvector=float(m.get("eigenvector", 0)),
            conflict_ratio=float(m.get("conflict_ratio", 0)),
            avg_out_tone=float(m.get("avg_out_tone", 0)),
            total_events=int(m.get("total_events", 0)),
            top_partners=top_partners,
        )

        result = self.llm.complete(prompt, max_tokens=450)
        self._cache[cache_key] = result
        return result

    def summarize_network(
        self,
        G: nx.DiGraph,
        stats: Dict,
        metrics_df: pd.DataFrame
    ) -> str:
        """Generate global network executive summary."""
        top5 = ", ".join(metrics_df.head(5).index.tolist())

        # Find most conflictual pairs
        conflict_pairs = []
        for u, v, d in G.edges(data=True):
            if d.get("tone", 0) < -0.15:
                conflict_pairs.append(f"{u}-{v} (tone: {d['tone']:.2f})")
        conflict_pairs = ", ".join(conflict_pairs[:5]) if conflict_pairs else "None identified"

        prompt = NETWORK_SUMMARY_PROMPT.format(
            nodes=stats["nodes"],
            edges=stats["edges"],
            density=stats["density"],
            avg_tone=stats["avg_tone"],
            negative_ratio=stats["negative_edge_ratio"],
            modularity=stats["modularity"],
            num_communities=stats["num_communities"],
            ggpi=stats["ggpi"],
            top5=top5,
            conflict_pairs=conflict_pairs,
        )

        return self.llm.complete(prompt, max_tokens=600)

    def batch_summarize_countries(
        self,
        G: nx.DiGraph,
        metrics_df: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, str]:
        """Generate profiles for the top N countries."""
        summaries = {}
        top_countries = metrics_df.head(top_n).index.tolist()
        for i, country in enumerate(top_countries):
            print(f"  Summarizing {country} ({i+1}/{len(top_countries)})...")
            summaries[country] = self.summarize_country(G, country, metrics_df)
        return summaries

    def save_summaries(self, summaries: Dict[str, str], path: str = "summaries.json"):
        """Save all summaries to JSON."""
        with open(path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Summaries saved to {path}")

    def load_summaries(self, path: str = "summaries.json") -> Dict[str, str]:
        """Load cached summaries."""
        if os.path.exists(path):
            with open(path) as f:
                self._cache.update(json.load(f))
        return self._cache
