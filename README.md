# GeoIntel — Geopolitical Intelligence Pipeline

A modular, end-to-end pipeline for automated geopolitical analysis. Collects bilateral event data from GDELT, constructs directed country interaction graphs, computes network centrality metrics, classifies events with a fine-tunable DistilBERT model, and generates intelligence-style narratives via LLM. Includes an interactive Streamlit dashboard.

---

## System Architecture

```
GDELT / Mock Data
       |
       v
  [gdelt_collector.py]  — fetch, decode, filter, preprocess
       |
       v
  [graph_builder.py]    — directed weighted graph (NetworkX)
       |
       v
  [graph_builder.py]    — centrality metrics, GGPI, communities
       |
       v
  [event_classifier.py] — DistilBERT event type classifier
       |
       v
  [narrator.py]         — LLM narrative generation (Claude / GPT / offline)
       |
       v
  [app.py]              — Streamlit dashboard + Markdown report
```

---

## Features

- GDELT 1.0 ingestion with correct 58-column schema and Latin-1 encoding
- Directed weighted country interaction graph construction (NetworkX)
- Network metrics: PageRank, betweenness, eigenvector, in/out-degree, conflict ratio
- Global Geopolitical Polarization Index (GGPI)
- Community detection via greedy modularity maximization
- Monthly / quarterly / annual temporal graph snapshots
- DistilBERT fine-tuning for 5-class geopolitical event classification
- LLM narratives: bilateral summaries, country profiles, global executive summary
- Supports Anthropic Claude, OpenAI GPT, and a deterministic offline fallback
- Interactive Streamlit dashboard with world map, heatmap, radar chart, bilateral panel
- Offline operation: mock data generator and rule-based classifier require no external services

---

## Technology Stack

| Component | Library / Tool |
|---|---|
| Data ingestion | `requests`, `pandas` |
| Graph analysis | `networkx` |
| ML classification | `transformers`, `datasets`, `torch` |
| LLM integration | `anthropic`, `openai` |
| Dashboard | `streamlit`, `plotly` |
| Numerics | `numpy` |
| Serialization | `json`, CSV |

Python 3.9+ required.

---

## Project Structure

```
geointel/
├── main.py                    # Pipeline orchestrator
├── data/
│   └── gdelt_collector.py     # GDELT fetch, mock generator, preprocessor
├── analysis/
│   ├── graph_builder.py       # Graph construction and network analysis
│   ├── event_classifier.py    # DistilBERT event classifier
│   └── narrator.py            # LLM narrative generator
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── output/                    # Generated outputs (created at runtime)
│   ├── events_clean.csv
│   ├── edges.csv
│   ├── country_metrics.csv
│   ├── temporal_metrics.csv
│   ├── network_stats.json
│   ├── summaries.json
│   └── report.md
└── models/
    └── event_classifier/      # Saved DistilBERT checkpoint (after training)
```

---

## Installation

```bash
git clone https://github.com/your-org/geointel.git
cd geointel

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Minimum requirements (`requirements.txt`):**

```
pandas
numpy
networkx
requests
streamlit
plotly
torch
transformers
datasets
scikit-learn
anthropic        # optional — for Claude narratives
openai           # optional — for GPT narratives
```

---

## Configuration

API keys are read from environment variables. Neither key is required; the system falls back to offline operation automatically.

```bash
# Anthropic Claude (preferred)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI GPT (fallback)
export OPENAI_API_KEY="sk-..."
```

On Windows use `set` instead of `export`.

---

## Usage

### Running the Pipeline

**Quick start with mock data (no API keys, no network):**
```bash
python main.py
```

**GDELT live data, specific date range:**
```bash
python main.py --source gdelt --start 2024-01-01 --end 2024-03-31
```

**Mock data with LLM narratives:**
```bash
python main.py --llm --llm-provider anthropic
```

**Bilateral analysis between two countries:**
```bash
python main.py --bilateral USA CHN --llm
```

**Temporal granularity:**
```bash
python main.py --temporal quarter
```

**Full example with all options:**
```bash
python main.py \
  --source gdelt \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --llm \
  --llm-provider anthropic \
  --bilateral RUS UKR \
  --temporal month \
  --output ./output
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

Or via the pipeline flag (after running the pipeline first):
```bash
python main.py --dashboard
```

The dashboard reads pre-computed data from `output/events_clean.csv` by default, or can fetch live GDELT data or generate mock data directly.

---

## Example Commands

| Goal | Command |
|---|---|
| Run offline pipeline, 5000 events | `python main.py` |
| Run with 10,000 mock events | `python main.py --events 10000` |
| Use GDELT, last 30 days | `python main.py --source gdelt --start $(date -d '30 days ago' +%F) --end $(date +%F)` |
| Generate LLM summaries offline | `python main.py --llm --llm-provider offline` |
| Analyze USA-CHN bilateral | `python main.py --bilateral USA CHN --llm` |
| Train event classifier | `python -c "from analysis.event_classifier import *; c = GeopoliticalEventClassifier(); c.load_tokenizer_model(); c.train()"` |
| Launch dashboard only | `streamlit run dashboard/app.py` |

---

## Output Files

| File | Description |
|---|---|
| `output/events_clean.csv` | Preprocessed bilateral event records |
| `output/edges.csv` | Graph edge list with weights and attributes |
| `output/country_metrics.csv` | Per-country centrality metrics |
| `output/temporal_metrics.csv` | Time-series network statistics |
| `output/network_stats.json` | Global network statistics including GGPI |
| `output/summaries.json` | LLM-generated country profiles and narratives |
| `output/report.md` | Formatted Markdown intelligence report |
| `models/event_classifier/` | Saved DistilBERT checkpoint and label map |

---

## Future Improvements

- GDELT 2.0 and GKG integration for full article text
- ICEWS labeled data for classifier training
- Graph attention network (GAT) for dynamic influence modeling
- Empirical GGPI coefficient calibration against historical datasets
- REST API endpoint for programmatic access
- Docker container for one-command deployment
- Multi-language news source integration

---

## License

MIT License. See `LICENSE` for details.

GDELT data is provided by the GDELT Project under their own terms of use. See [gdeltproject.org](https://www.gdeltproject.org/) for details.
