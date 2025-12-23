# StoryGraph Lab

**Algorithmic story analysis → graphs → exports → ML fingerprints**

StoryGraph Lab is a professional-grade framework for analyzing **movies, books, TV episodes, and short stories** as structured data.  
It converts a single **story JSON** into:

- Interactive **character networks**
- **Emotional and pacing time-series graphs**
- **D3-ready graph data**
- A fixed-length **story fingerprint vector** for clustering, similarity, and ML

It supports both:
- a **CLI pipeline** for batch processing
- a **Plotly Dash dashboard** for interactive uploads and visualization

---

## Table of Contents

- [Why StoryGraph Lab](#why-storygraph-lab)
- [What You Get](#what-you-get)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Input Format](#input-format)
- [Sample Output Artifacts](#sample-output-artifacts)
- [CLI Pipeline](#cli-pipeline)
- [Dashboard](#dashboard)
- [Outputs](#outputs)
- [Story Fingerprint (ML Features)](#story-fingerprint-ml-features)
- [D3.js Integration](#d3js-integration)
- [SQL Schema](#sql-schema)
- [Development & Installation](#development--installation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why StoryGraph Lab

Most story analysis tools are qualitative. StoryGraph Lab is **quantitative-first**:

- Stories become **graphs, vectors, and time series**
- Narrative structure becomes **comparable across works**
- Results are **machine-learning ready**
- Visuals remain **human-readable and explorable**

Use cases include:
- Narrative analysis & research
- Content similarity & recommendation systems
- Creative tooling for writers
- Media analytics & dashboards
- Experimental storytelling + AI pipelines

---

## What You Get

From **one story JSON**, StoryGraph Lab produces:

### Visual Outputs (Plotly)
- Character interaction network
- Emotional arcs across story progression
- Tension vs pacing comparison

### Data Exports
- Plotly HTML dashboards (shareable)
- D3-compatible graph JSON
- CSV fingerprint vectors for ML

### Interfaces
- CLI for automation & batch runs
- Web dashboard for upload → visualize → export

---

## Quick Start

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows

pip install -r requirements.txt
````

### 2) Run the pipeline on a sample story

```bash
python -c "from storygraph.pipeline.run_pipeline import run_pipeline; print(run_pipeline('data/samples/matrix_story.json', 'matrix_demo', repo_root='.'))"
```

### 3) Open the generated HTML files

```text
exports/html/matrix_demo_network.html
exports/html/matrix_demo_emotion.html
exports/html/matrix_demo_tension_pace.html
```

### 4) Launch the dashboard

```bash
python src/storygraph/dashboard/app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:8050`).

---

## Directory Structure

```
story-graph-lab/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ samples/                 # Example input stories
│  │  └─ matrix_story.json
│  └─ uploads/                 # Dashboard uploads (auto-created)
│
├─ exports/
│  ├─ html/                    # Plotly HTML outputs
│  ├─ d3/                      # D3 graph JSON
│  ├─ fingerprints/            # ML-ready fingerprint CSVs
│  └─ reports/                 # Reserved for future summaries
│
├─ src/
│  └─ storygraph/
│     ├─ schema/               # JSON + SQL contracts
│     ├─ io/                   # Load + normalize
│     ├─ graph/                # NetworkX graph logic
│     ├─ viz/                  # Plotly visualizations
│     ├─ features/             # Fingerprint feature extraction
│     ├─ pipeline/             # End-to-end runner
│     └─ dashboard/            # Plotly Dash app
│
└─ tests/                      # Optional unit tests
```

---

## Input Format

StoryGraph Lab consumes a **single JSON file per story**, validated against:

```
src/storygraph/schema/story.schema.json
```

### Required Top-Level Fields

* `story_id`
* `meta`
* `structure`
* `characters`
* `relationships`
* `timeseries`
* `themes`
* `global_scores`

### Key Constraints

* Character IDs referenced in `relationships` **must exist**
* All emotion and pacing arrays must be the **same length**
* Numeric values must be numbers (not strings)

A complete, working example is included:

```
data/samples/matrix_story.json
```

---

## Sample Output Artifacts

Below are **real outputs** generated from the included Matrix example.

### 1) Character Network (Plotly HTML)

**File:** `exports/html/matrix_demo_network.html`

* Nodes = characters
* Node size = relationship centrality
* Edges = relationships (weighted by intensity)

---

### 2) Emotion Arcs Over Story Progression

**File:** `exports/html/matrix_demo_emotion.html`

* Multi-line time series (joy, fear, anger, sadness, hope, tension)
* X-axis = story progression bins (default: 10)

---

### 3) Tension vs Pace Comparison

**File:** `exports/html/matrix_demo_tension_pace.html`

* Overlaid curves show how narrative tension tracks pacing
* Useful for structure and rhythm analysis

---

### 4) D3 Graph Export

**File:** `exports/d3/matrix_demo_graph.json`

```json
{
  "nodes": [{ "id": "neo", "label": "Neo", "role": "protagonist" }],
  "links": [{ "source": "neo", "target": "smith", "intensity": 9 }]
}
```

Use directly in D3 force layouts or other graph tools.

---

### 5) Story Fingerprint Vector (CSV)

**File:** `exports/fingerprints/matrix_demo_fingerprint.csv`

* One row per story
* Hundreds of numeric features
* Designed for clustering, similarity search, and ML pipelines

---

## CLI Pipeline

Implemented in:

```
src/storygraph/pipeline/run_pipeline.py
```

### What it does

1. Loads and validates story JSON
2. Normalizes into Pandas DataFrames
3. Builds character graph (NetworkX)
4. Generates Plotly figures
5. Writes HTML, D3 JSON, and fingerprint CSV

### Example

```bash
python -c "from storygraph.pipeline.run_pipeline import run_pipeline; print(run_pipeline('data/samples/matrix_story.json', 'matrix_demo', repo_root='.'))"
```

---

## Dashboard

The dashboard provides a **zero-code interface**.

**Location:**

```
src/storygraph/dashboard/app.py
```

### Features

* Drag & drop story JSON upload
* Instant interactive graphs
* Automatic export to `exports/`
* Saved uploads for reproducibility

### Run

```bash
python src/storygraph/dashboard/app.py
```

---

## Outputs

| Type            | Directory               |
| --------------- | ----------------------- |
| Plotly HTML     | `exports/html/`         |
| D3 graph JSON   | `exports/d3/`           |
| Fingerprint CSV | `exports/fingerprints/` |

All outputs are deterministic given the same input JSON.

---

## Story Fingerprint (ML Features)

The fingerprint is a **fixed-length numeric vector** created by:

```
src/storygraph/features/fingerprint.py
```

### Feature Categories

* Plot structure & beats
* Graph topology statistics
* Relationship intensity distributions
* Theme aggregation metrics
* Time-series summaries (mean, std, slope, peak, volatility)

### ML Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd, glob

df = pd.concat([pd.read_csv(p) for p in glob.glob("exports/fingerprints/*.csv")])
X = StandardScaler().fit_transform(df.drop(columns=["story_id"]))
labels = KMeans(n_clusters=5).fit_predict(X)
```

---

## D3.js Integration

Graph exports are compatible with D3 force layouts:

```
exports/d3/<prefix>_graph.json
```

Structure:

* `nodes[]`
* `links[]`

No transformation required.

---

## SQL Schema

A PostgreSQL-style relational schema is provided in:

```
src/storygraph/schema/sql/schema.sql
```

Use this if you want to persist stories, graphs, and metrics in a database.

---

## Development & Installation

### Python path

Run from repo root or set:

```bash
export PYTHONPATH=src
```

### Editable install (optional)

If you add `pyproject.toml`:

```bash
pip install -e .
```

---

## Testing

Optional tests live under:

```
tests/
```

Recommended focus areas:

* Fingerprint vector stability
* Schema validation failures
* Empty or sparse story edge cases

---

## Troubleshooting

**Module not found: `storygraph`**

* Ensure `PYTHONPATH=src`
* Or run from repo root

**Dashboard uploads fail**

* Check schema validation errors in terminal
* Ensure JSON matches contract

**Blank Plotly HTML**

* Confirm numeric arrays
* Open in modern browser

---

## Roadmap

Planned extensions:

* Scene/chapter-level granularity
* Dynamic graphs over time
* ZIP bundle exports
* Story similarity search API
* Dashboard comparison mode
* Automatic bin detection

---

## License

Choose one and add a `LICENSE` file:

* MIT
* Apache-2.0
* Proprietary / Internal

---

**StoryGraph Lab turns narrative into data — without losing meaning.**
