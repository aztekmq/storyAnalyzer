## Directory structure diagram

```
story-graph-lab/
├─ README.md
├─ pyproject.toml                (or requirements.txt)
├─ .gitignore
│
├─ data/
│  ├─ samples/
│  │  └─ matrix_story.json       (worked example input)
│  └─ uploads/                   (dashboard saves uploaded JSON here)
│
├─ exports/
│  ├─ html/                      (plotly HTML outputs)
│  ├─ d3/                        (graph JSON for D3)
│  ├─ fingerprints/              (fingerprint CSV outputs)
│  └─ reports/                   (optional: future summary reports)
│
├─ src/
│  ├─ storygraph/
│  │  ├─ __init__.py
│  │  ├─ schema/
│  │  │  ├─ story.schema.json    (JSON Schema contract)
│  │  │  └─ sql/
│  │  │     └─ schema.sql        (Postgres DDL)
│  │  │
│  │  ├─ io/
│  │  │  ├─ load_json.py         (load/validate)
│  │  │  └─ normalize.py         (JSON → Pandas dataframes)
│  │  │
│  │  ├─ graph/
│  │  │  ├─ build_graph.py       (NetworkX graph construction)
│  │  │  └─ export_d3.py         (NetworkX → D3 JSON)
│  │  │
│  │  ├─ viz/
│  │  │  ├─ plot_network.py      (Plotly character network)
│  │  │  ├─ plot_emotion.py      (Plotly emotion arcs)
│  │  │  └─ plot_tension_pace.py (Plotly tension vs pace)
│  │  │
│  │  ├─ features/
│  │  │  ├─ fingerprint.py       (story fingerprint vector)
│  │  │  └─ ts_stats.py          (mean/std/slope/peak/etc)
│  │  │
│  │  ├─ pipeline/
│  │  │  └─ run_pipeline.py      (end-to-end runner + exporters)
│  │  │
│  │  └─ dashboard/
│  │     └─ app.py               (Plotly Dash upload dashboard)
│  │
│  └─ cli/
│     └─ storygraph_cli.py       (optional CLI entry point)
│
└─ tests/
   └─ test_fingerprint.py        (optional)
```

---

## “Which object goes where?” mapping

### Schemas

* **JSON schema** → `src/storygraph/schema/story.schema.json`
* **SQL schema (DDL)** → `src/storygraph/schema/sql/schema.sql`
* **Pandas “schema” (dataframe construction)** → `src/storygraph/io/normalize.py`

### Core pipeline objects

* JSON loader + (optional) validation → `src/storygraph/io/load_json.py`
* JSON → DataFrames normalization → `src/storygraph/io/normalize.py`
* NetworkX graph build → `src/storygraph/graph/build_graph.py`
* Export to D3 JSON → `src/storygraph/graph/export_d3.py`
* Plotly charts → `src/storygraph/viz/*.py`
* Fingerprint vector builder → `src/storygraph/features/fingerprint.py` and `ts_stats.py`
* End-to-end runner that writes outputs → `src/storygraph/pipeline/run_pipeline.py`

### Dashboard

* Plotly Dash app (upload JSON, generate graphs, export files) → `src/storygraph/dashboard/app.py`

### Data + exports

* Example input JSON(s) → `data/samples/`
* Dashboard uploads (saved) → `data/uploads/`
* Generated outputs:

  * HTML → `exports/html/`
  * D3 graph JSON → `exports/d3/`
  * Fingerprints CSV → `exports/fingerprints/`

---

## Minimal file contents (copy/paste)

### 1) `requirements.txt` (repo root)

```txt
pandas
numpy
networkx
plotly
dash
jsonschema
```

---

### 2) `src/storygraph/io/load_json.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import validate


def load_story_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_schema(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_story(story: Dict[str, Any], schema: Dict[str, Any]) -> None:
    # Raises jsonschema.ValidationError if invalid
    validate(instance=story, schema=schema)


def load_and_validate(story_path: str | Path, schema_path: Optional[str | Path] = None) -> Dict[str, Any]:
    story = load_story_json(story_path)
    if schema_path:
        schema = load_json_schema(schema_path)
        validate_story(story, schema)
    return story
```

---

### 3) `src/storygraph/io/normalize.py`

```python
from __future__ import annotations

from typing import Any, Dict
import pandas as pd


def to_dataframes(story: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    meta = story["meta"]
    structure = story["structure"]
    gs = story["global_scores"]

    df_story = pd.DataFrame([{
        "story_id": story["story_id"],
        "title": meta.get("title"),
        "medium": meta.get("medium"),
        "year": meta.get("year"),
        "genre_primary": meta.get("genre_primary"),
        "length_minutes": meta.get("length_minutes"),
        "plot_major_plotlines": structure["plot"]["major_plotlines"],
        "plot_subplots": structure["plot"]["subplots"],
        "plot_interweaving": structure["plot"]["interweaving"],
        "plot_causality_clarity": structure["plot"]["causality_clarity"],
        "plot_nonlinear_degree": structure["plot"]["nonlinear_degree"],
        **{f"beat_{k}": v for k, v in structure["beats"].items()},
        **{f"score_{k}": v for k, v in gs.items()},
    }])

    df_characters = pd.DataFrame(story.get("characters", []))
    if not df_characters.empty:
        df_characters.insert(0, "story_id", story["story_id"])

    df_relationships = pd.DataFrame(story.get("relationships", []))
    if not df_relationships.empty:
        df_relationships.insert(0, "story_id", story["story_id"])

    emo = story["timeseries"]["emotion"]
    bins = len(next(iter(emo.values())))
    df_emotion_ts = pd.DataFrame({"story_id": [story["story_id"]] * bins, "bin": list(range(bins)), **emo})

    pac = story["timeseries"]["pacing"]
    df_pacing_ts = pd.DataFrame({"story_id": [story["story_id"]] * bins, "bin": list(range(bins)), **pac})

    df_themes = pd.DataFrame(story.get("themes", []))
    if not df_themes.empty:
        df_themes.insert(0, "story_id", story["story_id"])

    df_global_scores = pd.DataFrame([{"story_id": story["story_id"], **gs}])

    return {
        "story": df_story,
        "characters": df_characters,
        "relationships": df_relationships,
        "emotion_ts": df_emotion_ts,
        "pacing_ts": df_pacing_ts,
        "themes": df_themes,
        "global_scores": df_global_scores,
    }
```

---

### 4) `src/storygraph/graph/build_graph.py`

```python
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_character_graph(df_characters: pd.DataFrame, df_relationships: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    if df_characters is not None and not df_characters.empty:
        for _, row in df_characters.iterrows():
            G.add_node(
                row["id"],
                label=row.get("name"),
                role=row.get("role"),
                agency=int(row.get("agency", 0) or 0),
                moral=int(row.get("moral", 0) or 0),
                arc_type=row.get("arc_type"),
            )

    if df_relationships is not None and not df_relationships.empty:
        for _, r in df_relationships.iterrows():
            a, b = r.get("a"), r.get("b")
            if a not in G.nodes or b not in G.nodes:
                continue
            intensity = int(r.get("intensity", 0) or 0)
            G.add_edge(
                a, b,
                rel_type=r.get("type"),
                intensity=intensity,
                volatility=int(r.get("volatility", 0) or 0),
                power=int(r.get("power", 0) or 0),
                outcome=r.get("outcome"),
                weight=float(intensity if intensity > 0 else 1.0),
            )

    return G
```

---

### 5) `src/storygraph/graph/export_d3.py`

```python
from __future__ import annotations

from typing import Any, Dict
import networkx as nx


def export_d3_json(G: nx.Graph) -> Dict[str, Any]:
    nodes = [{"id": n, **data} for n, data in G.nodes(data=True)]
    links = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    return {"nodes": nodes, "links": links}
```

---

### 6) `src/storygraph/viz/plot_network.py`

```python
from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go


def plot_network(G: nx.Graph, title: str) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(title=title)

    pos = nx.spring_layout(G, seed=42, k=0.8, weight="weight")

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none")

    node_x, node_y, node_text, node_size = [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(
            f"{data.get('label')}<br>"
            f"role={data.get('role')}<br>"
            f"agency={data.get('agency')} moral={data.get('moral')} arc={data.get('arc_type')}"
        )
        node_size.append(10 + 4 * sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[G.nodes[n].get("label") for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    return fig
```

---

### 7) `src/storygraph/viz/plot_emotion.py`

```python
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_emotion_arcs(df_emotion_ts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    x = df_emotion_ts["bin"]
    for col in ["joy", "fear", "anger", "sadness", "hope", "tension"]:
        if col in df_emotion_ts.columns:
            fig.add_trace(go.Scatter(x=x, y=df_emotion_ts[col], mode="lines+markers", name=col))
    fig.update_layout(title=title, xaxis_title="Story bin", yaxis_title="Intensity (0..10)",
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig
```

---

### 8) `src/storygraph/viz/plot_tension_pace.py`

```python
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_tension_vs_pace(df_emotion_ts: pd.DataFrame, df_pacing_ts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    x = df_emotion_ts["bin"]
    fig.add_trace(go.Scatter(x=x, y=df_emotion_ts["tension"], mode="lines+markers", name="tension"))
    fig.add_trace(go.Scatter(x=x, y=df_pacing_ts["pace"], mode="lines+markers", name="pace"))
    fig.update_layout(title=title, xaxis_title="Story bin", yaxis_title="Value",
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig
```

---

### 9) `src/storygraph/features/ts_stats.py`

```python
from __future__ import annotations

from typing import Dict
import numpy as np


def summarize_ts(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    x = np.arange(n, dtype=float)

    slope = float(np.polyfit(x, arr, 1)[0]) if n >= 2 else 0.0
    peak = float(arr.max()) if n else 0.0
    peak_pos = float(arr.argmax() / max(n - 1, 1)) if n else 0.0
    volatility = float(np.mean(np.abs(np.diff(arr)))) if n >= 2 else 0.0
    area = float(arr.mean()) if n else 0.0

    return {
        "mean": float(arr.mean()) if n else 0.0,
        "std": float(arr.std()) if n else 0.0,
        "slope": slope,
        "peak": peak,
        "peak_pos": peak_pos,
        "area": area,
        "volatility": volatility,
    }
```

---

### 10) `src/storygraph/features/fingerprint.py`

```python
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from storygraph.graph.build_graph import build_character_graph
from storygraph.features.ts_stats import summarize_ts


def build_fingerprint(
    df_story: pd.DataFrame,
    df_characters: pd.DataFrame,
    df_relationships: pd.DataFrame,
    df_emotion_ts: pd.DataFrame,
    df_pacing_ts: pd.DataFrame,
    df_themes: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    features: Dict[str, float] = {}

    row = df_story.iloc[0].to_dict()
    for k, v in row.items():
        if any(k.startswith(p) for p in ["plot_", "beat_", "score_"]):
            if isinstance(v, (int, float, np.number)) and pd.notna(v):
                features[k] = float(v)

    G = build_character_graph(df_characters, df_relationships)
    n_nodes = G.number_of_nodes()
    features["graph_nodes"] = float(n_nodes)
    features["graph_edges"] = float(G.number_of_edges())
    features["graph_density"] = float(nx.density(G)) if n_nodes > 1 else 0.0

    if n_nodes:
        degs = np.array([d for _, d in G.degree()], dtype=float)
        features["graph_degree_mean"] = float(degs.mean())
        features["graph_degree_std"] = float(degs.std())
    else:
        features["graph_degree_mean"] = 0.0
        features["graph_degree_std"] = 0.0

    if len(df_relationships):
        features["rel_intensity_mean"] = float(df_relationships["intensity"].mean())
        features["rel_intensity_std"] = float(df_relationships["intensity"].std(ddof=0))
        features["rel_volatility_mean"] = float(df_relationships["volatility"].mean())
    else:
        features["rel_intensity_mean"] = 0.0
        features["rel_intensity_std"] = 0.0
        features["rel_volatility_mean"] = 0.0

    features["theme_count"] = float(len(df_themes))
    if len(df_themes):
        for col in ["explicitness", "subtlety", "consistency", "resolution"]:
            features[f"theme_{col}_mean"] = float(df_themes[col].mean())
            features[f"theme_{col}_std"] = float(df_themes[col].std(ddof=0))
    else:
        for col in ["explicitness", "subtlety", "consistency", "resolution"]:
            features[f"theme_{col}_mean"] = 0.0
            features[f"theme_{col}_std"] = 0.0

    for series_name in ["joy", "fear", "anger", "sadness", "hope", "tension"]:
        stats = summarize_ts(df_emotion_ts[series_name].to_numpy())
        for stat_name, val in stats.items():
            features[f"emo_{series_name}_{stat_name}"] = float(val)

    for col, prefix in [("pace", "pace"), ("action_ratio", "actionratio")]:
        stats = summarize_ts(df_pacing_ts[col].to_numpy())
        for stat_name, val in stats.items():
            features[f"{prefix}_{stat_name}"] = float(val)

    feature_names = sorted(features.keys())
    vector = np.array([features[k] for k in feature_names], dtype=float)
    return vector, feature_names
```

---

### 11) `src/storygraph/pipeline/run_pipeline.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from storygraph.io.load_json import load_and_validate
from storygraph.io.normalize import to_dataframes
from storygraph.graph.build_graph import build_character_graph
from storygraph.graph.export_d3 import export_d3_json
from storygraph.viz.plot_network import plot_network
from storygraph.viz.plot_emotion import plot_emotion_arcs
from storygraph.viz.plot_tension_pace import plot_tension_vs_pace
from storygraph.features.fingerprint import build_fingerprint


def ensure_dirs(base: Path) -> Dict[str, Path]:
    paths = {
        "html": base / "exports" / "html",
        "d3": base / "exports" / "d3",
        "fingerprints": base / "exports" / "fingerprints",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def run_pipeline(story_path: str | Path, out_prefix: str, repo_root: str | Path = ".") -> Dict[str, Path]:
    repo_root = Path(repo_root).resolve()
    out_dirs = ensure_dirs(repo_root)

    schema_path = repo_root / "src" / "storygraph" / "schema" / "story.schema.json"
    story: Dict[str, Any] = load_and_validate(story_path, schema_path=str(schema_path))

    dfs = to_dataframes(story)
    G = build_character_graph(dfs["characters"], dfs["relationships"])

    title = story["meta"]["title"]

    fig_net = plot_network(G, f"Character Network — {title}")
    fig_emo = plot_emotion_arcs(dfs["emotion_ts"], f"Emotion Arcs — {title}")
    fig_tp = plot_tension_vs_pace(dfs["emotion_ts"], dfs["pacing_ts"], f"Tension vs Pace — {title}")

    net_html = out_dirs["html"] / f"{out_prefix}_network.html"
    emo_html = out_dirs["html"] / f"{out_prefix}_emotion.html"
    tp_html = out_dirs["html"] / f"{out_prefix}_tension_pace.html"

    fig_net.write_html(str(net_html))
    fig_emo.write_html(str(emo_html))
    fig_tp.write_html(str(tp_html))

    d3_json = out_dirs["d3"] / f"{out_prefix}_graph.json"
    with d3_json.open("w", encoding="utf-8") as f:
        json.dump(export_d3_json(G), f, indent=2)

    vec, names = build_fingerprint(
        dfs["story"], dfs["characters"], dfs["relationships"],
        dfs["emotion_ts"], dfs["pacing_ts"], dfs["themes"]
    )
    fp = pd.DataFrame([vec], columns=names)
    fp.insert(0, "story_id", story["story_id"])

    fp_csv = out_dirs["fingerprints"] / f"{out_prefix}_fingerprint.csv"
    fp.to_csv(fp_csv, index=False)

    return {
        "network_html": net_html,
        "emotion_html": emo_html,
        "tension_pace_html": tp_html,
        "d3_json": d3_json,
        "fingerprint_csv": fp_csv,
    }
```

---

## Plotly Dash dashboard (upload JSON → auto graphs + exports)

### 12) `src/storygraph/dashboard/app.py`

```python
from __future__ import annotations

import base64
from pathlib import Path
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import plotly.io as pio

from storygraph.pipeline.run_pipeline import run_pipeline
from storygraph.io.load_json import load_story_json
from storygraph.io.normalize import to_dataframes
from storygraph.graph.build_graph import build_character_graph
from storygraph.viz.plot_network import plot_network
from storygraph.viz.plot_emotion import plot_emotion_arcs
from storygraph.viz.plot_tension_pace import plot_tension_vs_pace


REPO_ROOT = Path(__file__).resolve().parents[3]  # story-graph-lab/
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = dash.Dash(__name__)
app.title = "StoryGraph Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "20px auto", "fontFamily": "system-ui"},
    children=[
        html.H2("StoryGraph Dashboard"),
        html.Div("Upload a story JSON to auto-generate graphs + exports (HTML, D3 JSON, fingerprint CSV)."),

        dcc.Upload(
            id="upload-json",
            children=html.Div(["Drag & drop or ", html.A("select a story JSON file")]),
            style={
                "width": "100%", "height": "70px", "lineHeight": "70px",
                "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "10px",
                "textAlign": "center", "marginTop": "12px"
            },
            multiple=False
        ),

        html.Div(style={"marginTop": "12px"}, children=[
            html.Label("Output prefix (optional)"),
            dcc.Input(id="out-prefix", type="text", placeholder="e.g., matrix_run1", style={"width": "320px"}),
            html.Button("Generate", id="btn-generate", n_clicks=0, style={"marginLeft": "10px"}),
        ]),

        html.Hr(),

        html.Div(id="status", style={"whiteSpace": "pre-wrap"}),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "18px", "marginTop": "16px"}, children=[
            dcc.Graph(id="fig-network"),
            dcc.Graph(id="fig-emotion"),
            dcc.Graph(id="fig-tension-pace"),
        ]),
    ]
)


def _save_upload(contents: str, filename: str) -> Path:
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace(" ", "_")
    path = UPLOAD_DIR / f"{ts}__{safe_name}"
    path.write_bytes(decoded)
    return path


@app.callback(
    Output("status", "children"),
    Output("fig-network", "figure"),
    Output("fig-emotion", "figure"),
    Output("fig-tension-pace", "figure"),
    Input("btn-generate", "n_clicks"),
    State("upload-json", "contents"),
    State("upload-json", "filename"),
    State("out-prefix", "value"),
    prevent_initial_call=True
)
def generate(n_clicks: int, contents: str | None, filename: str | None, out_prefix: str | None):
    if not contents or not filename:
        return "No file uploaded.", {}, {}, {}

    # Save upload
    saved_path = _save_upload(contents, filename)

    # Compute live figures (immediate display)
    story = load_story_json(saved_path)
    dfs = to_dataframes(story)
    G = build_character_graph(dfs["characters"], dfs["relationships"])

    title = story["meta"]["title"]
    fig_net = plot_network(G, f"Character Network — {title}")
    fig_emo = plot_emotion_arcs(dfs["emotion_ts"], f"Emotion Arcs — {title}")
    fig_tp = plot_tension_vs_pace(dfs["emotion_ts"], dfs["pacing_ts"], f"Tension vs Pace — {title}")

    # Exports (HTML, D3 JSON, fingerprint CSV)
    prefix = (out_prefix or story.get("story_id") or "story").strip()
    outputs = run_pipeline(saved_path, out_prefix=prefix, repo_root=REPO_ROOT)

    status = (
        f"Uploaded: {saved_path}\n\n"
        f"Exports written:\n"
        f"- Network HTML: {outputs['network_html']}\n"
        f"- Emotion HTML: {outputs['emotion_html']}\n"
        f"- Tension/Pace HTML: {outputs['tension_pace_html']}\n"
        f"- D3 graph JSON: {outputs['d3_json']}\n"
        f"- Fingerprint CSV: {outputs['fingerprint_csv']}\n"
    )
    return status, fig_net, fig_emo, fig_tp


if __name__ == "__main__":
    app.run(debug=True)
```

---

## How to run

### CLI pipeline

```bash
pip install -r requirements.txt
python -c "from storygraph.pipeline.run_pipeline import run_pipeline; print(run_pipeline('data/samples/matrix_story.json', 'matrix_demo', repo_root='.'))"
```

### Dashboard

```bash
pip install -r requirements.txt
python src/storygraph/dashboard/app.py
```
