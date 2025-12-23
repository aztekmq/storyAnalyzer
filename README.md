Absolutely — below is a full, end-to-end setup you can drop into a repo:

* **Data schema** in **JSON**, **SQL**, and **Pandas**
* A **Python pipeline** that:

  * loads a story JSON
  * builds a **character interaction graph** (NetworkX)
  * generates **Plotly** visualizations (network + arcs)
  * exports graph JSON for **D3.js**
* A **story “fingerprint” vector** (fixed-length numeric features for ML clustering)
* A **worked example** using a real story: ***The Matrix (1999)***
  (Scores are *illustrative/subjective*—the point is to show how the system works.)

---

## 1) Data schema

### 1A) JSON schema (practical “data contract”)

This is a *compact* but algorithm-friendly structure.

```json
{
  "story_id": "matrix_1999",
  "meta": {
    "title": "The Matrix",
    "medium": "movie",
    "year": 1999,
    "genre_primary": "science fiction",
    "genre_secondary": ["action", "cyberpunk"],
    "length_minutes": 136,
    "setting": {
      "setting_type": "futuristic",
      "locations": ["The Matrix (simulated)", "Machine City (implied)", "Zion (implied)"],
      "time_span_days": 7
    }
  },
  "structure": {
    "model": "three_act",
    "beats": {
      "inciting_pct": 0.08,
      "turn1_pct": 0.25,
      "midpoint_pct": 0.52,
      "lowpoint_pct": 0.72,
      "climax_pct": 0.90,
      "resolution_pct": 0.10
    },
    "plot": {
      "major_plotlines": 1,
      "subplots": 2,
      "interweaving": 8,
      "causality_clarity": 8,
      "nonlinear_degree": 2
    }
  },
  "characters": [
    {"id": "neo", "name": "Neo", "role": "protagonist", "agency": 8, "moral": 4, "arc_type": "positive"},
    {"id": "morpheus", "name": "Morpheus", "role": "mentor", "agency": 7, "moral": 4, "arc_type": "flat"},
    {"id": "trinity", "name": "Trinity", "role": "ally", "agency": 7, "moral": 4, "arc_type": "flat"},
    {"id": "smith", "name": "Agent Smith", "role": "antagonist", "agency": 9, "moral": -4, "arc_type": "flat"},
    {"id": "cypher", "name": "Cypher", "role": "traitor", "agency": 6, "moral": -2, "arc_type": "negative"}
  ],
  "relationships": [
    {
      "a": "neo",
      "b": "morpheus",
      "type": "mentor",
      "intensity": 8,
      "volatility": 4,
      "power": -2,
      "outcome": "improved"
    },
    {
      "a": "neo",
      "b": "trinity",
      "type": "love",
      "intensity": 7,
      "volatility": 3,
      "power": 0,
      "outcome": "improved"
    },
    {
      "a": "neo",
      "b": "smith",
      "type": "enemy",
      "intensity": 9,
      "volatility": 8,
      "power": 0,
      "outcome": "escalated"
    },
    {
      "a": "morpheus",
      "b": "smith",
      "type": "enemy",
      "intensity": 8,
      "volatility": 7,
      "power": 1,
      "outcome": "improved"
    },
    {
      "a": "neo",
      "b": "cypher",
      "type": "betrayal",
      "intensity": 7,
      "volatility": 9,
      "power": 0,
      "outcome": "destroyed"
    }
  ],
  "timeseries": {
    "binning": {"bins": 10, "unit": "percent_of_story"},
    "emotion": {
      "joy":     [2,2,3,3,4,4,2,3,4,5],
      "fear":    [3,4,5,5,6,6,7,7,8,6],
      "anger":   [2,2,3,3,4,5,6,6,7,5],
      "sadness": [1,1,2,2,3,3,4,4,5,3],
      "hope":    [2,3,3,4,4,5,4,5,7,8],
      "tension": [3,4,5,6,6,7,8,8,9,6]
    },
    "pacing": {
      "pace": [5,6,6,7,6,6,7,8,9,6],
      "action_ratio": [0.4,0.5,0.6,0.6,0.5,0.5,0.6,0.7,0.8,0.5]
    }
  },
  "themes": [
    {"name": "reality_vs_illusion", "explicitness": 9, "subtlety": 4, "consistency": 9, "resolution": 8},
    {"name": "identity_and_choice", "explicitness": 8, "subtlety": 6, "consistency": 8, "resolution": 8},
    {"name": "control_and_freedom", "explicitness": 8, "subtlety": 5, "consistency": 8, "resolution": 7}
  ],
  "global_scores": {
    "emotional_range": 8,
    "emotional_volatility": 7,
    "stakes_clarity": 9,
    "stakes_escalation": 8,
    "coherence": 8,
    "engagement": 9,
    "ending_satisfaction": 8,
    "originality": 9,
    "predictability": 4,
    "worldbuilding": 9,
    "dialogue_quality": 7,
    "symbolism_density": 7
  }
}
```

---

### 1B) SQL schema (PostgreSQL style)

```sql
CREATE TABLE story (
  story_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  medium TEXT NOT NULL,
  year INT,
  genre_primary TEXT,
  length_minutes INT,
  length_pages INT
);

CREATE TABLE story_genre_secondary (
  story_id TEXT REFERENCES story(story_id),
  genre TEXT NOT NULL,
  PRIMARY KEY (story_id, genre)
);

CREATE TABLE character (
  story_id TEXT REFERENCES story(story_id),
  character_id TEXT,
  name TEXT NOT NULL,
  role TEXT,
  agency INT,
  moral INT,
  arc_type TEXT,
  PRIMARY KEY (story_id, character_id)
);

CREATE TABLE relationship (
  story_id TEXT REFERENCES story(story_id),
  rel_id SERIAL PRIMARY KEY,
  a_character_id TEXT NOT NULL,
  b_character_id TEXT NOT NULL,
  rel_type TEXT NOT NULL,
  intensity INT,
  volatility INT,
  power INT,
  outcome TEXT,
  FOREIGN KEY (story_id, a_character_id) REFERENCES character(story_id, character_id),
  FOREIGN KEY (story_id, b_character_id) REFERENCES character(story_id, character_id)
);

CREATE TABLE theme (
  story_id TEXT REFERENCES story(story_id),
  theme_name TEXT,
  explicitness INT,
  subtlety INT,
  consistency INT,
  resolution INT,
  PRIMARY KEY (story_id, theme_name)
);

CREATE TABLE timeseries_emotion (
  story_id TEXT REFERENCES story(story_id),
  bin_index INT,
  joy INT, fear INT, anger INT, sadness INT, hope INT, tension INT,
  PRIMARY KEY (story_id, bin_index)
);

CREATE TABLE timeseries_pacing (
  story_id TEXT REFERENCES story(story_id),
  bin_index INT,
  pace INT,
  action_ratio REAL,
  PRIMARY KEY (story_id, bin_index)
);

CREATE TABLE global_scores (
  story_id TEXT PRIMARY KEY REFERENCES story(story_id),
  emotional_range INT,
  emotional_volatility INT,
  stakes_clarity INT,
  stakes_escalation INT,
  coherence INT,
  engagement INT,
  ending_satisfaction INT,
  originality INT,
  predictability INT,
  worldbuilding INT,
  dialogue_quality INT,
  symbolism_density INT
);
```

---

### 1C) Pandas schema (recommended “dataframes”)

* `df_story` (1 row/story)
* `df_characters`
* `df_relationships`
* `df_emotion_ts`
* `df_pacing_ts`
* `df_themes`
* `df_global_scores`

(You’ll see the exact construction in the pipeline code below.)

---

## 2) Python pipeline (NetworkX + Plotly + D3 export)

This is a single-file script you can run.

```python
# story_graph_pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


# -----------------------------
# Load + normalize
# -----------------------------
def load_story_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    df_characters.insert(0, "story_id", story["story_id"])

    df_relationships = pd.DataFrame(story.get("relationships", []))
    df_relationships.insert(0, "story_id", story["story_id"])

    # timeseries: emotion
    emo = story["timeseries"]["emotion"]
    bins = len(next(iter(emo.values())))
    df_emotion_ts = pd.DataFrame({
        "story_id": [story["story_id"]] * bins,
        "bin": list(range(bins)),
        **emo
    })

    # timeseries: pacing
    pac = story["timeseries"]["pacing"]
    df_pacing_ts = pd.DataFrame({
        "story_id": [story["story_id"]] * bins,
        "bin": list(range(bins)),
        **pac
    })

    df_themes = pd.DataFrame(story.get("themes", []))
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


# -----------------------------
# Graph building
# -----------------------------
def build_character_graph(df_characters: pd.DataFrame, df_relationships: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    # nodes
    for _, row in df_characters.iterrows():
        G.add_node(
            row["id"],
            label=row["name"],
            role=row.get("role"),
            agency=int(row.get("agency", 0)),
            moral=int(row.get("moral", 0)),
            arc_type=row.get("arc_type"),
        )

    # edges
    for _, r in df_relationships.iterrows():
        a, b = r["a"], r["b"]
        if a not in G.nodes or b not in G.nodes:
            continue
        G.add_edge(
            a, b,
            rel_type=r.get("type"),
            intensity=int(r.get("intensity", 0)),
            volatility=int(r.get("volatility", 0)),
            power=int(r.get("power", 0)),
            outcome=r.get("outcome"),
            weight=float(r.get("intensity", 1))  # used for layout or centrality
        )

    return G


def export_d3_json(G: nx.Graph) -> Dict[str, Any]:
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({"id": n, **data})

    links = []
    for u, v, data in G.edges(data=True):
        links.append({"source": u, "target": v, **data})

    return {"nodes": nodes, "links": links}


# -----------------------------
# Plotly visualization
# -----------------------------
def plot_network(G: nx.Graph, title: str = "Character Network") -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=0.8, weight="weight")

    # edges
    edge_x, edge_y = [], []
    edge_text = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{G.nodes[u].get('label')} ↔ {G.nodes[v].get('label')} | {d.get('rel_type')} | intensity={d.get('intensity')}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none"
    )

    # nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(
            f"{data.get('label')}<br>"
            f"role={data.get('role')}<br>"
            f"agency={data.get('agency')} moral={data.get('moral')} arc={data.get('arc_type')}"
        )
        # size based on weighted degree
        node_size.append(10 + 4 * sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[G.nodes[n].get("label") for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_timeseries(df_emotion_ts: pd.DataFrame, title: str = "Emotion Arcs") -> go.Figure:
    fig = go.Figure()
    x = df_emotion_ts["bin"]

    for col in ["joy", "fear", "anger", "sadness", "hope", "tension"]:
        if col in df_emotion_ts.columns:
            fig.add_trace(go.Scatter(x=x, y=df_emotion_ts[col], mode="lines+markers", name=col))

    fig.update_layout(
        title=title,
        xaxis_title="Story bin (0..9)",
        yaxis_title="Intensity (0..10)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_tension_vs_pace(df_emotion_ts: pd.DataFrame, df_pacing_ts: pd.DataFrame, title: str = "Tension vs Pace") -> go.Figure:
    fig = go.Figure()
    x = df_emotion_ts["bin"]
    fig.add_trace(go.Scatter(x=x, y=df_emotion_ts["tension"], mode="lines+markers", name="tension"))
    fig.add_trace(go.Scatter(x=x, y=df_pacing_ts["pace"], mode="lines+markers", name="pace"))
    fig.update_layout(
        title=title,
        xaxis_title="Story bin",
        yaxis_title="Value",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# -----------------------------
# Fingerprint vector (ML features)
# -----------------------------
def summarize_ts(arr: np.ndarray) -> Dict[str, float]:
    """
    Produces stable features for clustering:
    - mean, std, slope, peak, peak_pos, area, volatility (mean abs diff)
    """
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0]) if n >= 2 else 0.0
    peak = float(arr.max()) if n else 0.0
    peak_pos = float(arr.argmax() / max(n - 1, 1)) if n else 0.0
    volatility = float(np.mean(np.abs(np.diff(arr)))) if n >= 2 else 0.0
    area = float(arr.mean())  # normalized area proxy
    return {
        "mean": float(arr.mean()) if n else 0.0,
        "std": float(arr.std()) if n else 0.0,
        "slope": slope,
        "peak": peak,
        "peak_pos": peak_pos,
        "area": area,
        "volatility": volatility,
    }


def build_fingerprint(
    df_story: pd.DataFrame,
    df_characters: pd.DataFrame,
    df_relationships: pd.DataFrame,
    df_emotion_ts: pd.DataFrame,
    df_pacing_ts: pd.DataFrame,
    df_themes: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (vector, feature_names) fixed-length.
    Great for clustering (k-means, HDBSCAN, etc.)
    """
    features: Dict[str, float] = {}

    # --- Global numeric scores from df_story
    row = df_story.iloc[0].to_dict()
    keep_prefixes = ["plot_", "beat_", "score_"]
    for k, v in row.items():
        if any(k.startswith(p) for p in keep_prefixes):
            if isinstance(v, (int, float, np.number)) and pd.notna(v):
                features[k] = float(v)

    # --- Character graph stats
    G = build_character_graph(df_characters, df_relationships)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    features["graph_nodes"] = float(n_nodes)
    features["graph_edges"] = float(n_edges)
    features["graph_density"] = float(nx.density(G)) if n_nodes > 1 else 0.0

    if n_nodes:
        degs = np.array([d for _, d in G.degree()], dtype=float)
        features["graph_degree_mean"] = float(degs.mean())
        features["graph_degree_std"] = float(degs.std())
    else:
        features["graph_degree_mean"] = 0.0
        features["graph_degree_std"] = 0.0

    if n_nodes > 2 and nx.is_connected(G):
        features["graph_avg_path"] = float(nx.average_shortest_path_length(G))
    else:
        features["graph_avg_path"] = 0.0

    # --- Relationship intensity distribution
    if len(df_relationships):
        features["rel_intensity_mean"] = float(df_relationships["intensity"].mean())
        features["rel_intensity_std"] = float(df_relationships["intensity"].std(ddof=0))
        features["rel_volatility_mean"] = float(df_relationships["volatility"].mean())
    else:
        features["rel_intensity_mean"] = 0.0
        features["rel_intensity_std"] = 0.0
        features["rel_volatility_mean"] = 0.0

    # --- Theme stats
    features["theme_count"] = float(len(df_themes))
    if len(df_themes):
        for col in ["explicitness", "subtlety", "consistency", "resolution"]:
            features[f"theme_{col}_mean"] = float(df_themes[col].mean())
            features[f"theme_{col}_std"] = float(df_themes[col].std(ddof=0))
    else:
        for col in ["explicitness", "subtlety", "consistency", "resolution"]:
            features[f"theme_{col}_mean"] = 0.0
            features[f"theme_{col}_std"] = 0.0

    # --- Timeseries summaries (emotion + pacing)
    for series_name in ["joy", "fear", "anger", "sadness", "hope", "tension"]:
        stats = summarize_ts(df_emotion_ts[series_name].to_numpy())
        for stat_name, val in stats.items():
            features[f"emo_{series_name}_{stat_name}"] = val

    stats = summarize_ts(df_pacing_ts["pace"].to_numpy())
    for stat_name, val in stats.items():
        features[f"pace_{stat_name}"] = val

    stats = summarize_ts(df_pacing_ts["action_ratio"].to_numpy())
    for stat_name, val in stats.items():
        features[f"actionratio_{stat_name}"] = val

    # stable ordering
    feature_names = sorted(features.keys())
    vector = np.array([features[k] for k in feature_names], dtype=float)
    return vector, feature_names


# -----------------------------
# Main runner
# -----------------------------
def run(path: str, out_prefix: str = "out_matrix") -> None:
    story = load_story_json(path)
    dfs = to_dataframes(story)

    G = build_character_graph(dfs["characters"], dfs["relationships"])

    # Visuals
    fig_net = plot_network(G, title=f"Character Network — {story['meta']['title']}")
    fig_emo = plot_timeseries(dfs["emotion_ts"], title=f"Emotion Arcs — {story['meta']['title']}")
    fig_tp = plot_tension_vs_pace(dfs["emotion_ts"], dfs["pacing_ts"], title=f"Tension vs Pace — {story['meta']['title']}")

    fig_net.write_html(f"{out_prefix}_network.html")
    fig_emo.write_html(f"{out_prefix}_emotion.html")
    fig_tp.write_html(f"{out_prefix}_tension_pace.html")

    # D3 export
    d3 = export_d3_json(G)
    with open(f"{out_prefix}_d3_graph.json", "w", encoding="utf-8") as f:
        json.dump(d3, f, indent=2)

    # Fingerprint
    vec, names = build_fingerprint(
        dfs["story"], dfs["characters"], dfs["relationships"],
        dfs["emotion_ts"], dfs["pacing_ts"], dfs["themes"]
    )
    fp = pd.DataFrame([vec], columns=names)
    fp.insert(0, "story_id", story["story_id"])
    fp.to_csv(f"{out_prefix}_fingerprint.csv", index=False)

    print("Wrote:")
    print(f"  {out_prefix}_network.html")
    print(f"  {out_prefix}_emotion.html")
    print(f"  {out_prefix}_tension_pace.html")
    print(f"  {out_prefix}_d3_graph.json")
    print(f"  {out_prefix}_fingerprint.csv")


if __name__ == "__main__":
    # Example:
    # python story_graph_pipeline.py matrix_story.json
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python story_graph_pipeline.py <story.json> [out_prefix]")
    in_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) >= 3 else "out_story"
    run(in_path, prefix)
```

---

## 3) D3.js usage (minimal example)

Once you have `out_matrix_d3_graph.json`, you can visualize it with D3 force layout. Here’s a minimal sketch:

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
<svg id="viz" width="900" height="600"></svg>
<script>
(async function () {
  const data = await d3.json("out_matrix_d3_graph.json");
  const svg = d3.select("#viz");
  const width = +svg.attr("width"), height = +svg.attr("height");

  const sim = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(140))
    .force("charge", d3.forceManyBody().strength(-420))
    .force("center", d3.forceCenter(width/2, height/2));

  const link = svg.append("g")
    .selectAll("line").data(data.links).enter().append("line")
    .attr("stroke", "#999").attr("stroke-opacity", 0.6);

  const node = svg.append("g")
    .selectAll("circle").data(data.nodes).enter().append("circle")
    .attr("r", 10).attr("fill", "#555")
    .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended));

  const label = svg.append("g")
    .selectAll("text").data(data.nodes).enter().append("text")
    .text(d => d.label).attr("font-size", 12).attr("dx", 12).attr("dy", 4);

  sim.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  });

  function dragstarted(event, d) { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
  function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
  function dragended(event, d) { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }
})();
</script>
</body>
</html>
```

---

## 4) Worked example: “The Matrix (1999)”

1. Save the example JSON above as:

`matrix_story.json`

2. Run:

```bash
pip install pandas numpy networkx plotly
python story_graph_pipeline.py matrix_story.json out_matrix
```

3. Outputs:

* `out_matrix_network.html` (interactive character network)
* `out_matrix_emotion.html` (emotion arcs)
* `out_matrix_tension_pace.html` (tension vs pace)
* `out_matrix_d3_graph.json` (for D3)
* `out_matrix_fingerprint.csv` (ML vector)

---

## 5) Clustering stories with the fingerprint vectors

If you have multiple `*_fingerprint.csv` files, concatenate them:

```python
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

fps = []
for p in glob.glob("*_fingerprint.csv"):
    fps.append(pd.read_csv(p))
df = pd.concat(fps, ignore_index=True)

story_ids = df["story_id"].values
X = df.drop(columns=["story_id"]).values

X = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=5, random_state=42, n_init="auto").fit_predict(X)

out = pd.DataFrame({"story_id": story_ids, "cluster": labels})
print(out.sort_values("cluster"))
```

---

If you want, I can also add:

* **scene/chapter-level ingestion** (so edges intensify per scene and you get *dynamic* network graphs over time)
* a **“story genome”** embedding that combines the fingerprint + bag-of-themes + relationship-type frequencies
* a **dashboard** (Plotly Dash) where you upload JSON and it auto-generates all graphs + exports.
