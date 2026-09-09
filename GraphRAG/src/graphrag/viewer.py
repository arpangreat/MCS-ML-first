"""Generates a modern, clean, high-readability interactive Knowledge Graph viewer."""

from __future__ import annotations

import json
from pathlib import Path

CATEGORY_THEMES = {
    "PERSON": {"bg": "#1d4ed8", "border": "#60a5fa", "text": "#ffffff"},        # Vibrant Blue
    "ORGANIZATION": {"bg": "#b91c1c", "border": "#f87171", "text": "#ffffff"},  # Deep Red
    "TECHNOLOGY": {"bg": "#047857", "border": "#34d399", "text": "#ffffff"},    # Emerald Green
    "CONCEPT": {"bg": "#6d28d9", "border": "#a78bfa", "text": "#ffffff"},       # Rich Purple
    "LOCATION": {"bg": "#c2410c", "border": "#fb923c", "text": "#ffffff"},      # Orange
    "EVENT": {"bg": "#b45309", "border": "#fbbf24", "text": "#ffffff"},         # Amber
    "PRODUCT": {"bg": "#0f766e", "border": "#2dd4bf", "text": "#ffffff"},       # Teal
    "DOCUMENT": {"bg": "#374151", "border": "#9ca3af", "text": "#ffffff"},      # Slate
}


def build_html_viewer(nodes: list[dict], edges: list[dict], title: str = "Neo4j Knowledge Graph Viewer") -> str:
    """Build a modern, readable, collision-free HTML graph visualizer."""
    formatted_nodes = []
    for n in nodes:
        cat = n.get("type", "CONCEPT").upper()
        theme = CATEGORY_THEMES.get(cat, {"bg": "#334155", "border": "#64748b", "text": "#ffffff"})
        
        # Pill/box style prevents text overlapping and ensures crisp readability
        formatted_nodes.append({
            "id": n["id"],
            "label": f" {n['label']} ",
            "shape": "box",
            "margin": {"top": 8, "bottom": 8, "left": 14, "right": 14},
            "borderRadius": 8,
            "borderWidth": 1.5,
            "color": {
                "background": theme["bg"],
                "border": theme["border"],
                "highlight": {"background": theme["bg"], "border": "#ffffff"},
                "hover": {"background": theme["border"], "border": "#ffffff"},
            },
            "font": {
                "color": theme["text"],
                "size": 13,
                "face": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                "bold": {"color": "#ffffff"},
            },
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.4)", "size": 6, "x": 1, "y": 2},
            "type": cat,
            "desc": n.get("desc", ""),
            "mentions": n.get("mentions", 1),
        })

    formatted_edges = []
    for idx, e in enumerate(edges):
        formatted_edges.append({
            "id": f"e_{idx}",
            "from": e["from"],
            "to": e["to"],
            "label": e.get("label", "").replace("_", " ").title(),
            "raw_label": e.get("label", ""),
            "desc": e.get("desc", ""),
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "color": {"color": "#475569", "highlight": "#38bdf8", "hover": "#60a5fa"},
            "width": 1.5,
            "smooth": {"type": "continuous", "roundness": 0.2},
            "font": {
                "color": "#cbd5e1",
                "size": 10,
                "face": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "background": "#0f172a",
                "strokeWidth": 0,
                "align": "middle",
            },
        })

    nodes_json = json.dumps(formatted_nodes)
    edges_json = json.dumps(formatted_edges)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #0b0f19;
      --panel: #131b2e;
      --border: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --primary: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}

    /* Left Sidebar */
    #sidebar {{
      width: 380px;
      background: var(--panel);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 20px;
      gap: 16px;
      z-index: 10;
      box-shadow: 4px 0 20px rgba(0, 0, 0, 0.4);
    }}
    .header-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    h1 {{
      font-size: 18px;
      font-weight: 700;
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .stats-badge {{
      font-size: 12px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--primary);
      padding: 4px 8px;
      border-radius: 6px;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }}

    /* Search & Filter */
    .search-wrap {{
      position: relative;
    }}
    .search-input {{
      width: 100%;
      padding: 10px 14px;
      padding-left: 36px;
      background: var(--bg);
      border: 1px solid #334155;
      border-radius: 8px;
      color: #fff;
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{
      border-color: var(--primary);
    }}
    .search-icon {{
      position: absolute;
      left: 12px;
      top: 11px;
      color: var(--muted);
      font-size: 14px;
    }}

    /* Category Filter Chips */
    .category-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      font-size: 11px;
      font-weight: 600;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid transparent;
      user-select: none;
      transition: all 0.15s;
    }}
    .chip.active {{
      filter: brightness(1.1);
      box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
    }}
    .chip.inactive {{
      opacity: 0.35;
      filter: grayscale(0.8);
    }}

    /* Toggles and Spacing Controls */
    .controls-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: var(--bg);
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .ctrl-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .slider-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    input[type=range] {{
      flex: 1;
      accent-color: var(--primary);
      cursor: pointer;
    }}
    .btn {{
      padding: 6px 10px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      color: var(--text);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background 0.15s;
    }}
    .btn:hover {{
      background: #334155;
      border-color: #475569;
    }}
    .btn.active {{
      background: var(--primary);
      color: #0f172a;
      font-weight: 700;
      border-color: var(--primary);
    }}

    /* Details Panel */
    #details-panel {{
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      overflow-y: auto;
      font-size: 13px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .detail-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fff;
    }}
    .detail-badge {{
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      color: #fff;
      width: fit-content;
    }}
    .detail-desc {{
      color: #cbd5e1;
      line-height: 1.5;
      background: rgba(255, 255, 255, 0.03);
      padding: 10px;
      border-radius: 6px;
      border-left: 3px solid var(--primary);
    }}
    .rel-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 250px;
      overflow-y: auto;
      padding-right: 4px;
    }}
    .rel-card {{
      background: #1e293b;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 12px;
      border: 1px solid #334155;
    }}
    .rel-card .arrow {{
      color: var(--primary);
      font-weight: 600;
      margin-right: 4px;
    }}

    /* Canvas */
    #network-canvas {{
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at center, #111827 0%, #0b0f19 100%);
      position: relative;
    }}
    .canvas-overlay {{
      position: absolute;
      top: 16px;
      right: 16px;
      display: flex;
      gap: 8px;
      z-index: 5;
    }}
  </style>
</head>
<body>

  <div id="sidebar">
    <div class="header-row">
      <h1>🕸️ Knowledge Graph</h1>
      <span class="stats-badge" id="stats-pill">{len(formatted_nodes)} entities • {len(formatted_edges)} links</span>
    </div>

    <!-- Search -->
    <div class="search-wrap">
      <span class="search-icon">🔍</span>
      <input type="text" id="search-input" class="search-input" placeholder="Search entity by name..." oninput="handleSearch()">
    </div>

    <!-- Category Filter Chips -->
    <div class="category-chips">
      <span class="chip active" style="background:#1d4ed8; color:#fff;" onclick="toggleCategory('PERSON', this)">Person</span>
      <span class="chip active" style="background:#b91c1c; color:#fff;" onclick="toggleCategory('ORGANIZATION', this)">Organization</span>
      <span class="chip active" style="background:#047857; color:#fff;" onclick="toggleCategory('TECHNOLOGY', this)">Technology</span>
      <span class="chip active" style="background:#6d28d9; color:#fff;" onclick="toggleCategory('CONCEPT', this)">Concept</span>
      <span class="chip active" style="background:#c2410c; color:#fff;" onclick="toggleCategory('LOCATION', this)">Location</span>
      <span class="chip active" style="background:#b45309; color:#fff;" onclick="toggleCategory('EVENT', this)">Event</span>
      <span class="chip active" style="background:#0f766e; color:#fff;" onclick="toggleCategory('PRODUCT', this)">Product</span>
    </div>

    <!-- Spacing & Visibility Controls -->
    <div class="controls-grid">
      <div class="ctrl-item">
        <span>Node Spacing</span>
        <div class="slider-row">
          <input type="range" id="spacing-slider" min="120" max="360" value="220" oninput="updateSpacing(this.value)">
        </div>
      </div>
      <div class="ctrl-item">
        <span>Edge Labels</span>
        <button class="btn active" id="labels-btn" onclick="toggleEdgeLabels()">Visible</button>
      </div>
    </div>

    <!-- Inspector Panel -->
    <div id="details-panel">
      <p style="color: #64748b; font-style: italic; margin: auto; text-align: center;">
        Click any entity on the canvas to inspect its context and all connected relationships.
      </p>
    </div>
  </div>

  <div id="network-canvas">
    <div class="canvas-overlay">
      <button class="btn" onclick="resetZoom()">⟲ Reset Zoom</button>
      <button class="btn" id="physics-btn" onclick="togglePhysics()">Freeze Layout</button>
    </div>
  </div>

  <script>
    const rawNodes = {nodes_json};
    const rawEdges = {edges_json};

    const nodesDataSet = new vis.DataSet(rawNodes);
    const edgesDataSet = new vis.DataSet(rawEdges);

    const container = document.getElementById("network-canvas");
    const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

    // Tuned Barnes-Hut solver prevents node overlapping and prevents chaotic spinning
    let currentSpringLength = 220;
    const options = {{
      physics: {{
        solver: "barnesHut",
        barnesHut: {{
          gravitationalConstant: -7000,
          centralGravity: 0.12,
          springLength: currentSpringLength,
          springConstant: 0.03,
          damping: 0.15,
          avoidOverlap: 1.0
        }},
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{
        hover: true,
        hoverConnectedEdges: true,
        selectConnectedEdges: true,
        tooltipDelay: 100,
        navigationButtons: true,
        keyboard: true
      }}
    }};

    const network = new vis.Network(container, data, options);

    let activeCategories = new Set(["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "DOCUMENT"]);
    let showEdgeLabels = true;
    let isPhysicsActive = true;

    // Node Selection & Inspection
    network.on("click", function(params) {{
      if (params.nodes.length > 0) {{
        const selectedId = params.nodes[0];
        highlightNeighborhood(selectedId);
        showNodeDetails(selectedId);
      }} else {{
        resetHighlight();
        document.getElementById("details-panel").innerHTML = `
          <p style="color: #64748b; font-style: italic; margin: auto; text-align: center;">
            Click any entity on the canvas to inspect its context and connected relationships.
          </p>
        `;
      }}
    }});

    function showNodeDetails(nodeId) {{
      const node = rawNodes.find(n => n.id === nodeId);
      if (!node) return;

      const connectedEdges = rawEdges.filter(e => e.from === nodeId || e.to === nodeId);
      const relCards = connectedEdges.map(e => {{
        const isSource = e.from === nodeId;
        const targetId = isSource ? e.to : e.from;
        const otherNode = rawNodes.find(n => n.id === targetId);
        const otherName = otherNode ? otherNode.label.trim() : targetId;
        const arrow = isSource ? "──[" + e.raw_label + "]──>" : "<──[" + e.raw_label + "]──";

        return `
          <div class="rel-card">
            <div><span class="arrow">${{arrow}}</span> <b>${{otherName}}</b></div>
            ${{e.desc ? '<div style="color:#94a3b8; font-size:11px; margin-top:2px;">' + e.desc + '</div>' : ''}}
          </div>
        `;
      }}).join("");

      document.getElementById("details-panel").innerHTML = `
        <div class="detail-title">${{node.label.trim()}}</div>
        <div class="detail-badge" style="background:${{node.color.background}}">${{node.type}}</div>
        ${{node.desc ? '<div class="detail-desc">' + node.desc + '</div>' : '<div style="color:#64748b; font-size:12px;">No description extracted.</div>'}}
        <div style="font-weight:600; font-size:12px; color:#94a3b8; margin-top:4px;">
          Connected Relationships (${{connectedEdges.length}}):
        </div>
        <div class="rel-list">
          ${{relCards || '<div style="color:#64748b; font-size:12px;">No connected edges in this view.</div>'}}
        </div>
      `;
    }}

    function highlightNeighborhood(selectedId) {{
      const connectedNodeIds = new Set();
      connectedNodeIds.add(selectedId);

      rawEdges.forEach(e => {{
        if (e.from === selectedId) connectedNodeIds.add(e.to);
        if (e.to === selectedId) connectedNodeIds.add(e.from);
      }});

      nodesDataSet.forEach(node => {{
        if (connectedNodeIds.has(node.id)) {{
          nodesDataSet.update({{ id: node.id, opacity: 1.0, hidden: false }});
        }} else {{
          nodesDataSet.update({{ id: node.id, opacity: 0.15 }});
        }}
      }});
    }}

    function resetHighlight() {{
      nodesDataSet.forEach(node => {{
        const isVisible = activeCategories.has(node.type);
        nodesDataSet.update({{ id: node.id, opacity: 1.0, hidden: !isVisible }});
      }});
    }}

    // Search Filter
    function handleSearch() {{
      const query = document.getElementById("search-input").value.trim().toLowerCase();
      if (!query) {{
        resetHighlight();
        return;
      }}

      let firstMatch = null;
      nodesDataSet.forEach(node => {{
        const match = node.label.toLowerCase().includes(query) || (node.desc && node.desc.toLowerCase().includes(query));
        if (match && !firstMatch) firstMatch = node.id;
        nodesDataSet.update({{ id: node.id, hidden: !match, opacity: 1.0 }});
      }});

      if (firstMatch) {{
        network.focus(firstMatch, {{ scale: 1.1, animation: true }});
        showNodeDetails(firstMatch);
      }}
    }}

    // Category Toggle
    function toggleCategory(cat, element) {{
      if (activeCategories.has(cat)) {{
        activeCategories.delete(cat);
        element.classList.remove("active");
        element.classList.add("inactive");
      }} else {{
        activeCategories.add(cat);
        element.classList.add("active");
        element.classList.remove("inactive");
      }}

      nodesDataSet.forEach(n => {{
        nodesDataSet.update({{ id: n.id, hidden: !activeCategories.has(n.type) }});
      }});
    }}

    // Spacing Slider
    function updateSpacing(val) {{
      currentSpringLength = parseInt(val);
      network.setOptions({{
        physics: {{
          barnesHut: {{ springLength: currentSpringLength }}
        }}
      }});
    }}

    // Edge Label Toggle
    function toggleEdgeLabels() {{
      showEdgeLabels = !showEdgeLabels;
      const btn = document.getElementById("labels-btn");
      btn.innerText = showEdgeLabels ? "Visible" : "Hidden";
      btn.classList.toggle("active", showEdgeLabels);

      edgesDataSet.forEach(edge => {{
        edgesDataSet.update({{
          id: edge.id,
          label: showEdgeLabels ? edge.raw_label.replace(/_/g, " ").toLowerCase() : ""
        }});
      }});
    }}

    // Physics Toggle
    function togglePhysics() {{
      isPhysicsActive = !isPhysicsActive;
      network.setOptions({{ physics: {{ enabled: isPhysicsActive }} }});
      const btn = document.getElementById("physics-btn");
      btn.innerText = isPhysicsActive ? "Freeze Layout" : "Unfreeze Layout";
    }}

    function resetZoom() {{
      network.fit({{ animation: true }});
    }}
  </script>
</body>
</html>
"""


def export_and_open_viewer(nodes: list[dict], edges: list[dict], output_path: Path) -> Path:
    """Build and write the HTML viewer file."""
    html = build_html_viewer(nodes, edges)
    output_path.write_text(html, encoding="utf-8")
    return output_path
