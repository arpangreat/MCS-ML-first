"""Export knowledge graph directly to high-resolution PNG and SVG images using Graphviz."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from rich.console import Console

from graphrag.graph import KnowledgeGraph

console = Console()

TYPE_STYLES = {
    "PERSON": {"fill": "#1e3a8a", "border": "#60a5fa", "font": "#ffffff"},       # Blue
    "ORGANIZATION": {"fill": "#881337", "border": "#fb7185", "font": "#ffffff"}, # Rose / Red
    "TECHNOLOGY": {"fill": "#064e3b", "border": "#34d399", "font": "#ffffff"},   # Emerald Green
    "CONCEPT": {"fill": "#4c1d95", "border": "#a78bfa", "font": "#ffffff"},      # Purple
    "LOCATION": {"fill": "#7c2d12", "border": "#fb923c", "font": "#ffffff"},     # Orange
    "EVENT": {"fill": "#713f12", "border": "#fde047", "font": "#ffffff"},        # Gold
    "PRODUCT": {"fill": "#134e4a", "border": "#2dd4bf", "font": "#ffffff"},      # Teal
    "DOCUMENT": {"fill": "#1e293b", "border": "#94a3b8", "font": "#ffffff"},     # Slate
    "OTHER": {"fill": "#334155", "border": "#64748b", "font": "#ffffff"},
}


def sanitize_id(raw_id: str) -> str:
    """Format string into a valid Graphviz node identifier."""
    return f"node_{abs(hash(raw_id))}"


def escape_label(text: str) -> str:
    """Escape characters for Graphviz label string."""
    return text.replace('"', '\\"').replace("\n", " ").strip()


def export_graph_to_image(
    output_png: Path = Path("knowledge_graph.png"),
    output_svg: Path = Path("knowledge_graph.svg"),
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Fetch all entities and relationships from Neo4j and render high-resolution PNG and SVG."""
    console.print("[cyan]Fetching knowledge graph from Neo4j...[/]")

    with KnowledgeGraph() as g:
        with g.driver.session(database=g.database) as s:
            # Query all entities
            nodes_res = s.run("""
                MATCH (e:Entity)
                RETURN e.id AS id, e.name AS name, coalesce(e.type, 'CONCEPT') AS type, coalesce(e.mentions, 1) AS mentions
            """).data()

            # Query all relationships between entities
            rels_res = s.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s <> t
                RETURN s.id AS src, t.id AS tgt, type(r) AS rel, coalesce(r.weight, 1) AS weight
            """).data()

    if not nodes_res:
        raise ValueError("Knowledge graph is empty. Please ingest a PDF first.")

    console.print(f"[green]Retrieved {len(nodes_res)} entities and {len(rels_res)} relationships.[/]")
    console.print("[cyan]Generating Graphviz layout and high-res image...[/]")

    dot_lines = [
        "digraph KnowledgeGraph {",
        "  graph [",
        "    bgcolor=\"#0b0f19\"",
        "    pad=\"1.0\"",
        "    overlap=false",
        "    splines=true",
        "    sep=\"+25\"",
        "    fontsize=16",
        "    fontname=\"Helvetica-Bold\"",
        "    fontcolor=\"#38bdf8\"",
        "    label=\"\\nPDF Knowledge Graph (Generated from Neo4j)\\n\\n\"",
        "    labelloc=\"t\"",
        "  ];",
        "  node [",
        "    shape=box",
        "    style=\"filled,rounded\"",
        "    penwidth=2.0",
        "    margin=\"0.22,0.12\"",
        "    fontname=\"Helvetica-Bold\"",
        "    fontsize=11",
        "  ];",
        "  edge [",
        "    color=\"#475569\"",
        "    fontname=\"Helvetica\"",
        "    fontsize=8",
        "    fontcolor=\"#94a3b8\"",
        "    arrowsize=0.7",
        "    penwidth=1.2",
        "  ];",
    ]

    # Map entity id to dot id
    node_id_map = {}
    for n in nodes_res:
        nid = sanitize_id(n["id"])
        node_id_map[n["id"]] = nid
        name = escape_label(n["name"] or n["id"])
        cat = n["type"].upper()
        style = TYPE_STYLES.get(cat, TYPE_STYLES["OTHER"])

        dot_lines.append(
            f'  {nid} [label="{name}" fillcolor="{style["fill"]}" color="{style["border"]}" fontcolor="{style["font"]}"];'
        )

    for r in rels_res:
        src = node_id_map.get(r["src"])
        tgt = node_id_map.get(r["tgt"])
        if src and tgt:
            rel_label = escape_label(r["rel"].replace("_", " ").lower())
            dot_lines.append(f'  {src} -> {tgt} [label="{rel_label}"];')

    dot_lines.append("}")
    dot_content = "\n".join(dot_lines)

    dot_file = Path("knowledge_graph.dot")
    dot_file.write_text(dot_content, encoding="utf-8")

    # Layout engines to try: sfdp (best for multi-hundred node graphs), fdp, neato
    # Render PNG with 300 DPI high resolution
    cmd_png = ["sfdp", f"-Gdpi={dpi}", "-Tpng", str(dot_file), "-o", str(output_png)]
    try:
        subprocess.run(cmd_png, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to fdp
        cmd_png = ["fdp", f"-Gdpi={dpi}", "-Tpng", str(dot_file), "-o", str(output_png)]
        subprocess.run(cmd_png, check=True)

    # Render vector SVG (lossless infinite zoom)
    cmd_svg = ["sfdp", "-Tsvg", str(dot_file), "-o", str(output_svg)]
    try:
        subprocess.run(cmd_svg, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cmd_svg = ["fdp", "-Tsvg", str(dot_file), "-o", str(output_svg)]
        subprocess.run(cmd_svg, check=True)

    return output_png.resolve(), output_svg.resolve()


if __name__ == "__main__":
    png_path, svg_path = export_graph_to_image()
    console.print(f"[bold green]Successfully exported:[/]")
    console.print(f"• High-Res PNG: [cyan]{png_path}[/]")
    console.print(f"• Vector SVG:   [cyan]{svg_path}[/]")
