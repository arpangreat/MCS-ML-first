"""Command-line interface for PDF Knowledge Graph construction and Neo4j inspection."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from graphrag.config import Config
from graphrag.extractor import Extractor
from graphrag.graph import KnowledgeGraph
from graphrag.pdf import load_pdf

console = Console()


def handle_ingest(args: argparse.Namespace) -> None:
    path = Path(args.pdf_path)
    if not path.is_file():
        console.print(f"[bold red]File not found:[/] {path}")
        sys.exit(1)

    console.print(f"[bold cyan]Reading PDF:[/] {path.name}")
    pdf_data = load_pdf(path, chunk_size=args.chunk_size)
    console.print(f"[green]Parsed {pdf_data.page_count} pages into {len(pdf_data.chunks)} chunks.[/]")

    with KnowledgeGraph() as graph:
        if graph.document_exists(pdf_data.file_hash) and not args.force:
            console.print(
                f"[yellow]Document already exists in graph (SHA-256 match). "
                f"Skipping duplicate ingestion. (Use --force to override)[/]"
            )
            return

        extractor = Extractor()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[cyan]Building graph...", total=len(pdf_data.chunks))

            def on_progress(idx, total, msg):
                prog.update(task, completed=idx, description=f"[cyan]{msg}")

            res = graph.ingest_pdf(pdf_data, extractor, progress_cb=on_progress, force=args.force)

    console.print(Panel(
        f"[bold green]Ingestion Complete![/]\n"
        f"• Chunks: [bold]{res['chunks_processed']}[/]\n"
        f"• Entities Discovered: [bold]{res['entities_found']}[/]\n"
        f"• Relationships Created: [bold]{res['relations_found']}[/]\n"
        f"• Deduplication: [bold green]Active (Zero Redundancy)[/]\n\n"
        f"[cyan]View the graph in Neo4j Browser using:[/] [bold]uv run graphrag browser[/]",
        title="Knowledge Graph Status",
        expand=False,
    ))


def handle_stats(args: argparse.Namespace) -> None:
    with KnowledgeGraph() as graph:
        stats = graph.get_stats()

    t = Table(title="Knowledge Graph Overview")
    t.add_column("Item", style="cyan")
    t.add_column("Count", style="green", justify="right")
    t.add_row("Documents", str(stats["documents"]))
    t.add_row("Text Chunks", str(stats["chunks"]))
    t.add_row("Unique Entities", str(stats["entities"]))
    t.add_row("Relationships", str(stats["relationships"]))
    console.print(t)

    if stats["types"]:
        console.print()
        tt = Table(title="Entity Types")
        tt.add_column("Type", style="magenta")
        tt.add_column("Count", style="green", justify="right")
        for k, v in stats["types"].items():
            tt.add_row(k, str(v))
        console.print(tt)

    if stats["top_entities"]:
        console.print()
        te = Table(title="Top Connected Entities")
        te.add_column("Entity", style="yellow")
        te.add_column("Type", style="magenta")
        te.add_column("Connections", style="green", justify="right")
        for e in stats["top_entities"]:
            te.add_row(e["name"], e.get("type", ""), str(e["connections"]))
        console.print(te)


def handle_search(args: argparse.Namespace) -> None:
    with KnowledgeGraph() as graph:
        results = graph.search_entities(args.term, limit=args.limit)

    if not results:
        console.print(f"[yellow]No entities found matching '{args.term}'[/]")
        return

    t = Table(title=f"Search Results for '{args.term}'")
    t.add_column("Name", style="yellow")
    t.add_column("Type", style="magenta")
    t.add_column("Mentions", style="cyan", justify="right")
    t.add_column("Connections", style="green", justify="right")
    t.add_column("Description", style="white")

    for r in results:
        t.add_row(
            r["name"],
            r.get("type", ""),
            str(r.get("mentions", 1)),
            str(r.get("connections", 0)),
            (r.get("description") or "")[:70],
        )
    console.print(t)


def handle_cypher(args: argparse.Namespace) -> None:
    """Execute arbitrary Cypher query and display results."""
    with KnowledgeGraph() as graph:
        records = graph.execute_cypher(args.query)

    if not records:
        console.print("[yellow]Query returned no records.[/]")
        return

    keys = list(records[0].keys())
    t = Table(title=f"Cypher Result ({len(records)} rows)")
    for k in keys:
        t.add_column(k, style="cyan")

    for r in records[:50]:
        t.add_row(*[str(r.get(k, "")) for k in keys])
    console.print(t)


def handle_clear(args: argparse.Namespace) -> None:
    if not args.yes:
        confirm = console.input("[bold red]Delete all data from Neo4j knowledge graph? (y/N): [/]")
        if confirm.lower() != "y":
            console.print("[yellow]Aborted.[/]")
            return

    with KnowledgeGraph() as graph:
        deleted = graph.clear()
    console.print(f"[bold green]Deleted {deleted} nodes and relationships.[/]")


def handle_browser(args: argparse.Namespace) -> None:
    """Show connection info and open Neo4j Browser to visualize the graph."""
    cfg = Config.load()
    with KnowledgeGraph() as graph:
        browser_url = graph.get_browser_url()

    console.print(Panel(
        f"[bold green]Neo4j Instance Details[/]\n"
        f"• URI: [cyan]{cfg.neo4j_uri}[/]\n"
        f"• Database: [cyan]{cfg.neo4j_database}[/]\n"
        f"• Username: [cyan]{cfg.neo4j_user}[/]\n"
        f"• Aura Console: [underline blue]https://console.neo4j.io[/]\n"
        f"• Neo4j Browser: [underline blue]{browser_url}[/]\n\n"
        f"[bold yellow]Recommended Visualization Cypher Queries (run inside Neo4j Browser):[/]\n"
        f"1. [bold white]MATCH (s:Entity)-[r]->(t:Entity) RETURN s, r, t LIMIT 100[/]\n"
        f"2. [bold white]MATCH (d:Document)-[r1:HAS_CHUNK]->(c:Chunk)-[r2:MENTIONS]->(e:Entity) RETURN d, r1, c, r2, e LIMIT 50[/]\n"
        f"3. [bold white]MATCH (p:Person)-[r]-(o:Organization) RETURN p, r, o[/]\n"
        f"4. [bold white]MATCH path = (e:Entity {{name: 'OpenAI'}})-[*1..2]-(other) RETURN path[/]",
        title="🌐 Visualize Graph in Neo4j Browser",
        expand=False,
    ))

    if not args.no_open:
        console.print("[cyan]Opening Neo4j Browser in your default web browser...[/]")
        webbrowser.open(browser_url)


def handle_view(args: argparse.Namespace) -> None:
    """Generate an instant interactive visual graph directly from Neo4j and open in browser."""
    from graphrag.viewer import export_and_open_viewer
    with KnowledgeGraph() as graph:
        with console.status("[yellow]Fetching knowledge graph from Neo4j...[/]"):
            sub = graph.get_subgraph(max_nodes=args.max_nodes)

    if not sub["nodes"]:
        console.print("[yellow]Knowledge graph is currently empty. Ingest a PDF first![/]")
        return

    out_file = Path(args.output).resolve()
    export_and_open_viewer(sub["nodes"], sub["edges"], out_file)
    console.print(Panel(
        f"[bold green]Interactive Graph Generated![/]\n"
        f"• Visual Nodes: [bold]{len(sub['nodes'])}[/]\n"
        f"• Visual Relationships: [bold]{len(sub['edges'])}[/]\n"
        f"• File: [cyan]{out_file}[/]\n\n"
        f"Opening in your web browser now...",
        title="Instant Visualizer",
        expand=False,
    ))
    webbrowser.open(f"file://{out_file}")


def handle_export_image(args: argparse.Namespace) -> None:
    """Export knowledge graph to PNG and SVG images."""
    from graphrag.export_image import export_graph_to_image
    png_path, svg_path = export_graph_to_image(
        output_png=Path(args.output_png),
        output_svg=Path(args.output_svg),
        dpi=args.dpi,
    )
    console.print(Panel(
        f"[bold green]Images Successfully Exported![/]\n"
        f"• PNG: [cyan]{png_path}[/]\n"
        f"• SVG (Lossless Vector): [cyan]{svg_path}[/]",
        title="Image Export",
        expand=False,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphrag", description="PDF Knowledge Graph Builder with Neo4j")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Ingest
    p_ing = sub.add_parser("ingest", help="Ingest a PDF file into Neo4j")
    p_ing.add_argument("pdf_path", help="Path to PDF")
    p_ing.add_argument("--force", action="store_true", help="Force re-ingest even if duplicate")
    p_ing.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    p_ing.set_defaults(func=handle_ingest)

    # Stats
    p_stat = sub.add_parser("stats", help="View graph statistics")
    p_stat.set_defaults(func=handle_stats)

    # Search
    p_sch = sub.add_parser("search", help="Search entities in graph")
    p_sch.add_argument("term", help="Search keyword")
    p_sch.add_argument("--limit", type=int, default=15)
    p_sch.set_defaults(func=handle_search)

    # Cypher query
    p_cyp = sub.add_parser("cypher", help="Execute arbitrary Cypher query in Neo4j")
    p_cyp.add_argument("query", help="Cypher query string")
    p_cyp.set_defaults(func=handle_cypher)

    # Browser
    p_brw = sub.add_parser("browser", help="Open Neo4j Browser to visualize the graph")
    p_brw.add_argument("--no-open", action="store_true", help="Do not launch browser window automatically")
    p_brw.set_defaults(func=handle_browser)

    # View (Instant local viewer from Neo4j)
    p_viw = sub.add_parser("view", help="Instant 1-click interactive visual graph (no manual queries needed)")
    p_viw.add_argument("--max-nodes", type=int, default=150, help="Max entities to display")
    p_viw.add_argument("--output", default="knowledge_graph.html", help="HTML output file path")
    p_viw.set_defaults(func=handle_view)

    # Export Image (PNG and SVG)
    p_img = sub.add_parser("export-image", help="Export entire graph to high-res PNG and SVG")
    p_img.add_argument("--output-png", default="knowledge_graph.png", help="PNG output path")
    p_img.add_argument("--output-svg", default="knowledge_graph.svg", help="SVG output path")
    p_img.add_argument("--dpi", type=int, default=150, help="DPI resolution for PNG (e.g. 150 or 300)")
    p_img.set_defaults(func=handle_export_image)

    # Clear
    p_clr = sub.add_parser("clear", help="Clear all data from Neo4j")
    p_clr.add_argument("-y", "--yes", action="store_true")
    p_clr.set_defaults(func=handle_clear)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
