#!/usr/bin/env python3
"""Knowledge Graph Extraction Benchmark & Evaluation Runner.

Builds upon Yamamoto et al. (ISWC 2025):
'Exploring LLM To Extract Knowledge Graph From Academic Abstracts'

Usage:
    # 1. Run live extraction with any Gemini model (e.g. 3.8-flash, 2.5-flash) with side-by-side paper comparisons:
    uv run python benchmark_runner.py --model 3.8-flash

    # 2. Reproduce the published paper baselines offline:
    uv run python benchmark_runner.py --mode paper

    # 3. Run live extraction with itemized triple alignments (matched vs missed):
    uv run python benchmark_runner.py --model 3.8-flash --details
"""

from __future__ import annotations

import argparse
from rich.console import Console

from graphrag.benchmark import (
    run_paper_baseline_evaluation,
    run_live_benchmark,
    resolve_model_name,
)
from graphrag.evaluation import (
    render_evaluation_summary,
    render_category_breakdown,
    render_predicate_complexity,
    render_alignment_details,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Knowledge Graph extraction with Precision, Recall, F1, and Predicate Analysis (with ISWC 2025 paper comparisons)"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "paper"],
        default="live",
        help="live: extract with Gemini and compare side-by-side with paper baselines; paper: view paper baselines offline",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Gemini model to benchmark (supports '3.8-flash', '2.5-flash', 'gemini-3.8-flash', etc.)",
    )
    parser.add_argument(
        "--methods",
        default="LLMGraphTransformer,KGGen,GT2KG,CurrentGraphRAG",
        help="Comma-separated extractor methods to benchmark",
    )
    parser.add_argument(
        "--no-paper-compare",
        action="store_true",
        help="Do not include the paper's original published baselines in the summary table",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Display itemized matched, missed, and extra triples",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=12,
        help="Maximum number of alignment entries to display per method",
    )
    args = parser.parse_args()

    resolved_model = resolve_model_name(args.model)

    if args.mode == "paper":
        console.print("\n[bold cyan]===========================================================[/]")
        console.print("[bold cyan]   ISWC 2025 Paper Benchmark: Baseline Reproduction   [/]")
        console.print("[bold cyan]   (Yamamoto et al. - Abstract 438)                    [/]")
        console.print("[bold cyan]===========================================================[/]\n")
        reports = run_paper_baseline_evaluation()
    else:
        console.print("\n[bold green]===========================================================[/]")
        console.print(f"[bold green]   Knowledge Graph Benchmark: Live Run vs. Paper Baselines   [/]")
        console.print(f"[bold green]   Target Model: [bold white]{resolved_model}[/]   [/]")
        console.print("[bold green]===========================================================[/]\n")
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        reports = run_live_benchmark(
            methods=methods,
            model_name=resolved_model,
            compare_with_paper=not args.no_paper_compare,
        )

    if not reports:
        console.print("[yellow]No reports to display.[/]")
        return

    # 1. Main Performance Summary Table (Precision, Recall, F1, Delta)
    console.print(render_evaluation_summary(reports))

    # 2. Predicate Complexity & Structure Analysis
    console.print(render_predicate_complexity(reports))

    # 3. Predicate Category Breakdown (Hierarchical vs Causal vs Relational)
    live_reports = [r for r in reports if not r.is_paper_baseline]
    display_category_reports = live_reports if live_reports else reports
    for r in display_category_reports:
        console.print(render_category_breakdown(r))

    # 4. Detailed Triple Alignments (if requested)
    if args.details:
        for r in display_category_reports:
            console.print(render_alignment_details(r, max_items=args.max_details))


if __name__ == "__main__":
    main()
