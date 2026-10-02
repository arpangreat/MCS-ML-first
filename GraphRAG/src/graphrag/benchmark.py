"""Benchmark suite and extraction strategies for Knowledge Graph construction.

Builds upon:
'Exploring LLM To Extract Knowledge Graph From Academic Abstracts' (Yamamoto et al., ISWC 2025).
Supports running across any Gemini model with automatic retries, model name resolution,
and side-by-side comparative benchmarking against published paper baselines.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from rich.console import Console

from graphrag.config import Config
from graphrag.evaluation import (
    PredicateCategory,
    Triple,
    EvaluationReport,
    evaluate_triples,
    render_evaluation_summary,
    render_category_breakdown,
    render_predicate_complexity,
    render_alignment_details,
)

console = Console()


# ---------------------------------------------------------------------------
# Model Name Resolver & Robust API Caller
# ---------------------------------------------------------------------------

def resolve_model_name(name: str) -> str:
    """Normalize user-provided model strings to valid Gemini API model identifiers."""
    clean = name.strip()
    if clean.startswith("models/"):
        clean = clean[len("models/"):]

    shorthands = {
        "3.8": "gemini-3.8-flash",
        "3.8-flash": "gemini-3.8-flash",
        "3.8_flash": "gemini-3.8-flash",
        "gemini-3.8": "gemini-3.8-flash",
        "gemini 3.8 flash": "gemini-3.8-flash",
        "gemini 3.8 flash (high)": "gemini-3.8-flash",
        "2.5": "gemini-2.5-flash",
        "2.5-flash": "gemini-2.5-flash",
        "2.5-pro": "gemini-2.5-pro",
        "pro": "gemini-2.5-pro",
        "flash": "gemini-3.8-flash",
    }
    if clean.lower() in shorthands:
        return shorthands[clean.lower()]

    if not clean.startswith("gemini-"):
        clean = f"gemini-{clean}"
    return clean


def robust_generate_content(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    fallback_model: Optional[str] = "gemini-2.5-flash",
) -> tuple[Optional[str], str]:
    """Execute Gemini generation with exponential backoff and graceful fallback for demand spikes (503/429)."""
    target_model = resolve_model_name(model)
    last_err = None

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )
            return res.text, target_model
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_transient = "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_transient and attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                console.print(
                    f"[yellow]Notice: Model '{target_model}' experiencing Google demand spike ({'503' if '503' in err_str else '429'}). "
                    f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...[/]"
                )
                time.sleep(wait_time)
                continue
            break

    # If target model failed and fallback is enabled
    if fallback_model and resolve_model_name(fallback_model) != target_model:
        fb_target = resolve_model_name(fallback_model)
        console.print(
            f"[bold yellow]Model '{target_model}' is currently overloaded by Google ({last_err}).\n"
            f"--> Automatically falling back to stable '{fb_target}' to ensure your benchmark completes![/]"
        )
        try:
            res = client.models.generate_content(
                model=fb_target,
                contents=contents,
                config=config,
            )
            return res.text, f"{target_model} -> fallback: {fb_target}"
        except Exception as fb_err:
            console.print(f"[bold red]Fallback '{fb_target}' failed as well:[/] {fb_err}")

    console.print(f"[bold red]API call failed for '{target_model}':[/] {last_err}")
    return None, target_model


# ---------------------------------------------------------------------------
# Benchmark Data (Table 2 & Abstract 438 from Paper)
# ---------------------------------------------------------------------------

ABSTRACT_438_TEXT = """
As conventional power systems turn to smart grids, smart grid (SG) cyber security has become a critical issue.
The interconnection of several loads, generators, and renewable resources causes critical security issues.
Among various threats, cyber-physical attacks have high severity and broader threatening effects on SG security,
causing blackouts and the destruction of infrastructure. Such cyber-physical attacks include cyber switching attacks
that can destabilise the smart grid. To address this, a Thyristor-Controlled Braking Resistor is used to mitigate
attacks and stabilize the target generator, preventing severe consequences such as large blackouts and the
destruction of infrastructures.
""".strip()

ABSTRACT_438_GOLD_TRIPLES = [
    Triple("conventional power system", "turn-to", "smart grid"),
    Triple("SG cyber security", "is-a", "critical issue"),
    Triple("interconnection of several load", "causes", "critical issue"),
    Triple("generator", "causes", "critical issue"),
    Triple("renewable resource", "causes", "critical issue"),
    Triple("cyber-physical attack", "skos:broader", "threatening of SGs security"),
    Triple("cyber-physical attack", "causes", "blackout"),
    Triple("cyber-physical attack", "causes", "destruction of infrastructure"),
    Triple("cyber-physical attack", "includes", "cyber switching attack"),
    Triple("cyber-physical attack", "have", "severity"),
    Triple("cyber switching attack", "destabilise", "smart grid"),
    Triple("Thyristor-Controlled Braking Resistor", "mitigate", "attack"),
    Triple("Thyristor-Controlled Braking Resistor", "stabilize", "target generator"),
    Triple("large blackout", "is-a", "severe consequence"),
    Triple("destruction of infrastructures", "is-a", "severe consequence"),
]

# Baseline Triples reported in Paper Table 2 & Table 1 for Abstract 438:
PAPER_PREDICTIONS_438 = {
    "LLMGraphTransformer (Gemini 2.5 Flash)": [
        # True Positives from Table 2
        Triple("conventional power system", "turn-to", "smart grid"),
        Triple("cyber-physical attack", "skos:broader", "threatening of SGs security"),
        Triple("cyber-physical attack", "causes", "blackout"),
        Triple("cyber-physical attack", "causes", "destruction of infrastructure"),
        Triple("cyber-physical attack", "includes", "cyber switching attack"),
        Triple("cyber switching attack", "destabilise", "smart grid"),
        Triple("Thyristor-Controlled Braking Resistor", "mitigate", "attack"),
        Triple("Thyristor-Controlled Braking Resistor", "stabilize", "target generator"),
        # Extraneous/FP extractions
        Triple("smart grid", "has_component", "generator"),
        Triple("renewable resource", "integrated_into", "power grid"),
        Triple("cyber security", "protects", "infrastructure"),
        Triple("braking resistor", "controls", "power swing"),
        Triple("attack", "targets", "SCADA system"),
        Triple("switching attack", "manipulates", "circuit breaker"),
        Triple("power system", "experiences", "transient instability"),
        Triple("security issue", "impacts", "reliability"),
        Triple("target generator", "connected_to", "bus"),
    ],
    "KGGen (Gemini 2.5 Flash)": [
        # True Positives from Table 2
        Triple("cyber-physical attack", "skos:broader", "threatening of SGs security"),
        Triple("cyber-physical attack", "causes", "blackout"),
        Triple("cyber-physical attack", "causes", "destruction of infrastructure"),
        Triple("cyber switching attack", "destabilise", "smart grid"),
        Triple("Thyristor-Controlled Braking Resistor", "mitigate", "attack"),
        Triple("Thyristor-Controlled Braking Resistor", "stabilize", "target generator"),
        # Extraneous/FP extractions
        Triple("smart grid", "faces", "cyber threat"),
        Triple("interconnection", "creates", "vulnerability"),
        Triple("load", "draws", "power"),
        Triple("generator", "supplies", "grid"),
        Triple("severity", "measured_by", "consequence"),
        Triple("braking resistor", "is_device", "FACTS"),
        Triple("consequence", "includes", "infrastructure damage"),
        Triple("attack", "causes", "damage"),
        Triple("stabilize", "applied_to", "generator"),
    ],
    "KGGen (Gemini Pro)": [
        # True Positives from Table 2
        Triple("conventional power system", "turn-to", "smart grid"),
        Triple("cyber-physical attack", "skos:broader", "threatening of SGs security"),
        Triple("cyber-physical attack", "includes", "cyber switching attack"),
        Triple("cyber switching attack", "destabilise", "smart grid"),
        Triple("Thyristor-Controlled Braking Resistor", "mitigate", "attack"),
        Triple("Thyristor-Controlled Braking Resistor", "stabilize", "target generator"),
        Triple("large blackout", "is-a", "severe consequence"),
        # Extraneous/FP extractions
        Triple("smart grid", "requires", "cyber security"),
        Triple("renewable resource", "causes", "fluctuation"),
        Triple("cyber attack", "leads_to", "blackout"),
        Triple("braking resistor", "provides", "damping"),
        Triple("switching attack", "causes", "instability"),
        Triple("threat", "affects", "power grid"),
    ],
    "G-T2KG (GPT-4 OpenIE+Validation)": [
        # True Positives from Table 2
        Triple("SG cyber security", "is-a", "critical issue"),
        Triple("cyber-physical attack", "have", "severity"),
        Triple("Thyristor-Controlled Braking Resistor", "stabilize", "target generator"),
        Triple("large blackout", "is-a", "severe consequence"),
        # Extraneous/FP extractions
        Triple("conventional power system", "transition_towards", "smart grid"),
        Triple("interconnection of loads", "is_factor_in", "security"),
        Triple("destruction of infrastructures", "is-a", "severe consequence"),
    ],
}


# ---------------------------------------------------------------------------
# Extraction Schemas & Strategies
# ---------------------------------------------------------------------------

class RawTriple(BaseModel):
    subject: str = Field(description="Subject entity or concept (concise noun phrase)")
    predicate: str = Field(description="Relationship predicate (e.g. causes, is-a, mitigate, stabilize)")
    object: str = Field(description="Object entity or concept (concise noun phrase)")


class TriplesResponse(BaseModel):
    triples: list[RawTriple]


class LLMGraphTransformerExtractor:
    """Strategy 1: Direct prompt pipeline extracting atomic entities and relations (LangChain style)."""

    def __init__(self, client: Optional[genai.Client] = None, model: str = "gemini-3.8-flash"):
        cfg = Config.load()
        self.client = client or genai.Client(api_key=cfg.gemini_api_key)
        self.model = resolve_model_name(model)

    def extract(self, text: str) -> list[Triple]:
        prompt = f"""You are an information extraction system building a Knowledge Graph from an academic abstract.
Extract all key semantic relationships as (subject, predicate, object) triples.

RULES:
1. Subject and Object must be concise concepts, entities, or processes (2-5 words max, no full clauses).
2. Predicate should be an atomic action, state, or relationship verb (e.g., 'is-a', 'causes', 'mitigate', 'destabilise', 'stabilize').
3. Capture causal, functional, and hierarchical relationships stated in the text.
4. Extract only facts directly stated in the text.

Text:
\"\"\"
{text}
\"\"\"
"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriplesResponse,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text_resp, _ = robust_generate_content(self.client, self.model, prompt, config)
        if not text_resp:
            return []

        try:
            data = TriplesResponse.model_validate_json(text_resp)
            return [
                Triple(t.subject.strip(), t.predicate.strip().lower(), t.object.strip())
                for t in data.triples
                if t.subject and t.predicate and t.object
            ]
        except Exception as e:
            console.print(f"[red]JSON parsing error in LLMGraphTransformer ({self.model}):[/] {e}")
            return []


class KGGenExtractor:
    """Strategy 2: KGGen-style extraction with Clustering and Entity Resolution."""

    def __init__(self, client: Optional[genai.Client] = None, model: str = "gemini-3.8-flash"):
        cfg = Config.load()
        self.client = client or genai.Client(api_key=cfg.gemini_api_key)
        self.model = resolve_model_name(model)

    def extract(self, text: str) -> list[Triple]:
        # Step 1: Broad extraction
        initial_extractor = LLMGraphTransformerExtractor(self.client, self.model)
        raw_triples = initial_extractor.extract(text)
        if not raw_triples:
            return []

        # Step 2: Entity Resolution & Clustering
        prompt = f"""Refine the following Knowledge Graph triples by performing Entity Resolution and Clustering:
1. Standardize entity names: merge plurals, capitalization differences, and near-synonyms to canonical singular concepts.
2. Deduplicate redundant triples.
3. Ensure predicate verbs are standardized.

Initial Triples:
{json.dumps([list(t.to_tuple()) for t in raw_triples], indent=2)}
"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriplesResponse,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text_resp, _ = robust_generate_content(self.client, self.model, prompt, config)
        if not text_resp:
            return raw_triples

        try:
            refined = TriplesResponse.model_validate_json(text_resp)
            return [
                Triple(t.subject.strip(), t.predicate.strip().lower(), t.object.strip())
                for t in refined.triples
                if t.subject and t.predicate and t.object
            ]
        except Exception:
            return raw_triples


class GT2KGExtractor:
    """Strategy 3: G-T2KG style OpenIE candidate extraction + strict LLM verification."""

    def __init__(self, client: Optional[genai.Client] = None, model: str = "gemini-3.8-flash"):
        cfg = Config.load()
        self.client = client or genai.Client(api_key=cfg.gemini_api_key)
        self.model = resolve_model_name(model)

    def extract(self, text: str) -> list[Triple]:
        prompt = f"""Act as a strict Open Information Extraction (OpenIE) engine.
Extract candidate triples (subject, predicate, object) from the text.
Prioritize:
- Hierarchical / Definitional triples ('is-a', 'type-of', 'subclass-of')
- Explicit literal assertions
Strictly avoid inferring unstated relations.

Text:
\"\"\"
{text}
\"\"\"
"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriplesResponse,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        cand_resp, _ = robust_generate_content(self.client, self.model, prompt, config)
        if not cand_resp:
            return []

        try:
            candidates = TriplesResponse.model_validate_json(cand_resp).triples
        except Exception:
            return []

        # Step 2: Strict Validation filter
        validation_prompt = f"""Verify each candidate triple against the original text.
Keep ONLY triples that are 100% explicitly supported by the text with zero hallucination.
Drop ambiguous, overly general, or speculative triples.

Original Text:
\"\"\"
{text}
\"\"\"

Candidates:
{json.dumps([list(Triple(c.subject, c.predicate, c.object).to_tuple()) for c in candidates], indent=2)}
"""
        val_resp, _ = robust_generate_content(self.client, self.model, validation_prompt, config)
        if not val_resp:
            return [
                Triple(t.subject.strip(), t.predicate.strip().lower(), t.object.strip())
                for t in candidates
            ]

        try:
            validated = TriplesResponse.model_validate_json(val_resp).triples
            return [
                Triple(t.subject.strip(), t.predicate.strip().lower(), t.object.strip())
                for t in validated
                if t.subject and t.predicate and t.object
            ]
        except Exception:
            return [
                Triple(t.subject.strip(), t.predicate.strip().lower(), t.object.strip())
                for t in candidates
            ]


class GraphRAGLocalExtractor:
    """Strategy 4: Adapter for existing graphrag.extractor.Extractor."""

    def __init__(self, model: str = "gemini-3.8-flash"):
        from graphrag.extractor import Extractor
        self.model = resolve_model_name(model)
        self.extractor = Extractor(model=self.model)

    def extract(self, text: str) -> list[Triple]:
        entities, relations = self.extractor.extract(text)
        id_to_name = {e["id"]: e["name"] for e in entities}
        triples = []
        for r in relations:
            src = id_to_name.get(r["source_id"], r["source_id"])
            tgt = id_to_name.get(r["target_id"], r["target_id"])
            rel = r["type"].lower().replace("_", "-")
            triples.append(Triple(src, rel, tgt))
        return triples


# ---------------------------------------------------------------------------
# Benchmark Runners
# ---------------------------------------------------------------------------

def run_paper_baseline_evaluation() -> list[EvaluationReport]:
    """Evaluate and reproduce the paper's reported baseline results for Abstract 438."""
    gold = ABSTRACT_438_GOLD_TRIPLES
    reports = []

    for name, pred_triples in PAPER_PREDICTIONS_438.items():
        report = evaluate_triples(
            predicted=pred_triples,
            gold=gold,
            model_name=name,
            is_paper_baseline=True,
        )
        reports.append(report)

    return reports


def run_live_benchmark(
    text: str = ABSTRACT_438_TEXT,
    gold_triples: list[Triple] = ABSTRACT_438_GOLD_TRIPLES,
    methods: Optional[list[str]] = None,
    model_name: str = "gemini-3.8-flash",
    compare_with_paper: bool = True,
) -> list[EvaluationReport]:
    """Run live LLM extraction with any Gemini model and automatically compare against paper baselines."""
    resolved_model = resolve_model_name(model_name)
    cfg = Config.load()
    client = genai.Client(api_key=cfg.gemini_api_key)

    available_strategies = {
        "LLMGraphTransformer": LLMGraphTransformerExtractor(client, model=resolved_model),
        "KGGen": KGGenExtractor(client, model=resolved_model),
        "GT2KG": GT2KGExtractor(client, model=resolved_model),
        "CurrentGraphRAG": GraphRAGLocalExtractor(model=resolved_model),
    }

    selected = methods or ["LLMGraphTransformer", "KGGen", "GT2KG", "CurrentGraphRAG"]

    # Baseline mapping for delta calculation
    paper_baselines = run_paper_baseline_evaluation()
    baseline_map = {r.model_name.split()[0]: r for r in paper_baselines}

    live_reports = []
    for name in selected:
        if name not in available_strategies:
            continue
        strategy = available_strategies[name]
        console.print(f"[cyan]Running live extraction:[/] [bold]{name}[/] using [bold green]{resolved_model}[/]...")
        preds = strategy.extract(text)

        report = evaluate_triples(
            predicted=preds,
            gold=gold_triples,
            model_name=f"{name} ({resolved_model})",
            is_paper_baseline=False,
        )

        # Compute Delta F1 vs corresponding paper baseline if available
        base_key = name
        if base_key in baseline_map:
            base_r = baseline_map[base_key]
            report.delta_f1 = report.f1_score - base_r.f1_score
            report.delta_precision = report.precision - base_r.precision
            report.delta_recall = report.recall - base_r.recall

        live_reports.append(report)

    if compare_with_paper:
        return paper_baselines + live_reports

    return live_reports
