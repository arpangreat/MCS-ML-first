"""Evaluation metrics, predicate analysis, and semantic matching for Knowledge Graph extraction.

Inspired by Yamamoto et al. (ISWC 2025):
'Exploring LLM To Extract Knowledge Graph From Academic Abstracts'
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class PredicateCategory(str, Enum):
    HIERARCHICAL = "Hierarchical / Definitional"
    CAUSAL_FUNCTIONAL = "Causal & Functional"
    RELATIONAL_STRUCTURAL = "Relational & Structural"
    OTHER = "Other"


# Known taxonomy & ontology mappings
HIERARCHICAL_KEYWORDS = {
    "is_a", "is-a", "isa", "type_of", "type-of", "subclass_of", "subclass-of",
    "skos:broader", "broader", "instance_of", "instance-of", "classified_as",
    "category_of", "kind_of", "form_of", "defined_as"
}

CAUSAL_KEYWORDS = {
    "causes", "cause", "caused_by", "mitigate", "mitigates", "mitigated_by",
    "stabilize", "stabilizes", "stabilised", "stabilized", "destabilise",
    "destabilises", "destabilize", "destabilizes", "prevents", "prevent",
    "leads_to", "lead_to", "results_in", "result_in", "triggers", "trigger",
    "affects", "affect", "impacts", "impact", "reduces", "reduces_risk"
}

RELATIONAL_KEYWORDS = {
    "includes", "include", "contains", "contain", "composed_of", "part_of",
    "have", "has", "possess", "possesses", "turn_to", "turn-to", "transforms_to",
    "connects_to", "connected_to", "associated_with", "related_to", "uses",
    "utilizes", "operates_on"
}

UK_US_VARIANTS = {
    "destabilise": "destabilize",
    "destabilises": "destabilize",
    "destabilizes": "destabilize",
    "stabilise": "stabilize",
    "stabilises": "stabilize",
    "stabilizes": "stabilize",
}


ABBREVIATIONS = {
    "sg": "smart grid",
    "tcbr": "thyristor controlled braking resistor",
    "cps": "cyber physical system",
}


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, expanding abbreviations, and normalizing whitespace."""
    t = text.lower().strip()
    t = t.replace("-", " ").replace("_", " ")
    t = re.sub(r"[^\w\s]", "", t)
    tokens = t.split()
    normalized_tokens = []
    for tok in tokens:
        # Expand domain abbreviation
        tok = ABBREVIATIONS.get(tok, tok)
        # Normalize UK/US variant
        tok = UK_US_VARIANTS.get(tok, tok)
        # Simple plural stripping for common nouns/verbs
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            tok = tok[:-1]
        normalized_tokens.append(tok)
    return " ".join(normalized_tokens)


def categorize_predicate(predicate: str) -> PredicateCategory:
    """Classify a predicate into Hierarchical, Causal/Functional, Relational, or Other."""
    clean = predicate.strip().lower().replace(" ", "_").replace("-", "_")
    norm = normalize_text(predicate).replace(" ", "_")

    if clean in {"is", "is_a", "isa", "are"} or norm in {"is", "is_a", "isa", "are"} or any(k in clean or k in norm for k in ["is_a", "subclass", "broader", "type_of", "skos", "category"]):
        return PredicateCategory.HIERARCHICAL

    if any(k in clean or k in norm for k in ["cause", "mitigat", "stabiliz", "destabiliz", "lead_to", "result", "trigger", "prevent", "impact"]):
        return PredicateCategory.CAUSAL_FUNCTIONAL

    if any(k in clean or k in norm for k in ["include", "have", "turn_to", "transform", "part_of", "use", "contain"]):
        return PredicateCategory.RELATIONAL_STRUCTURAL

    return PredicateCategory.OTHER


@dataclass(frozen=True)
class Triple:
    """Knowledge Graph Subject-Predicate-Object triple."""
    subject: str
    predicate: str
    object: str

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject.strip(), self.predicate.strip(), self.object.strip())

    def normalized(self) -> tuple[str, str, str]:
        return (
            normalize_text(self.subject),
            normalize_text(self.predicate),
            normalize_text(self.object),
        )

    @property
    def predicate_category(self) -> PredicateCategory:
        return categorize_predicate(self.predicate)

    @property
    def predicate_token_count(self) -> int:
        return len(re.split(r"[\s_\-]+", self.predicate.strip()))

    @property
    def is_compound_predicate(self) -> bool:
        """Single-verb vs compound / multi-word predicate."""
        return self.predicate_token_count > 1

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


@dataclass
class TripleMatch:
    """Record of an alignment between a predicted and a gold triple."""
    predicted: Triple
    gold: Optional[Triple]
    score: float
    match_type: str  # 'exact', 'normalized', 'semantic', or 'unmatched'


@dataclass
class PredicateMetrics:
    """Precision, recall, and F1 metrics for a specific predicate or category."""
    name: str
    category: PredicateCategory
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_count: int = 0
    pred_count: int = 0

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return (self.tp / total) if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return (self.tp / total) if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


@dataclass
class EvaluationReport:
    """Comprehensive evaluation results for a KG extraction run."""
    model_name: str
    tp: int
    fp: int
    fn: int
    gold_total: int
    pred_total: int
    precision: float
    recall: float
    f1_score: float

    # Baseline comparison flags
    is_paper_baseline: bool = False
    delta_f1: Optional[float] = None
    delta_precision: Optional[float] = None
    delta_recall: Optional[float] = None

    # Predicate breakdown
    predicate_metrics: dict[str, PredicateMetrics] = field(default_factory=dict)
    category_metrics: dict[PredicateCategory, PredicateMetrics] = field(default_factory=dict)

    # Predicate complexity analysis
    single_verb_predicates: int = 0
    compound_predicates: int = 0
    avg_predicate_tokens: float = 0.0

    # Alignment details
    matches: list[TripleMatch] = field(default_factory=list)
    unmatched_predicted: list[Triple] = field(default_factory=list)
    unmatched_gold: list[Triple] = field(default_factory=list)


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculate Precision, Recall, and F1."""
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class Matcher:
    """Base interface for matching predicted triples with gold standard triples."""
    def match(self, pred: Triple, gold: Triple) -> tuple[bool, float]:
        raise NotImplementedError


class ExactMatcher(Matcher):
    """Matches triples if subject, predicate, and object are case-insensitively identical."""
    def match(self, pred: Triple, gold: Triple) -> tuple[bool, float]:
        p_sub, p_pred, p_obj = pred.to_tuple()
        g_sub, g_pred, g_obj = gold.to_tuple()
        if (
            p_sub.lower() == g_sub.lower()
            and p_pred.lower() == g_pred.lower()
            and p_obj.lower() == g_obj.lower()
        ):
            return True, 1.0
        return False, 0.0


class NormalizedMatcher(Matcher):
    """Matches triples using morphological normalization, singularization, and synonym handling."""
    def __init__(self, token_threshold: float = 0.75):
        self.token_threshold = token_threshold

    def _elem_match(self, elem1: str, elem2: str) -> float:
        n1 = normalize_text(elem1)
        n2 = normalize_text(elem2)
        if n1 == n2:
            return 1.0

        if n1 in {"is", "is a", "isa"} and n2 in {"is", "is a", "isa"}:
            return 1.0

        # Substring or token overlap
        t1, t2 = set(n1.split()), set(n2.split())
        if t1 and t2:
            jaccard = len(t1 & t2) / len(t1 | t2)
            if jaccard >= 0.5:
                return jaccard
            # Check subset / noun modifier containment (e.g. "high severity" contains "severity")
            if t1.issubset(t2) or t2.issubset(t1):
                return 0.85

        # Difflib sequence similarity
        seq_sim = difflib.SequenceMatcher(None, n1, n2).ratio()
        if seq_sim >= 0.80:
            return seq_sim

        return 0.0

    def match(self, pred: Triple, gold: Triple) -> tuple[bool, float]:
        p_sub, p_pred, p_obj = pred.normalized()
        g_sub, g_pred, g_obj = gold.normalized()

        if (p_sub, p_pred, p_obj) == (g_sub, g_pred, g_obj):
            return True, 1.0

        s_score = self._elem_match(p_sub, g_sub)
        p_score = self._elem_match(p_pred, g_pred)
        o_score = self._elem_match(p_obj, g_obj)

        # Allow category equivalence for predicates (e.g. is-a vs type-of)
        if p_score < 0.5 and pred.predicate_category == gold.predicate_category and pred.predicate_category != PredicateCategory.OTHER:
            p_score = 0.85

        if s_score >= 0.7 and p_score >= 0.7 and o_score >= 0.7:
            avg_score = (s_score + p_score + o_score) / 3.0
            return True, avg_score

        return False, 0.0


class LLMSemanticMatcher(Matcher):
    """LLM-as-a-judge semantic matcher, as described in GraphJudge / paper methodology."""
    def __init__(self, client: Any, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name
        self._fallback = NormalizedMatcher()

    def batch_align(self, predicted: list[Triple], gold: list[Triple]) -> dict[int, tuple[int, float]]:
        """Batch evaluates semantic alignment using Gemini."""
        if not self.client or not predicted or not gold:
            return {}

        prompt = f"""You are an expert evaluator in Knowledge Graph triple extraction.
Determine if each predicted triple semantically matches any gold triple.
Follow the paper rule: A predicted triple is correct if it semantically matches a gold-standard triple,
regardless of differences in wording (synonyms), structure (active/passive), or morphology (plural/singular).

Gold Triples:
{json.dumps([{"id": i, "triple": list(g.to_tuple())} for i, g in enumerate(gold)], indent=2)}

Predicted Triples:
{json.dumps([{"id": i, "triple": list(p.to_tuple())} for i, p in enumerate(predicted)], indent=2)}

Return a JSON array of matches for the predicted triples that have a valid semantic match in the gold list:
[
  {{"predicted_id": 0, "gold_id": 2, "confidence": 0.95, "reason": "semantic match"}}
]
If a predicted triple has no match, do NOT include it in the array.
A gold triple cannot be assigned to more than one predicted triple.
"""
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            data = json.loads(res.text)
            matches = {}
            used_gold = set()
            for item in data:
                pid = item.get("predicted_id")
                gid = item.get("gold_id")
                conf = float(item.get("confidence", 1.0))
                if pid is not None and gid is not None and gid not in used_gold:
                    matches[pid] = (gid, conf)
                    used_gold.add(gid)
            return matches
        except Exception:
            return {}


def evaluate_triples(
    predicted: list[Triple],
    gold: list[Triple],
    model_name: str = "Model",
    matcher: Optional[Matcher] = None,
    semantic_alignments: Optional[dict[int, int]] = None,
    is_paper_baseline: bool = False,
) -> EvaluationReport:
    """Evaluate predicted triples against gold standard triples."""
    matcher = matcher or NormalizedMatcher()

    matched_pairs: list[tuple[int, int, float, str]] = []
    matched_pred_indices = set()
    matched_gold_indices = set()

    if semantic_alignments:
        for p_idx, g_idx in semantic_alignments.items():
            if p_idx < len(predicted) and g_idx < len(gold):
                matched_pairs.append((p_idx, g_idx, 1.0, "semantic"))
                matched_pred_indices.add(p_idx)
                matched_gold_indices.add(g_idx)

    for p_idx, p in enumerate(predicted):
        if p_idx in matched_pred_indices:
            continue
        best_g_idx = None
        best_score = 0.0
        best_match_type = "unmatched"

        for g_idx, g in enumerate(gold):
            if g_idx in matched_gold_indices:
                continue

            exact_ok, score = ExactMatcher().match(p, g)
            if exact_ok:
                best_g_idx = g_idx
                best_score = score
                best_match_type = "exact"
                break

            norm_ok, score = matcher.match(p, g)
            if norm_ok and score > best_score:
                best_g_idx = g_idx
                best_score = score
                best_match_type = "normalized"

        if best_g_idx is not None and best_score >= 0.70:
            matched_pairs.append((p_idx, best_g_idx, best_score, best_match_type))
            matched_pred_indices.add(p_idx)
            matched_gold_indices.add(best_g_idx)

    tp = len(matched_pairs)
    fp = len(predicted) - tp
    fn = len(gold) - tp
    precision, recall, f1 = calculate_metrics(tp, fp, fn)

    single_verb_count = sum(1 for p in predicted if not p.is_compound_predicate)
    compound_count = sum(1 for p in predicted if p.is_compound_predicate)
    avg_tokens = (
        sum(p.predicate_token_count for p in predicted) / len(predicted)
        if predicted else 0.0
    )

    matches: list[TripleMatch] = []
    for p_idx, g_idx, score, m_type in matched_pairs:
        matches.append(TripleMatch(
            predicted=predicted[p_idx],
            gold=gold[g_idx],
            score=score,
            match_type=m_type,
        ))

    unmatched_pred = [p for i, p in enumerate(predicted) if i not in matched_pred_indices]
    unmatched_gold = [g for i, g in enumerate(gold) if i not in matched_gold_indices]

    all_predicate_names = {g.predicate for g in gold} | {p.predicate for p in predicted}
    predicate_metrics: dict[str, PredicateMetrics] = {}

    for pred_name in sorted(all_predicate_names):
        cat = categorize_predicate(pred_name)
        g_for_p = [g for g in gold if g.predicate == pred_name]
        p_for_p = [p for p in predicted if p.predicate == pred_name]

        pred_tp = sum(
            1 for m in matches
            if m.gold and (m.gold.predicate == pred_name or m.predicted.predicate == pred_name)
        )
        pred_fp = max(0, len(p_for_p) - pred_tp)
        pred_fn = max(0, len(g_for_p) - pred_tp)

        predicate_metrics[pred_name] = PredicateMetrics(
            name=pred_name,
            category=cat,
            tp=pred_tp,
            fp=pred_fp,
            fn=pred_fn,
            gold_count=len(g_for_p),
            pred_count=len(p_for_p),
        )

    category_metrics: dict[PredicateCategory, PredicateMetrics] = {}
    for cat in PredicateCategory:
        cat_gold = [g for g in gold if g.predicate_category == cat]
        cat_pred = [p for p in predicted if p.predicate_category == cat]

        cat_tp = sum(
            1 for m in matches
            if (m.gold and m.gold.predicate_category == cat) or (m.predicted.predicate_category == cat)
        )
        cat_fp = max(0, len(cat_pred) - cat_tp)
        cat_fn = max(0, len(cat_gold) - cat_tp)

        category_metrics[cat] = PredicateMetrics(
            name=cat.value,
            category=cat,
            tp=cat_tp,
            fp=cat_fp,
            fn=cat_fn,
            gold_count=len(cat_gold),
            pred_count=len(cat_pred),
        )

    return EvaluationReport(
        model_name=model_name,
        tp=tp,
        fp=fp,
        fn=fn,
        gold_total=len(gold),
        pred_total=len(predicted),
        precision=precision,
        recall=recall,
        f1_score=f1,
        is_paper_baseline=is_paper_baseline,
        predicate_metrics=predicate_metrics,
        category_metrics=category_metrics,
        single_verb_predicates=single_verb_count,
        compound_predicates=compound_count,
        avg_predicate_tokens=avg_tokens,
        matches=matches,
        unmatched_predicted=unmatched_pred,
        unmatched_gold=unmatched_gold,
    )


# ---------------------------------------------------------------------------
# Rich Visualizers for Terminal Output
# ---------------------------------------------------------------------------

def render_evaluation_summary(reports: list[EvaluationReport]) -> Table:
    """Render a comprehensive comparison table showing paper baselines and live runs."""
    has_deltas = any(r.delta_f1 is not None for r in reports)
    has_baselines = any(r.is_paper_baseline for r in reports)
    has_live = any(not r.is_paper_baseline for r in reports)

    table = Table(
        title="Knowledge Graph Extraction: Comparative Benchmark (ISWC 2025 Paper vs. Live Model)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Extraction Strategy / Model", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim", width=12)
    table.add_column("Pred", justify="right")
    table.add_column("TP", justify="right", style="green")
    table.add_column("FP", justify="right", style="red")
    table.add_column("FN", justify="right", style="yellow")
    table.add_column("Precision", justify="right", style="bold")
    table.add_column("Recall", justify="right", style="bold")
    table.add_column("F1-Measure", justify="right", style="bold green")

    if has_deltas:
        table.add_column("Δ F1 (vs Baseline)", justify="right", style="bold")

    # Render paper baselines first if mixed
    for r in reports:
        row_type = "[yellow]Paper[/]" if r.is_paper_baseline else "[bold green]Live[/]"
        delta_str = "-"
        if r.delta_f1 is not None:
            if r.delta_f1 > 0:
                delta_str = f"[bold green]+{r.delta_f1:.2f}[/]"
            elif r.delta_f1 < 0:
                delta_str = f"[bold red]{r.delta_f1:.2f}[/]"
            else:
                delta_str = "[dim]0.00[/]"

        cols = [
            r.model_name,
            row_type,
            str(r.pred_total),
            str(r.tp),
            str(r.fp),
            str(r.fn),
            f"{r.precision:.2f}",
            f"{r.recall:.2f}",
            f"{r.f1_score:.2f}",
        ]
        if has_deltas:
            cols.append(delta_str)

        table.add_row(*cols)

    return table


def render_category_breakdown(report: EvaluationReport) -> Table:
    """Render performance by Predicate Category (Hierarchical vs Causal vs Relational)."""
    table = Table(
        title=f"Predicate Category Breakdown: [bold cyan]{report.model_name}[/]",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Predicate Category", style="white")
    table.add_column("Gold Triples", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("TP", justify="right", style="green")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right", style="bold green")

    for cat, m in report.category_metrics.items():
        if m.gold_count == 0 and m.pred_count == 0:
            continue
        table.add_row(
            cat.value,
            str(m.gold_count),
            str(m.pred_count),
            str(m.tp),
            f"{m.precision:.2f}",
            f"{m.recall:.2f}",
            f"{m.f1_score:.2f}",
        )
    return table


def render_predicate_complexity(reports: list[EvaluationReport]) -> Table:
    """Render predicate complexity analysis (atomic verbs vs compound phrases)."""
    table = Table(
        title="Predicate Structure & Complexity Analysis",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Model / Strategy", style="cyan")
    table.add_column("Total Preds", justify="right")
    table.add_column("Single-Verb (Atomic)", justify="right", style="green")
    table.add_column("Compound / Phrase", justify="right", style="magenta")
    table.add_column("Avg Token Count", justify="right", style="bold")
    table.add_column("Primary Focus", style="white")

    for r in reports:
        focus = []
        if r.category_metrics.get(PredicateCategory.HIERARCHICAL, PredicateMetrics("", PredicateCategory.OTHER)).tp > 0:
            focus.append("Definitional/Hierarchy")
        if r.category_metrics.get(PredicateCategory.CAUSAL_FUNCTIONAL, PredicateMetrics("", PredicateCategory.OTHER)).tp > 0:
            focus.append("Causal/Functional")
        table.add_row(
            r.model_name,
            str(r.pred_total),
            str(r.single_verb_predicates),
            str(r.compound_predicates),
            f"{r.avg_predicate_tokens:.1f}",
            ", ".join(focus) if focus else "None",
        )
    return table


def render_alignment_details(report: EvaluationReport, max_items: int = 15) -> Table:
    """Render detailed triple alignment (TP matches, FPs, and missed gold FNs)."""
    table = Table(
        title=f"Detailed Triple Alignments for [bold cyan]{report.model_name}[/]",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Status", style="bold", width=8)
    table.add_column("Predicted Triple", style="cyan")
    table.add_column("Gold Triple", style="yellow")
    table.add_column("Match Mode / Category", style="white")

    # True Positives
    for m in report.matches[:max_items]:
        gold_str = str(m.gold) if m.gold else "N/A"
        cat = m.predicted.predicate_category.value
        table.add_row("[green]MATCH[/]", str(m.predicted), gold_str, f"{m.match_type} ({cat})")

    # False Negatives (Missed)
    for g in report.unmatched_gold[:max_items]:
        table.add_row("[red]MISSED[/]", "-", str(g), f"FN ({g.predicate_category.value})")

    # False Positives (Extraneous / Hallucinated)
    for p in report.unmatched_predicted[:max_items]:
        table.add_row("[yellow]EXTRA[/]", str(p), "-", f"FP ({p.predicate_category.value})")

    return table
