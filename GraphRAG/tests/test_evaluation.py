"""Unit tests for Knowledge Graph evaluation, predicate metrics, and paper benchmark reproduction."""

import pytest
from graphrag.evaluation import (
    PredicateCategory,
    Triple,
    categorize_predicate,
    calculate_metrics,
    ExactMatcher,
    NormalizedMatcher,
    evaluate_triples,
)
from graphrag.benchmark import (
    ABSTRACT_438_GOLD_TRIPLES,
    PAPER_PREDICTIONS_438,
    run_paper_baseline_evaluation,
)


def test_triple_properties():
    t1 = Triple("smart grid", "destabilise", "power system")
    assert t1.to_tuple() == ("smart grid", "destabilise", "power system")
    assert t1.predicate_token_count == 1
    assert not t1.is_compound_predicate
    assert t1.predicate_category == PredicateCategory.CAUSAL_FUNCTIONAL

    t2 = Triple("conventional power system", "turn-to", "smart grid")
    assert t2.predicate_token_count == 2
    assert t2.is_compound_predicate
    assert t2.predicate_category == PredicateCategory.RELATIONAL_STRUCTURAL

    t3 = Triple("SG cyber security", "is-a", "critical issue")
    assert t3.predicate_category == PredicateCategory.HIERARCHICAL


def test_predicate_categorization():
    # Hierarchical
    assert categorize_predicate("is-a") == PredicateCategory.HIERARCHICAL
    assert categorize_predicate("is_a") == PredicateCategory.HIERARCHICAL
    assert categorize_predicate("is") == PredicateCategory.HIERARCHICAL
    assert categorize_predicate("skos:broader") == PredicateCategory.HIERARCHICAL
    assert categorize_predicate("subclass-of") == PredicateCategory.HIERARCHICAL

    # Causal & Functional
    assert categorize_predicate("causes") == PredicateCategory.CAUSAL_FUNCTIONAL
    assert categorize_predicate("mitigate") == PredicateCategory.CAUSAL_FUNCTIONAL
    assert categorize_predicate("stabilize") == PredicateCategory.CAUSAL_FUNCTIONAL
    assert categorize_predicate("destabilise") == PredicateCategory.CAUSAL_FUNCTIONAL

    # Relational & Structural
    assert categorize_predicate("includes") == PredicateCategory.RELATIONAL_STRUCTURAL
    assert categorize_predicate("have") == PredicateCategory.RELATIONAL_STRUCTURAL
    assert categorize_predicate("turn-to") == PredicateCategory.RELATIONAL_STRUCTURAL


def test_calculate_metrics():
    # Regular values
    p, r, f1 = calculate_metrics(tp=8, fp=9, fn=7)
    assert round(p, 2) == 0.47
    assert round(r, 2) == 0.53
    assert round(f1, 2) == 0.50

    # Zero values
    p_zero, r_zero, f1_zero = calculate_metrics(tp=0, fp=0, fn=5)
    assert p_zero == 0.0
    assert r_zero == 0.0
    assert f1_zero == 0.0


def test_exact_and_normalized_matchers():
    exact = ExactMatcher()
    norm = NormalizedMatcher()

    t_gold = Triple("cyber-physical attack", "causes", "blackout")
    t_exact = Triple("Cyber-Physical Attack", "CAUSES", "blackout")
    t_plural = Triple("cyber-physical attacks", "cause", "blackouts")
    t_unrelated = Triple("generator", "causes", "critical issue")

    # Exact matcher
    assert exact.match(t_exact, t_gold)[0] is True
    assert exact.match(t_plural, t_gold)[0] is False

    # Normalized matcher
    assert norm.match(t_exact, t_gold)[0] is True
    assert norm.match(t_plural, t_gold)[0] is True
    assert norm.match(t_unrelated, t_gold)[0] is False


def test_evaluate_triples_comprehensive():
    gold = [
        Triple("SG cyber security", "is-a", "critical issue"),
        Triple("cyber switching attack", "destabilise", "smart grid"),
        Triple("generator", "causes", "critical issue"),
    ]
    predicted = [
        # Match 1: exact
        Triple("SG cyber security", "is-a", "critical issue"),
        # Match 2: morphological normalized variant
        Triple("cyber switching attacks", "destabilises", "smart grids"),
        # False Positive (extraneous)
        Triple("solar panel", "generates", "clean electricity"),
    ]

    report = evaluate_triples(predicted, gold, model_name="TestModel")

    assert report.gold_total == 3
    assert report.pred_total == 3
    assert report.tp == 2
    assert report.fp == 1
    assert report.fn == 1
    assert round(report.precision, 2) == 0.67
    assert round(report.recall, 2) == 0.67
    assert round(report.f1_score, 2) == 0.67

    # Check predicate category breakdown
    assert PredicateCategory.HIERARCHICAL in report.category_metrics
    hier = report.category_metrics[PredicateCategory.HIERARCHICAL]
    assert hier.tp == 1
    assert hier.gold_count == 1

    causal = report.category_metrics[PredicateCategory.CAUSAL_FUNCTIONAL]
    assert causal.tp == 1
    assert causal.gold_count == 2  # destabilise + causes

    # Check predicate complexity
    assert report.avg_predicate_tokens >= 1.0


def test_paper_baseline_reproduction():
    reports = run_paper_baseline_evaluation()
    report_dict = {r.model_name: r for r in reports}

    # Verify LLMGraphTransformer
    lgt = report_dict["LLMGraphTransformer (Gemini 2.5 Flash)"]
    assert round(lgt.precision, 2) == 0.47
    assert round(lgt.recall, 2) == 0.53
    assert round(lgt.f1_score, 2) == 0.50

    # Verify KGGen Flash
    kgf = report_dict["KGGen (Gemini 2.5 Flash)"]
    assert round(kgf.precision, 2) == 0.40
    assert round(kgf.recall, 2) == 0.40
    assert round(kgf.f1_score, 2) == 0.40

    # Verify KGGen Pro
    kgp = report_dict["KGGen (Gemini Pro)"]
    assert round(kgp.precision, 2) == 0.54
    assert round(kgp.recall, 2) == 0.47
    assert round(kgp.f1_score, 2) == 0.50

    # Verify GT2KG exhibits highest precision on definitional/hierarchical triples
    gt2kg = report_dict["G-T2KG (GPT-4 OpenIE+Validation)"]
    assert gt2kg.precision > 0.65
    hier_tp = gt2kg.category_metrics[PredicateCategory.HIERARCHICAL].tp
    assert hier_tp >= 3  # captures hierarchical is-a relations
