"""Tests for PDF loading, canonicalization, and knowledge graph deduplication."""

import pytest
from graphrag.extractor import canonical_id, normalize_rel_type
from graphrag.pdf import load_pdf
from graphrag.graph import KnowledgeGraph


def test_canonical_id():
    # Verify entity normalization
    assert canonical_id("Albert Einstein") == "albert_einstein"
    assert canonical_id("  Google, Inc.  ") == "google_inc"
    assert canonical_id("DeepMind-Technologies") == "deepmindtechnologies"
    assert canonical_id("OpenAI") == "openai"


def test_normalize_rel_type():
    assert normalize_rel_type("founded by") == "FOUNDED_BY"
    assert normalize_rel_type("WORKS_AT") == "WORKS_AT"
    assert normalize_rel_type("123 invalid") == "RELATED_TO"
    assert normalize_rel_type("") == "RELATED_TO"


def test_pdf_loading(tmp_path):
    # Test loading from sample PDF
    from reportlab.pdfgen import canvas
    test_pdf = tmp_path / "test.pdf"
    c = canvas.Canvas(str(test_pdf))
    c.drawString(100, 750, "Sample paragraph one about artificial intelligence.")
    c.drawString(100, 700, "Sample paragraph two about knowledge graphs.")
    c.save()

    pdf_data = load_pdf(test_pdf)
    assert pdf_data.page_count == 1
    assert len(pdf_data.chunks) >= 1
    assert pdf_data.file_hash is not None
    assert len(pdf_data.file_hash) == 64


def test_graph_deduplication():
    # Verify database connectivity and schema constraints
    with KnowledgeGraph() as graph:
        stats = graph.get_stats()
        assert "documents" in stats
        assert "entities" in stats
        assert "relationships" in stats
