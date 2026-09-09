"""Entity and Relationship Extractor with Canonicalization."""

from __future__ import annotations

import re
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from graphrag.config import Config


class RawEntity(BaseModel):
    name: str = Field(description="Canonical entity name (e.g. 'Albert Einstein', 'Google'). Avoid pronouns.")
    type: str = Field(description="Category: PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, LOCATION, EVENT, PRODUCT")
    description: str = Field(description="Brief 1-sentence description based on text")


class RawRelation(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="UPPER_SNAKE_CASE verb phrase (e.g. FOUNDED, DEVELOPED, LOCATED_IN)")
    description: str = Field(description="Brief explanation of the relationship")


class ExtractionData(BaseModel):
    entities: list[RawEntity]
    relations: list[RawRelation]


def canonical_id(name: str) -> str:
    """Normalize entity name to a unique, deterministic ID to prevent duplicates."""
    cleaned = re.sub(r"[^\w\s]", "", name.strip().lower())
    return re.sub(r"\s+", "_", cleaned)


def normalize_rel_type(rel: str) -> str:
    """Convert relation verb to uppercase Cypher-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", rel.strip().upper()).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        return "RELATED_TO"
    return cleaned


class Extractor:
    """Extracts deduplicated entities and relations using Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        cfg = Config.load()
        self.client = genai.Client(api_key=api_key or cfg.gemini_api_key)

    def extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """Extract canonical entities and relations from text."""
        prompt = f"""Extract all key named entities and the explicit relationships connecting them.
RULES FOR ZERO DUPLICATION:
1. Always use full canonical names (never pronouns like "he", "they", "this company").
2. Source and target in relations must exactly match names in the entities list.

Text:
\"\"\"
{text}
\"\"\"
"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionData,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            res = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            data = ExtractionData.model_validate_json(res.text)
        except Exception:
            return [], []

        # Deduplicate entities by canonical ID
        entities_by_id: dict[str, dict] = {}
        for ent in data.entities:
            eid = canonical_id(ent.name)
            if not eid:
                continue
            if eid not in entities_by_id:
                entities_by_id[eid] = {
                    "id": eid,
                    "name": ent.name.strip(),
                    "type": ent.type.strip().upper(),
                    "description": ent.description.strip(),
                }

        # Deduplicate relations
        relations_map: dict[tuple[str, str, str], dict] = {}
        for r in data.relations:
            src_id = canonical_id(r.source)
            tgt_id = canonical_id(r.target)
            rel_type = normalize_rel_type(r.relation)

            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            # Ensure endpoints exist
            if src_id not in entities_by_id:
                entities_by_id[src_id] = {"id": src_id, "name": r.source.strip(), "type": "CONCEPT", "description": ""}
            if tgt_id not in entities_by_id:
                entities_by_id[tgt_id] = {"id": tgt_id, "name": r.target.strip(), "type": "CONCEPT", "description": ""}

            key = (src_id, tgt_id, rel_type)
            relations_map[key] = {
                "source_id": src_id,
                "target_id": tgt_id,
                "type": rel_type,
                "description": r.description.strip(),
            }

        return list(entities_by_id.values()), list(relations_map.values())
