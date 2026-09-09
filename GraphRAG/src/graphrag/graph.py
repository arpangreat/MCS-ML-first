"""Neo4j Knowledge Graph Client with Uniqueness Constraints and Deduplication."""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, Callable, Optional
from neo4j import GraphDatabase, Driver, NotificationMinimumSeverity

from graphrag.config import Config
from graphrag.extractor import Extractor, normalize_rel_type
from graphrag.pdf import PDFData

logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """Sanitize entity type to a clean Cypher PascalCase label (e.g. 'ORGANIZATION' -> 'Organization')."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", label.strip())
    return cleaned.capitalize() if cleaned else "Concept"


class KnowledgeGraph:
    """Handles connection, deduplication, schema, and queries in Neo4j."""

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config.load()
        self.driver: Driver = GraphDatabase.driver(
            self.cfg.neo4j_uri,
            auth=(self.cfg.neo4j_user, self.cfg.neo4j_password),
            notifications_min_severity=NotificationMinimumSeverity.OFF,
        )
        self.database = self.cfg.neo4j_database
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> KnowledgeGraph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints in Neo4j to enforce zero duplication."""
        queries = [
            "CREATE CONSTRAINT document_hash_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.file_hash IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception:
                    pass

    def document_exists(self, file_hash: str) -> bool:
        """Check if document hash already exists in graph."""
        with self.driver.session(database=self.database) as session:
            res = session.run("MATCH (d:Document {file_hash: $h}) RETURN count(d) > 0 AS ex", h=file_hash).single()
            return bool(res and res["ex"])

    def ingest_pdf(
        self,
        pdf: PDFData,
        extractor: Extractor,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest PDF into Neo4j without data redundancy:
        1. Checks content SHA-256 to avoid duplicate ingestion.
        2. Merges Document & Chunk nodes.
        3. Extracts entities & relations per chunk and merges them canonically.
        """
        if self.document_exists(pdf.file_hash) and not force:
            return {
                "status": "skipped",
                "message": f"Document '{pdf.filename}' already exists in graph (SHA-256 match).",
                "chunks": len(pdf.chunks),
            }

        with self.driver.session(database=self.database) as session:
            # 1. Merge Document node
            session.run(
                """
                MERGE (d:Document {file_hash: $hash})
                ON CREATE SET d.filename = $filename, d.pages = $pages, d.chunks = $chunks, d.created_at = datetime()
                ON MATCH SET d.filename = $filename, d.updated_at = datetime()
                """,
                hash=pdf.file_hash,
                filename=pdf.filename,
                pages=pdf.page_count,
                chunks=len(pdf.chunks),
            )

        total_chunks = len(pdf.chunks)
        total_ents = 0
        total_rels = 0

        for idx, chunk in enumerate(pdf.chunks, start=1):
            if progress_cb:
                progress_cb(idx, total_chunks, f"Processing chunk {idx}/{total_chunks}...")

            # Extract entities and relations
            ents, rels = extractor.extract(chunk.text)
            total_ents += len(ents)
            total_rels += len(rels)

            with self.driver.session(database=self.database) as session:
                # 2. Merge Chunk node and link to Document
                session.run(
                    """
                    MATCH (d:Document {file_hash: $doc_hash})
                    MERGE (c:Chunk {id: $chunk_id})
                    ON CREATE SET c.text = $text, c.index = $index, c.pages = [$p_start, $p_end]
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    doc_hash=chunk.doc_hash,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    index=chunk.index,
                    p_start=chunk.page_start,
                    p_end=chunk.page_end,
                )

                # 3. Merge Entities grouped by type label (idempotent, increments mention_count)
                # Grouping by type allows setting category labels (e.g. :Entity:Person) for rich Neo4j Browser visualization
                by_type: dict[str, list[dict]] = {}
                for e in ents:
                    by_type.setdefault(e.get("type", "CONCEPT"), []).append(e)

                for raw_type, group_ents in by_type.items():
                    label = sanitize_label(raw_type)
                    cypher_ent = f"""
                    UNWIND $ents AS ent
                    MERGE (e:Entity:`{label}` {{id: ent.id}})
                    ON CREATE SET
                        e.name = ent.name,
                        e.type = ent.type,
                        e.description = ent.description,
                        e.mentions = 1
                    ON MATCH SET
                        e.mentions = coalesce(e.mentions, 0) + 1,
                        e.description = CASE
                            WHEN size(coalesce(e.description, '')) < size(ent.description) THEN ent.description
                            ELSE e.description
                        END
                    WITH e
                    MATCH (c:Chunk {{id: $chunk_id}})
                    MERGE (c)-[:MENTIONS]->(e)
                    """
                    session.run(cypher_ent, ents=group_ents, chunk_id=chunk.id)

                # 4. Merge Relations (prevents duplicate edges between same entities)
                for r in rels:
                    rel_type = normalize_rel_type(r["type"])
                    session.run(
                        f"""
                        MATCH (s:Entity {{id: $src}})
                        MATCH (t:Entity {{id: $tgt}})
                        WHERE s <> t
                        MERGE (s)-[r:`{rel_type}`]->(t)
                        ON CREATE SET
                            r.description = $desc,
                            r.weight = 1,
                            r.chunks = [$chunk_id]
                        ON MATCH SET
                            r.weight = coalesce(r.weight, 1) + 1,
                            r.chunks = CASE
                                WHEN NOT $chunk_id IN coalesce(r.chunks, []) THEN coalesce(r.chunks, []) + $chunk_id
                                ELSE r.chunks
                            END
                        """,
                        src=r["source_id"],
                        tgt=r["target_id"],
                        desc=r["description"],
                        chunk_id=chunk.id,
                    )

        return {
            "status": "success",
            "filename": pdf.filename,
            "hash": pdf.file_hash,
            "chunks_processed": total_chunks,
            "entities_found": total_ents,
            "relations_found": total_rels,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of the knowledge graph."""
        with self.driver.session(database=self.database) as session:
            doc_cnt = session.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
            chk_cnt = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            ent_cnt = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            rel_cnt = session.run("MATCH ()-[r]->() WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS' RETURN count(r) AS c").single()["c"]

            type_res = session.run("MATCH (e:Entity) RETURN e.type AS type, count(e) AS count ORDER BY count DESC")
            types = {r["type"]: r["count"] for r in type_res}

            top_res = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-(o:Entity)
                WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS'
                RETURN e.name AS name, e.type AS type, count(r) AS connections
                ORDER BY connections DESC LIMIT 10
            """)
            top = [dict(r) for r in top_res]

        return {
            "documents": doc_cnt,
            "chunks": chk_cnt,
            "entities": ent_cnt,
            "relationships": rel_cnt,
            "types": types,
            "top_entities": top,
        }

    def search_entities(self, query: str, limit: int = 15) -> list[dict[str, Any]]:
        """Search entities by name or type."""
        q = query.strip().lower()
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS $q OR toLower(coalesce(e.type, '')) CONTAINS $q
        OPTIONAL MATCH (e)-[r]-(o:Entity)
        WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS'
        RETURN e.id AS id, e.name AS name, e.type AS type, coalesce(e.description, '') AS description,
               coalesce(e.mentions, 1) AS mentions, count(r) AS connections
        ORDER BY connections DESC LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, q=q, limit=limit)]

    def execute_cypher(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Execute arbitrary Cypher query and return dictionaries."""
        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, **(params or {}))]

    def get_subgraph(self, max_nodes: int = 150) -> dict[str, list]:
        """Fetch nodes and edges for visual rendering."""
        cypher = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS'
        WITH s, r, t LIMIT $max
        WITH collect(DISTINCT s) + collect(DISTINCT t) AS all_nodes, collect(r) AS rels
        UNWIND all_nodes AS n
        WITH DISTINCT n, rels
        RETURN collect(DISTINCT {
            id: n.id,
            label: n.name,
            type: coalesce(n.type, 'CONCEPT'),
            desc: coalesce(n.description, ''),
            mentions: coalesce(n.mentions, 1)
        }) AS nodes,
        [rel IN rels | {
            from: startNode(rel).id,
            to: endNode(rel).id,
            label: type(rel),
            desc: coalesce(rel.description, '')
        }] AS edges
        """
        with self.driver.session(database=self.database) as session:
            res = session.run(cypher, max=max_nodes).single()
            if not res or not res["nodes"]:
                return {"nodes": [], "edges": []}
            return {"nodes": res["nodes"], "edges": res["edges"]}

    def get_browser_url(self) -> str:
        """Construct the direct Neo4j Browser connection URL with target database preselected."""
        connect_param = urllib.parse.quote(self.cfg.neo4j_uri, safe="")
        return f"https://browser.neo4j.io/?connectURL={connect_param}&db={self.database}"

    def clear(self) -> int:
        """Clear all nodes and relationships."""
        with self.driver.session(database=self.database) as session:
            cnt = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            session.run("MATCH (n) DETACH DELETE n")
            return cnt
