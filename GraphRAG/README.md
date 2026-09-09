# PDF Knowledge Graph Builder (Neo4j)

A lightweight Python application running inside a `uv` virtual environment that extracts text from any PDF document, extracts key entities and relationships using Google Gemini, and constructs an interconnected knowledge graph directly in **Neo4j Aura** with **zero data redundancy**.

Visualization and graph exploration are performed using **Neo4j's native Browser & Bloom interfaces**.

---

## Key Features

1. **Any PDF Retrieval & Processing**:
   - Reads arbitrary PDFs from disk via `pypdf`.
   - Computes SHA-256 cryptographic hashes for document-level deduplication.
   - Splits text into semantically coherent chunks preserving page provenance.

2. **Strict Zero Data Redundancy**:
   - **Document Uniqueness**: Neo4j constraint `d.file_hash IS UNIQUE`. Duplicate PDFs are automatically detected and skipped (or updated idempotently with `--force`).
   - **Chunk Uniqueness**: Neo4j constraint `c.id IS UNIQUE`.
   - **Canonical Entity Deduplication**:
     - Normalizes entity names to deterministic IDs (`canonical_id`), stripping punctuation and normalizing case (e.g. `"Google, Inc."` and `"Google"` converge to `google`).
     - Neo4j constraint `e.id IS UNIQUE`.
     - Assigns secondary category labels (e.g., `:Entity:Person`, `:Entity:Organization`) so Neo4j Browser automatically color-codes categories.
     - Uses Cypher `MERGE (e:Entity {id: $id})` to increment mention counts and enrich descriptions without creating duplicate nodes.
   - **Relationship Deduplication**:
     - Uses Cypher `MERGE (s)-[r:REL_TYPE]->(t)`. Re-occurring relationships increment connection `weight` and append chunk IDs to provenance instead of creating duplicate edges.

3. **Visualized Directly in Neo4j**:
   - One-command launcher (`uv run graphrag browser`) opens Neo4j Browser with the remote Aura connection pre-configured.
   - Run interactive Cypher queries directly from the CLI or in the Neo4j web console.

---

## Project Structure

```
GraphRAG/
├── .env                  # Neo4j credentials and Gemini API key
├── pyproject.toml        # UV project configuration and dependencies
├── README.md             # Documentation
├── data/                 # Sample documents (e.g., ai_revolution.pdf)
├── tests/
│   └── test_pipeline.py  # Automated test suite
└── src/
    └── graphrag/
        ├── __init__.py   # Package exports (KnowledgeGraph, load_pdf, Extractor, main)
        ├── config.py     # Configuration loader (.env & ~/Downloads/Neo4j-*.txt fallback)
        ├── pdf.py        # PDF text extraction & chunker
        ├── extractor.py  # Gemini entity & relation extraction with canonicalization
        ├── graph.py      # Neo4j database service with uniqueness constraints
        └── cli.py        # Clean CLI interface
```

---

## Configuration

Credentials can be stored in `.env` (automatically loaded from `~/Downloads/Neo4j-91fe5d8f-Created-2026-09-08.txt`):

```env
NEO4J_URI=neo4j+s://91fe5d8f.databases.neo4j.io
NEO4J_USERNAME=91fe5d8f
NEO4J_PASSWORD=fLALwy-bdXet6O-cEukr_lfpedlwkSiO8lEkAsdKLXc
NEO4J_DATABASE=91fe5d8f
GEMINI_API_KEY=<your-gemini-api-key>
```

---

## CLI Usage

### 1. Ingest a PDF File
```bash
uv run graphrag ingest paper49.pdf
```
*(To re-ingest an already processed document, add `--force`).*

### 2. Open Neo4j Browser to Visualize the Graph
```bash
uv run graphrag browser
```
This prints your Aura instance credentials and automatically opens the Neo4j Browser in your default web browser pre-connected to your database.

### 3. Recommended Cypher Queries for Neo4j Browser

Run any of these queries in Neo4j Browser's query bar to visualize your graph:

- **Explore all entities and connections**:
  ```cypher
  MATCH (s:Entity)-[r]->(t:Entity)
  RETURN s, r, t
  LIMIT 100
  ```
- **Inspect provenance (Document ➔ Chunk ➔ Entity)**:
  ```cypher
  MATCH (d:Document)-[r1:HAS_CHUNK]->(c:Chunk)-[r2:MENTIONS]->(e:Entity)
  RETURN d, r1, c, r2, e
  LIMIT 50
  ```
- **View relationships between Persons and Organizations**:
  ```cypher
  MATCH (p:Person)-[r]-(o:Organization)
  RETURN p, r, o
  ```
- **2-hop neighborhood around a specific entity**:
  ```cypher
  MATCH path = (e:Entity {name: 'OpenAI'})-[*1..2]-(other)
  RETURN path
  ```

### 4. Run Cypher Queries Directly from Terminal
```bash
uv run graphrag cypher "MATCH (e:Entity) RETURN e.name, e.type, e.mentions ORDER BY e.mentions DESC LIMIT 10"
```

### 5. View Graph Summary Statistics
```bash
uv run graphrag stats
```

### 6. Search Entities
```bash
uv run graphrag search "AlphaGo"
```

### 7. Clear / Reset Knowledge Graph
```bash
uv run graphrag clear
```

---

## Running Tests

```bash
uv run pytest
```
