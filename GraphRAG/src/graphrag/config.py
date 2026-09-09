"""Configuration for Neo4j and Gemini API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CREDS_FILE = Path("/home/arpangreat/Downloads/Neo4j-91fe5d8f-Created-2026-09-08.txt")


def _read_file_creds() -> dict[str, str]:
    creds = {}
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                creds[k] = v
    return creds


@dataclass
class Config:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    gemini_api_key: str

    @classmethod
    def load(cls) -> Config:
        file_creds = _read_file_creds()
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI") or file_creds.get("NEO4J_URI", "neo4j+s://91fe5d8f.databases.neo4j.io"),
            neo4j_user=os.getenv("NEO4J_USERNAME") or file_creds.get("NEO4J_USERNAME", "91fe5d8f"),
            neo4j_password=os.getenv("NEO4J_PASSWORD") or file_creds.get("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE") or file_creds.get("NEO4J_DATABASE", "91fe5d8f"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        )
