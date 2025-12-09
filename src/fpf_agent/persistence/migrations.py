"""
Database migrations for FPF episteme store.

Schema v2 supports:
- Bounded contexts with glossaries
- Versioned epistemes with edition chains
- Evidence graph with CL tracking
- ADI cycle state persistence
- Research session management
"""
import sqlite3
from pathlib import Path


SCHEMA_V2 = """
-- Bounded Contexts (FPF A.1.1)
CREATE TABLE IF NOT EXISTS contexts (
    context_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    glossary_json TEXT DEFAULT '{}',
    invariants_json TEXT DEFAULT '[]',
    parent_context_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version TEXT DEFAULT '1.0.0',
    FOREIGN KEY (parent_context_id) REFERENCES contexts(context_id)
);

-- Epistemes with versioning (FPF C.2.1)
CREATE TABLE IF NOT EXISTS epistemes (
    id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    edition_number INTEGER DEFAULT 1,
    supersedes_id TEXT,

    -- Core content
    described_entity TEXT NOT NULL,
    claim_graph_json TEXT DEFAULT '{}',
    grounding_holon_id TEXT,
    viewpoint TEXT,

    -- Strict Distinction slots (A.7/A.15)
    structure_json TEXT,
    order_json TEXT,
    time_json TEXT,
    work_json TEXT,
    values_json TEXT,

    -- Lifecycle & Assurance (B.5.1, B.3.3)
    lifecycle_state TEXT DEFAULT 'exploration',
    assurance_level TEXT DEFAULT 'L0',
    temporal_stance INTEGER DEFAULT 0,

    -- F-G-R (B.3)
    formality INTEGER DEFAULT 0,
    claim_scope_json TEXT DEFAULT '{}',
    reliability REAL DEFAULT 0.0,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (context_id) REFERENCES contexts(context_id),
    FOREIGN KEY (supersedes_id) REFERENCES epistemes(id)
);

-- Evidence Graph (FPF A.10)
CREATE TABLE IF NOT EXISTS evidence_links (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    congruence_level INTEGER DEFAULT 3,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (claim_id) REFERENCES epistemes(id),
    FOREIGN KEY (evidence_id) REFERENCES epistemes(id)
);

-- Hypotheses (FPF B.5.2)
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    episteme_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    triggered_by TEXT,

    -- NQD metrics (C.17)
    novelty_score REAL,
    quality_score REAL,
    diversity_score REAL,

    -- Status
    status TEXT DEFAULT 'proposed',

    -- Alternatives (for NQD tracking)
    alternatives_json TEXT DEFAULT '[]',
    rejection_reasons_json TEXT DEFAULT '[]',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (episteme_id) REFERENCES epistemes(id)
);

-- ADI Cycle tracking (FPF B.5)
CREATE TABLE IF NOT EXISTS adi_cycles (
    id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    current_phase TEXT DEFAULT 'abduction',

    -- Phase outputs (JSON)
    abduction_result_json TEXT,
    deduction_result_json TEXT,
    induction_result_json TEXT,

    -- Final episteme
    result_episteme_id TEXT,

    -- Iteration tracking
    iteration_count INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 3,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (context_id) REFERENCES contexts(context_id),
    FOREIGN KEY (result_episteme_id) REFERENCES epistemes(id)
);

-- Research sessions
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    research_question TEXT NOT NULL,
    methodology TEXT DEFAULT 'systematic',

    -- State
    current_state TEXT DEFAULT 'active',

    -- Related entities
    hypothesis_ids_json TEXT DEFAULT '[]',
    episteme_ids_json TEXT DEFAULT '[]',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (context_id) REFERENCES contexts(context_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_epistemes_context ON epistemes(context_id);
CREATE INDEX IF NOT EXISTS idx_epistemes_entity ON epistemes(described_entity);
CREATE INDEX IF NOT EXISTS idx_epistemes_lifecycle ON epistemes(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_epistemes_supersedes ON epistemes(supersedes_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence_links(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_evidence ON evidence_links(evidence_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_episteme ON hypotheses(episteme_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_adi_context ON adi_cycles(context_id);
CREATE INDEX IF NOT EXISTS idx_sessions_context ON research_sessions(context_id);
"""


def migrate_to_v2(db_path: str | Path) -> None:
    """
    Run migration to v2 schema.

    Safe to run multiple times — uses IF NOT EXISTS.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_V2)
    conn.commit()
    conn.close()


def get_schema_version(db_path: str | Path) -> int:
    """Get current schema version from database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='epistemes'"
        )
        if not cursor.fetchone():
            conn.close()
            return 0

        cursor.execute("PRAGMA table_info(epistemes)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        if "claim_scope_json" in columns:
            return 2
        if "claim_graph_json" in columns:
            return 1
        return 0

    except sqlite3.Error:
        return 0


def check_migration_needed(db_path: str | Path) -> bool:
    """Check if database needs migration."""
    return get_schema_version(db_path) < 2
