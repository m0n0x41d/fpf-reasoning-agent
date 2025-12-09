"""
Evidence Graph implementation with NetworkX + ArangoDB.

FPF A.10: Evidence-Provenance DAG
- ArangoDB: Persistent storage, cross-session
- NetworkX: In-memory algorithms, traversals, analysis

Architecture:
- All mutations go directly to ArangoDB
- NetworkX graph is loaded lazily for analysis
- Sync is one-way: ArangoDB is source of truth
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional, Set, Tuple
from uuid import UUID, uuid4

import networkx as nx
from arango.database import StandardDatabase

from ..trust.fgr import ClaimScope, CongruenceLevel, FGRTuple, FormalityLevel


class EvidenceRelation(str, Enum):
    """FPF A.10: Normative edge vocabulary."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    QUALIFIES = "qualifies"
    VERIFIED_BY = "verifiedBy"
    VALIDATED_BY = "validatedBy"
    DERIVED_FROM = "derivedFrom"
    FROM_WORK_SET = "fromWorkSet"
    HAPPENED_BEFORE = "happenedBefore"


@dataclass
class EvidenceLink:
    """A link in the evidence graph."""
    id: UUID
    from_id: UUID
    to_id: UUID
    relation: EvidenceRelation
    strength: float
    congruence_level: CongruenceLevel
    timespan_from: Optional[datetime] = None
    timespan_to: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_valid_at(self, t: datetime) -> bool:
        """Check if link is valid at given time."""
        if self.timespan_from and t < self.timespan_from:
            return False
        if self.timespan_to and t > self.timespan_to:
            return False
        return True


class EvidenceGraph:
    """
    Graph of evidential relationships between epistemes.

    FPF A.10: Evidence-Provenance DAG
    - Typed edges with normative vocabulary
    - CL penalties applied on composition
    - Temporal validity tracking

    Implementation:
    - ArangoDB for persistence
    - NetworkX for algorithms (loaded on demand)
    """

    def __init__(self, db: StandardDatabase):
        self.db = db
        self._links = db.collection("evidence_links")
        self._graph = db.graph("fpf_evidence_graph")

        self._nx_graph: Optional[nx.DiGraph] = None
        self._nx_dirty = True

    def add_link(
        self,
        claim_id: UUID,
        evidence_id: UUID,
        relation: EvidenceRelation | str,
        strength: float = 0.5,
        cl: CongruenceLevel = CongruenceLevel.CL3_PARTIAL,
        timespan: Optional[Tuple[datetime, datetime]] = None,
        notes: Optional[str] = None
    ) -> EvidenceLink:
        """
        Add evidence link (persisted immediately).

        FPF A.10: Links epistemes in evidence DAG.
        """
        if isinstance(relation, str):
            relation = EvidenceRelation(relation)

        link_id = uuid4()
        now = datetime.now(timezone.utc)

        doc = {
            "_key": str(link_id),
            "_from": f"epistemes/{claim_id}",
            "_to": f"epistemes/{evidence_id}",
            "relation": relation.value,
            "strength": strength,
            "congruence_level": cl.value,
            "timespan": {
                "valid_from": timespan[0].isoformat() if timespan and timespan[0] else None,
                "valid_until": timespan[1].isoformat() if timespan and timespan[1] else None
            },
            "notes": notes,
            "created_at": now.isoformat()
        }

        self._links.insert(doc)
        self._nx_dirty = True

        return EvidenceLink(
            id=link_id,
            from_id=claim_id,
            to_id=evidence_id,
            relation=relation,
            strength=strength,
            congruence_level=cl,
            timespan_from=timespan[0] if timespan else None,
            timespan_to=timespan[1] if timespan else None,
            notes=notes,
            created_at=now
        )

    def remove_link(self, link_id: UUID) -> bool:
        """Remove evidence link."""
        try:
            self._links.delete(str(link_id))
            self._nx_dirty = True
            return True
        except Exception:
            return False

    def get_evidence_for(
        self,
        claim_id: UUID,
        relation_filter: Optional[List[EvidenceRelation]] = None,
        valid_at: Optional[datetime] = None
    ) -> List[EvidenceLink]:
        """Get all evidence supporting/refuting a claim."""
        filters = ["e._from == @claim"]
        bind_vars = {"claim": f"epistemes/{claim_id}"}

        if relation_filter:
            filters.append("e.relation IN @relations")
            bind_vars["relations"] = [r.value for r in relation_filter]

        if valid_at:
            filters.append("""
                (e.timespan.valid_from == null OR e.timespan.valid_from <= @time)
                AND (e.timespan.valid_until == null OR e.timespan.valid_until >= @time)
            """)
            bind_vars["time"] = valid_at.isoformat()

        query = f"""
        FOR e IN evidence_links
            FILTER {' AND '.join(filters)}
            RETURN e
        """

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        return [self._doc_to_link(doc) for doc in cursor]

    def get_claims_supported_by(self, evidence_id: UUID) -> List[EvidenceLink]:
        """Get all claims this episteme provides evidence for."""
        query = """
        FOR e IN evidence_links
            FILTER e._to == @evidence
            RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={"evidence": f"epistemes/{evidence_id}"}
        )
        return [self._doc_to_link(doc) for doc in cursor]

    def _ensure_nx_graph(self) -> nx.DiGraph:
        """Load/refresh NetworkX graph from ArangoDB."""
        if self._nx_graph is None or self._nx_dirty:
            self._nx_graph = nx.DiGraph()

            query = "FOR e IN evidence_links RETURN e"
            cursor = self.db.aql.execute(query)

            for doc in cursor:
                from_id = doc["_from"].split("/")[1]
                to_id = doc["_to"].split("/")[1]

                self._nx_graph.add_edge(
                    from_id,
                    to_id,
                    id=doc["_key"],
                    relation=doc["relation"],
                    strength=doc["strength"],
                    congruence_level=doc["congruence_level"],
                    timespan=doc.get("timespan", {})
                )

            self._nx_dirty = False

        return self._nx_graph

    def invalidate_cache(self) -> None:
        """Force reload of NetworkX graph on next access."""
        self._nx_dirty = True

    def compute_support(
        self,
        claim_id: UUID,
        valid_at: Optional[datetime] = None
    ) -> Tuple[float, List[str]]:
        """
        Compute aggregate support for a claim using NetworkX.

        Returns (reliability_contribution, audit_trail)
        """
        G = self._ensure_nx_graph()
        claim_str = str(claim_id)

        if claim_str not in G:
            return 0.0, ["Claim not found in evidence graph"]

        audit = []
        total_support = 0.0
        total_refute = 0.0

        for pred in G.predecessors(claim_str):
            edge = G.edges[pred, claim_str]

            if valid_at:
                ts = edge.get("timespan", {})
                if ts.get("valid_from"):
                    vf = datetime.fromisoformat(ts["valid_from"].replace("Z", "+00:00"))
                    if valid_at < vf:
                        continue
                if ts.get("valid_until"):
                    vu = datetime.fromisoformat(ts["valid_until"].replace("Z", "+00:00"))
                    if valid_at > vu:
                        audit.append(f"Expired: {pred[:8]}... (until {vu})")
                        continue

            cl = CongruenceLevel(edge["congruence_level"])
            cl_factor = cl.value / 5.0
            weighted = edge["strength"] * cl_factor
            relation = edge["relation"]

            if relation in ["supports", "verifiedBy", "validatedBy"]:
                total_support += weighted
                audit.append(f"+{weighted:.2f} from {pred[:8]}... [{relation}] (CL{cl.value})")
            elif relation == "refutes":
                total_refute += weighted
                audit.append(f"-{weighted:.2f} from {pred[:8]}... [{relation}] (CL{cl.value})")
            else:
                audit.append(f"~ {pred[:8]}... [{relation}]")

        net_support = max(0.0, min(1.0, total_support - total_refute))
        audit.append(f"Net support: {net_support:.2f}")

        return net_support, audit

    def find_evidence_paths(
        self,
        claim_id: UUID,
        max_depth: int = 5
    ) -> List[List[str]]:
        """
        Find all evidence paths leading to a claim.

        FPF G.6: Path-addressable provenance.
        """
        G = self._ensure_nx_graph()
        claim_str = str(claim_id)

        if claim_str not in G:
            return []

        paths: List[List[str]] = []

        def find_paths_recursive(node: str, path: List[str], depth: int) -> None:
            if depth > max_depth:
                return

            current_path = path + [node]
            preds = list(G.predecessors(node))

            if not preds:
                paths.append(current_path)
            else:
                for pred in preds:
                    if pred not in path:
                        find_paths_recursive(pred, current_path, depth + 1)

        find_paths_recursive(claim_str, [], 0)
        return paths

    def compute_path_trust(
        self,
        path: List[str]
    ) -> Tuple[FGRTuple, List[str]]:
        """
        Compute trust tuple for an evidence path.

        FPF B.3: WLNK invariant — result bounded by weakest link.
        """
        G = self._ensure_nx_graph()
        audit = []

        if len(path) < 2:
            return FGRTuple(
                formality=FormalityLevel.F0_INFORMAL,
                claim_scope=ClaimScope(),
                reliability=0.0
            ), ["Path too short"]

        min_strength = 1.0
        min_cl = CongruenceLevel.CL5_EXACT

        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i+1]):
                edge = G.edges[path[i], path[i+1]]
                strength = edge["strength"]
                cl = CongruenceLevel(edge["congruence_level"])

                min_strength = min(min_strength, strength)
                if cl.value < min_cl.value:
                    min_cl = cl

                audit.append(f"{path[i][:8]}...→{path[i+1][:8]}...: S={strength:.2f}, CL{cl.value}")

        cl_penalty = (5 - min_cl.value) * 0.1
        final_r = max(0.0, min_strength - cl_penalty)

        audit.append(f"WLNK: min_S={min_strength:.2f}, min_CL={min_cl.value}, penalty={cl_penalty:.2f}")
        audit.append(f"Final R: {final_r:.2f}")

        return FGRTuple(
            formality=FormalityLevel.F0_INFORMAL,
            claim_scope=ClaimScope(),
            reliability=final_r
        ), audit

    def get_orphan_claims(self, context_id: Optional[str] = None) -> List[str]:
        """Find claims with no evidence support (NetworkX analysis)."""
        G = self._ensure_nx_graph()

        orphans = []
        for node in G.nodes():
            has_support = False
            for pred in G.predecessors(node):
                edge = G.edges[pred, node]
                if edge["relation"] in ["supports", "verifiedBy", "validatedBy"]:
                    has_support = True
                    break
            if not has_support:
                orphans.append(node)

        return orphans

    def compute_transitive_closure(
        self,
        root_id: UUID,
        relation_types: Optional[List[EvidenceRelation]] = None
    ) -> Set[str]:
        """
        Get all nodes reachable from root via specified relations.

        Useful for impact analysis.
        """
        G = self._ensure_nx_graph()
        root_str = str(root_id)

        if root_str not in G:
            return set()

        if relation_types:
            relations = {r.value for r in relation_types}
            filtered = nx.DiGraph()
            for u, v, data in G.edges(data=True):
                if data["relation"] in relations:
                    filtered.add_edge(u, v, **data)
            G = filtered

        return nx.ancestors(G, root_str) | {root_str}

    def visualize_subgraph(
        self,
        center_id: UUID,
        depth: int = 2
    ) -> dict:
        """
        Get subgraph data for visualization.

        Returns dict suitable for rendering.
        """
        G = self._ensure_nx_graph()
        center_str = str(center_id)

        if center_str not in G:
            return {"nodes": [], "edges": []}

        subgraph = nx.ego_graph(G, center_str, radius=depth, undirected=True)

        nodes = [
            {"id": n, "is_center": n == center_str}
            for n in subgraph.nodes()
        ]

        edges = [
            {
                "from": u,
                "to": v,
                "relation": data["relation"],
                "strength": data["strength"],
                "cl": data["congruence_level"]
            }
            for u, v, data in subgraph.edges(data=True)
        ]

        return {"nodes": nodes, "edges": edges}

    def _doc_to_link(self, doc: dict) -> EvidenceLink:
        """Convert ArangoDB document to EvidenceLink."""
        ts = doc.get("timespan", {})

        ts_from = None
        if ts.get("valid_from"):
            ts_from = datetime.fromisoformat(ts["valid_from"].replace("Z", "+00:00"))

        ts_to = None
        if ts.get("valid_until"):
            ts_to = datetime.fromisoformat(ts["valid_until"].replace("Z", "+00:00"))

        created = datetime.now(timezone.utc)
        if doc.get("created_at"):
            created = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))

        return EvidenceLink(
            id=UUID(doc["_key"]),
            from_id=UUID(doc["_from"].split("/")[1]),
            to_id=UUID(doc["_to"].split("/")[1]),
            relation=EvidenceRelation(doc["relation"]),
            strength=doc["strength"],
            congruence_level=CongruenceLevel(doc["congruence_level"]),
            timespan_from=ts_from,
            timespan_to=ts_to,
            notes=doc.get("notes"),
            created_at=created
        )
