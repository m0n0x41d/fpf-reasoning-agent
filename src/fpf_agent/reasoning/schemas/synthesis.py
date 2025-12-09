"""
SGR Schema for Knowledge Synthesis

Synthesis: Integrate knowledge from multiple epistemes into coherent understanding.

The schema forces the LLM through:
1. Source analysis - what sources contribute
2. Consensus identification - claims supported by multiple sources
3. Conflict detection - claims where sources disagree
4. Gap identification - what is NOT covered
5. Integration - synthesized statement with confidence

This is used after ADI cycles to combine validated epistemes.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class ConflictResolution(str, Enum):
    """How a conflict between sources was resolved."""

    MAJORITY = "majority"
    """Went with the majority view."""

    RECENCY = "recency"
    """More recent source preferred."""

    AUTHORITY = "authority"
    """Higher authority/reliability source preferred."""

    SPECIFICITY = "specificity"
    """More specific claim preferred over general."""

    RECONCILED = "reconciled"
    """Found a way to reconcile both claims."""

    UNRESOLVED = "unresolved"
    """Conflict remains unresolved - noted as disputed."""


class SourceContribution(BaseModel):
    """Contribution from a single source episteme."""

    source_id: str = Field(
        description="ID of the contributing episteme."
    )

    source_entity: str = Field(
        description="What the source describes."
    )

    assurance_level: str = Field(
        description="Assurance level of the source (L0/L1/L2)."
    )

    key_claims: list[str] = Field(
        default_factory=list,
        description="Key claims extracted from this source."
    )

    reliability: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Reliability of this source.",
        ),
    ]

    weight_in_synthesis: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Weight given to this source in synthesis [0-1].",
        ),
    ]


class DisputedClaim(BaseModel):
    """A claim where sources disagree."""

    claim_a: str = Field(
        description="One version of the claim."
    )

    source_a_ids: list[str] = Field(
        description="Sources supporting claim A."
    )

    claim_b: str = Field(
        description="Conflicting version of the claim."
    )

    source_b_ids: list[str] = Field(
        description="Sources supporting claim B."
    )

    nature_of_conflict: str = Field(
        description="What kind of conflict is this? "
        "(contradiction, scope difference, level of detail, etc.)"
    )

    resolution: ConflictResolution = Field(
        description="How was this conflict resolved (or not)?"
    )

    resolution_rationale: str = Field(
        default="",
        description="Why this resolution approach was chosen."
    )

    synthesized_claim: str = Field(
        default="",
        description="The resulting claim after resolution (if resolved)."
    )


class KnowledgeGap(BaseModel):
    """An identified gap in the knowledge base."""

    gap_description: str = Field(
        description="What is missing or unknown."
    )

    importance: Annotated[
        str,
        Field(
            description="How important is filling this gap? "
            "'critical', 'important', 'nice_to_have'",
        ),
    ] = "important"

    suggested_investigation: str = Field(
        default="",
        description="Suggested approach to fill this gap."
    )

    related_sources: list[str] = Field(
        default_factory=list,
        description="Source IDs that hint at this gap."
    )


class SynthesisOutput(BaseModel):
    """
    Complete output of knowledge synthesis.

    This schema ensures:
    - All sources are accounted for
    - Consensus and conflicts are explicit
    - Gaps are identified
    - Final synthesis has proper confidence bounds
    """

    # Context
    topic: str = Field(
        description="The topic being synthesized."
    )

    synthesis_scope: str = Field(
        description="What is in/out of scope for this synthesis."
    )

    # Step 1: Source analysis
    sources: list[SourceContribution] = Field(
        default_factory=list,
        description="All sources contributing to this synthesis."
    )

    # Step 2: Consensus claims
    consensus_claims: list[str] = Field(
        default_factory=list,
        description="Claims supported by multiple sources (or single high-authority source)."
    )

    # Step 3: Disputed claims
    disputed_claims: list[DisputedClaim] = Field(
        default_factory=list,
        description="Claims where sources disagree."
    )

    # Step 4: Gaps
    gaps_identified: list[KnowledgeGap] = Field(
        default_factory=list,
        description="What is NOT covered by the sources."
    )

    # Step 5: Synthesis
    synthesis_statement: str = Field(
        min_length=20,
        description="Integrated summary of current knowledge on the topic."
    )

    key_findings: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Key findings from the synthesis."
    )

    # Confidence and limitations
    overall_confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Overall confidence in the synthesis [0-1]. "
            "Bounded by source reliability and conflict resolution.",
        ),
    ]

    confidence_rationale: str = Field(
        description="Why this confidence level was assigned."
    )

    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations of this synthesis."
    )

    # For episteme creation
    recommended_assurance: str = Field(
        default="L0",
        description="Recommended assurance level for the synthesized episteme."
    )

    @property
    def source_count(self) -> int:
        """Number of sources used."""
        return len(self.sources)

    @property
    def has_unresolved_conflicts(self) -> bool:
        """Check if any conflicts are unresolved."""
        return any(
            d.resolution == ConflictResolution.UNRESOLVED
            for d in self.disputed_claims
        )

    @property
    def critical_gaps(self) -> list[KnowledgeGap]:
        """Get critical knowledge gaps."""
        return [g for g in self.gaps_identified if g.importance == "critical"]

    def to_episteme_claim_graph(self) -> dict:
        """
        Convert to claim_graph format for UEpisteme.
        """
        return {
            "type": "synthesis",
            "statement": self.synthesis_statement,
            "key_findings": self.key_findings,
            "consensus_claims": self.consensus_claims,
            "disputed_claims": [
                {
                    "claim": d.synthesized_claim or f"{d.claim_a} vs {d.claim_b}",
                    "resolution": d.resolution.value,
                }
                for d in self.disputed_claims
            ],
            "gaps": [g.gap_description for g in self.gaps_identified],
            "source_ids": [s.source_id for s in self.sources],
            "source_count": self.source_count,
        }


# =============================================================================
# SYSTEM PROMPT FOR SYNTHESIS
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a knowledge synthesis engine.

Your task: Integrate knowledge from multiple sources into coherent understanding.

## Rules

1. **Account for all sources**
   - Weight by reliability and assurance level
   - Note what each source contributes

2. **Identify consensus**
   - Claims supported by multiple sources
   - Claims from single high-authority source

3. **Detect and resolve conflicts**
   - Where do sources disagree?
   - Can they be reconciled?
   - If not, note as disputed

4. **Identify gaps**
   - What is NOT covered?
   - What questions remain?

5. **Synthesize carefully**
   - Don't introduce claims not in sources
   - Confidence bounded by weakest link
   - Note limitations explicitly

## Output Format

You must output a structured JSON object matching the SynthesisOutput schema.
Do not include any text outside the JSON object.
"""
