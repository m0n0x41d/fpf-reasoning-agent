"""
FPF-SGR Schemas: Schema-Guided Reasoning for First Principles Framework.

Following Rinat Abdullin's SGR patterns:
- Cascade: Sequential reasoning steps (broad → specific)
- Routing: Discriminated unions for branching
- Cycle: Lists with min/max for repeated reasoning
- Adaptive Planning: Plan N steps, execute 1, replan
"""

from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# CORE FPF TYPES (Minimal Set for Reasoning)
# =============================================================================

class ObjectOfTalk(BaseModel):
    """
    A.7 Strict Distinction: What are we reasoning about?
    CASCADE STEP 1: Always identify this first.
    """
    category: Literal[
        "System",      # Physical/operational holon
        "Episteme",    # Knowledge artifact
        "Role",        # Contextual responsibility
        "Method",      # Way of doing
        "Work",        # Execution record
        "Service",     # External promise
        "Claim",       # Assertion with evidence
        "Question"     # User query to resolve
    ]
    description: str = Field(description="Brief description of the subject")


class BoundedContext(BaseModel):
    """
    A.1.1: Semantic frame where meaning is local.
    CASCADE STEP 2: Always identify context.
    """
    context_id: str = Field(description="Unique context identifier")
    description: str = Field(description="What domain/scope this context covers")
    key_terms: list[str] = Field(
        default_factory=list,
        description="Important terms with local meaning in this context"
    )


class TemporalStance(BaseModel):
    """
    A.4: Design-time vs Run-time distinction.
    CASCADE STEP 3: Identify temporal scope.
    """
    scope: Literal["design_time", "run_time"]
    rationale: str = Field(description="Why this temporal scope applies")


# =============================================================================
# F-G-R TRUST MODEL
# =============================================================================

class FGRAssessment(BaseModel):
    """
    B.3: Trust as computed tuple (Formality, Scope, Reliability).
    Trust is computed from evidence, not intuition.
    """
    formality: Annotated[int, Field(ge=0, le=9, description="F0=vague prose, F9=machine-verified")]
    scope: str = Field(description="Where this claim/knowledge applies")
    reliability: Annotated[float, Field(ge=0, le=1, description="Evidence-backed confidence")]
    evidence_summary: str = Field(description="Brief summary of supporting evidence")
    assurance_level: Literal["L0_unsubstantiated", "L1_partial", "L2_assured"]


# =============================================================================
# REASONING PHASES (B.5)
# =============================================================================

class ReasoningPhase(BaseModel):
    """B.5: Current phase in Abduction-Deduction-Induction cycle"""
    phase: Literal["Abduction", "Deduction", "Induction"]
    description: str = Field(description="What this phase accomplishes")


class ArtifactState(BaseModel):
    """B.5.1: Explore → Shape → Evidence → Operate lifecycle"""
    state: Literal["Exploration", "Shaping", "Evidence", "Operation"]
    rationale: str = Field(description="Why artifact is in this state")


# =============================================================================
# SGR ROUTING: ACTION TYPES
# =============================================================================

class AnalyzeQuery(BaseModel):
    """Break down user query into FPF terms"""
    action_type: Literal["analyze_query"] = "analyze_query"
    original_query: str
    identified_objects: list[str] = Field(description="Objects of talk identified")
    identified_contexts: list[str] = Field(description="Contexts referenced")
    complexity: Literal["simple", "moderate", "complex"]


class RetrieveKnowledge(BaseModel):
    """Retrieve relevant knowledge from context/files"""
    action_type: Literal["retrieve_knowledge"] = "retrieve_knowledge"
    query: str = Field(description="What knowledge to retrieve")
    sources: list[str] = Field(description="Where to look (files, context, etc.)")


class GenerateHypothesis(BaseModel):
    """B.5.2: Abductive hypothesis generation"""
    action_type: Literal["generate_hypothesis"] = "generate_hypothesis"
    anomaly: str = Field(description="What triggered hypothesis generation")
    hypothesis: str = Field(description="Proposed explanation/solution")
    plausibility_rationale: str = Field(description="Why this hypothesis is plausible")


class DeduceConsequences(BaseModel):
    """B.5: Derive logical consequences from hypothesis"""
    action_type: Literal["deduce_consequences"] = "deduce_consequences"
    hypothesis: str
    consequences: Annotated[list[str], Field(min_length=1, max_length=5)]
    testable_predictions: list[str]


class SynthesizeResponse(BaseModel):
    """Synthesize final response from reasoning"""
    action_type: Literal["synthesize_response"] = "synthesize_response"
    key_findings: list[str]
    confidence: Literal["high", "medium", "low"]
    response_draft: str


class RequestClarification(BaseModel):
    """Ask user for clarification when needed"""
    action_type: Literal["request_clarification"] = "request_clarification"
    ambiguity: str = Field(description="What is unclear")
    options: list[str] = Field(description="Possible interpretations")
    question: str = Field(description="Question to ask user")


class ReadFPFSection(BaseModel):
    """Read a specific section from the FPF specification.

    Use this when you need detailed guidance on a specific FPF pattern.
    Pattern IDs follow the format: A.1, A.1.1, B.3.2, C.17, etc.
    You can also search by keyword if unsure of exact ID.
    """
    action_type: Literal["read_fpf_section"] = "read_fpf_section"
    pattern_id: str = Field(
        description="FPF pattern ID (e.g., 'A.1', 'B.3', 'C.17') or keyword to search"
    )
    reason: str = Field(
        description="Why this section is needed for current reasoning"
    )


# Union of all action types for routing
ActionType = (
    AnalyzeQuery
    | RetrieveKnowledge
    | GenerateHypothesis
    | DeduceConsequences
    | SynthesizeResponse
    | RequestClarification
    | ReadFPFSection
)


# =============================================================================
# MAIN SGR REASONING STEP
# =============================================================================

class FPFReasoningStep(BaseModel):
    """
    Main SGR schema for FPF-guided reasoning.

    Follows Cascade pattern:
    1. Strict Distinction (what are we talking about?)
    2. Context (where does meaning live?)
    3. Temporal Stance (design-time or run-time?)
    4. Current Understanding
    5. Gaps/Needs
    6. Plan (1-5 steps)
    7. Next Action (routing)
    """

    # === CASCADE STEP 1: STRICT DISTINCTION ===
    object_of_talk: ObjectOfTalk

    # === CASCADE STEP 2: BOUNDED CONTEXT ===
    context: BoundedContext

    # === CASCADE STEP 3: TEMPORAL STANCE ===
    temporal_stance: TemporalStance

    # === CASCADE STEP 4: CURRENT UNDERSTANDING ===
    current_understanding: str = Field(
        description="What we currently know/understand about the query"
    )
    reasoning_phase: ReasoningPhase
    artifact_state: ArtifactState

    # === CASCADE STEP 5: F-G-R ASSESSMENT (if applicable) ===
    trust_assessment: FGRAssessment | None = Field(
        default=None,
        description="Trust assessment of current knowledge"
    )

    # === CASCADE STEP 6: GAPS AND NEEDS ===
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        description="What we don't know yet"
    )

    # === CASCADE STEP 7: PLAN (Adaptive: plan N, execute 1) ===
    plan: Annotated[list[str], Field(min_length=1, max_length=5)] = Field(
        description="Planned steps (only first will be executed)"
    )

    # === CASCADE STEP 8: NEXT ACTION (Routing) ===
    next_action: ActionType = Field(
        description="Immediate next action to take"
    )

    # === COMPLETION STATUS ===
    is_complete: bool = Field(
        default=False,
        description="Whether reasoning is complete"
    )
    completion_confidence: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None,
        description="Confidence in completion (if complete)"
    )


# =============================================================================
# CONVERSATION CONTEXT
# =============================================================================

class FileAttachment(BaseModel):
    """Attached file with extracted content"""
    filename: str
    content_type: Literal["pdf", "markdown", "text", "code"]
    extracted_text: str
    char_count: int


class ConversationContext(BaseModel):
    """Full context for a reasoning session"""
    chat_id: str
    user_query: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    attached_files: list[FileAttachment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# FINAL RESPONSE
# =============================================================================

class FPFResponse(BaseModel):
    """Final response after reasoning completes"""
    response: str = Field(description="The response to the user")
    reasoning_trace: list[str] = Field(
        default_factory=list,
        description="Key reasoning steps taken"
    )
    confidence: Literal["high", "medium", "low"]
    sources_used: list[str] = Field(
        default_factory=list,
        description="Sources referenced in reasoning"
    )
