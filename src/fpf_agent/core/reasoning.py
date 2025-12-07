"""
Pure reasoning functions for FPF-SGR pipeline.

All functions are pure: no side effects, deterministic output.
"""

from .schemas import (
    ConversationContext,
    FileAttachment,
    FPFResponse,
)


def build_system_prompt(context: ConversationContext) -> str:
    """Build system prompt for FPF reasoning agent."""
    file_context = ""
    if context.attached_files:
        file_context = "\n\n## Attached Files\n"
        for f in context.attached_files:
            file_context += f"\n### {f.filename} ({f.content_type}, {f.char_count} chars)\n"
            file_context += f"```\n{f.extracted_text[:5000]}\n```\n"
            if f.char_count > 5000:
                file_context += f"[Truncated, {f.char_count - 5000} more chars]\n"

    history_context = ""
    if context.conversation_history:
        history_context = "\n\n## Conversation History\n"
        for msg in context.conversation_history[-10:]:  # Last 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]
            history_context += f"\n**{role}**: {content}\n"

    return f"""You are an FPF (First Principles Framework) reasoning agent.

## Your Reasoning Discipline

You follow Schema-Guided Reasoning (SGR) based on FPF principles:

### 1. Strict Distinction (A.7)
ALWAYS first identify what you're reasoning about:
- System: Physical/operational entity
- Episteme: Knowledge artifact
- Role: Contextual responsibility
- Method: Way of doing something
- Work: Record of execution
- Service: External promise/commitment
- Claim: Assertion with evidence
- Question: Query to resolve

### 2. Bounded Context (A.1.1)
Meaning is LOCAL. Always identify the context where terms have specific meaning.
Cross-context reasoning requires explicit bridges.

### 3. Temporal Stance (A.4)
Distinguish design-time (specifications, models) from run-time (execution, actuals).

### 4. F-G-R Trust Model (B.3)
For any claim, assess:
- F (Formality): 0-9, how rigorous is the expression?
- G (Scope): Where does this apply?
- R (Reliability): 0-1, how well-supported by evidence?

### 5. Reasoning Cycle (B.5)
- Abduction: Generate hypotheses for anomalies
- Deduction: Derive consequences from hypotheses
- Induction: Test predictions against evidence

### 6. Adaptive Planning
Plan 1-5 steps ahead, but only execute the first. Replan after each step.

## Response Guidelines
- Be concise and direct
- Ground claims in evidence when possible
- Acknowledge uncertainty explicitly
- Use structured reasoning visible to the user
{file_context}
{history_context}
"""


def build_user_prompt(query: str) -> str:
    """Build user prompt from query."""
    return f"""## User Query

{query}

## Your Task

Reason through this query using FPF principles. Structure your thinking, then provide a clear response."""


def format_reasoning_trace(steps: list[dict]) -> list[str]:
    """Format reasoning steps into human-readable trace."""
    trace = []
    for i, step in enumerate(steps, 1):
        object_of_talk = step.get("object_of_talk", {})
        category = object_of_talk.get("category", "Unknown")
        context = step.get("context", {}).get("context_id", "Unknown")
        phase = step.get("reasoning_phase", {}).get("phase", "Unknown")
        action = step.get("next_action", {})
        action_type = action.get("action_type", "unknown")

        trace.append(
            f"Step {i}: [{category}] in [{context}] | Phase: {phase} | Action: {action_type}"
        )
    return trace


def build_final_response(
    response_text: str,
    reasoning_steps: list[dict],
    confidence: str,
    sources: list[str],
) -> FPFResponse:
    """Build final FPF response from reasoning results."""
    return FPFResponse(
        response=response_text,
        reasoning_trace=format_reasoning_trace(reasoning_steps),
        confidence=confidence,
        sources_used=sources,
    )


def extract_sources_from_context(context: ConversationContext) -> list[str]:
    """Extract source references from context."""
    sources = []
    for f in context.attached_files:
        sources.append(f.filename)
    return sources


def estimate_complexity(query: str, file_count: int) -> str:
    """Estimate query complexity for reasoning budget."""
    word_count = len(query.split())

    if word_count < 20 and file_count == 0:
        return "simple"
    elif word_count < 100 and file_count <= 2:
        return "moderate"
    else:
        return "complex"


def should_request_clarification(query: str) -> tuple[bool, str | None]:
    """
    Check if query needs clarification.
    Returns (needs_clarification, reason).
    """
    query_lower = query.lower().strip()

    if len(query_lower) < 5:
        return True, "Query is too short to understand intent"

    ambiguous_patterns = [
        "this",
        "that",
        "it",
        "the thing",
        "you know",
    ]

    for pattern in ambiguous_patterns:
        if query_lower == pattern or query_lower.startswith(f"{pattern} "):
            return True, f"Query contains ambiguous reference: '{pattern}'"

    return False, None
