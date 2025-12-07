"""
LLM integration using Pydantic-AI.

Shell layer: handles all LLM API calls.
Supports structured output via Pydantic-AI's native capability.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel

from ..config import ModelConfig
from ..core.schemas import (
    FPFReasoningStep,
    FPFResponse,
    ConversationContext,
    ReadFPFSection,
)
from ..core.reasoning import extract_sources_from_context
from ..core.fpf_base_prompt import get_fpf_system_prompt, get_fpf_section_instructions
from ..core.fpf_index import (
    parse_fpf_spec,
    get_section_content,
    search_sections,
    build_section_index_summary,
    FPFSection,
)


# Default FPF spec path (relative to project root)
DEFAULT_FPF_SPEC_PATH = Path(__file__).parent.parent.parent.parent.parent / "First Principles Framework — Core Conceptual Specification (holonic).md"


def create_model(config: ModelConfig):
    """Create Pydantic-AI model from config."""
    if config.provider == "openai":
        return OpenAIModel(
            config.model_name,
            api_key=config.api_key,
        )
    elif config.provider == "anthropic":
        return AnthropicModel(
            config.model_name,
            api_key=config.api_key,
        )
    elif config.provider == "google":
        return GeminiModel(
            config.model_name,
            api_key=config.api_key,
        )
    elif config.provider == "xai":
        # xAI uses OpenAI-compatible API
        return OpenAIModel(
            config.model_name,
            api_key=config.api_key,
            base_url="https://api.x.ai/v1",
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


class FPFReasoningAgent:
    """
    FPF Reasoning Agent using Schema-Guided Reasoning.

    Uses Pydantic-AI for structured output generation.
    Follows adaptive planning: plan N steps, execute 1, replan.

    Features:
    - Static FPF base prompt (~10K chars) always in context
    - On-demand FPF section lookup via read_fpf_section action
    - Full ADI reasoning cycle support
    """

    def __init__(self, config: ModelConfig, fpf_spec_path: Path | None = None):
        self.config = config
        self.model = create_model(config)

        # Load FPF specification index
        self.fpf_spec_path = fpf_spec_path or DEFAULT_FPF_SPEC_PATH
        self.fpf_sections: dict[str, FPFSection] = {}
        self._load_fpf_index()

        # Build the full system prompt
        self.system_prompt = self._build_full_system_prompt()

        # Agent for structured reasoning steps
        self.reasoning_agent = Agent(
            self.model,
            result_type=FPFReasoningStep,
            system_prompt=self.system_prompt,
        )

        # Agent for final response generation
        self.response_agent = Agent(
            self.model,
            result_type=FPFResponse,
            system_prompt=self._get_response_system_prompt(),
        )

    def _load_fpf_index(self) -> None:
        """Load and index the FPF specification."""
        if self.fpf_spec_path.exists():
            self.fpf_sections = parse_fpf_spec(self.fpf_spec_path)
            print(f"[FPF] Loaded {len(self.fpf_sections)} sections from spec")
        else:
            print(f"[FPF] Warning: Spec not found at {self.fpf_spec_path}")

    def _build_full_system_prompt(self) -> str:
        """Build complete system prompt with FPF base + section index."""
        parts = [
            get_fpf_system_prompt(),
            get_fpf_section_instructions(),
        ]

        # Add dynamic section index if available
        if self.fpf_sections:
            parts.append("\n## Available FPF Sections\n")
            parts.append(build_section_index_summary(self.fpf_sections))

        return "\n".join(parts)

    def _get_response_system_prompt(self) -> str:
        return """You are synthesizing a final response after FPF reasoning.

Based on the reasoning trace provided, generate a clear, helpful response.

Your response should:
- Directly address the user's query
- Be grounded in the reasoning performed
- Acknowledge confidence level (based on F-G-R assessment if available)
- Reference sources when applicable
- Be concise but complete

Remember: Trust is computed, not intuited. State your confidence level with rationale."""

    def read_fpf_section(self, pattern_id: str) -> str:
        """
        Read a section from the FPF specification.

        Args:
            pattern_id: Either exact pattern ID (e.g., "A.1", "B.3.2") or keyword

        Returns:
            Section content or search results if multiple matches
        """
        if not self.fpf_sections:
            return "FPF specification not loaded."

        # Try exact match first
        content = get_section_content(
            self.fpf_spec_path,
            self.fpf_sections,
            pattern_id,
            max_chars=12000,  # Limit to ~12K chars per section
        )

        if content:
            return content

        # Try keyword search
        matches = search_sections(self.fpf_sections, pattern_id)
        if matches:
            result_parts = [f"No exact match for '{pattern_id}'. Related sections:"]
            for section in matches[:5]:
                result_parts.append(f"- {section.pattern_id}: {section.title}")
            return "\n".join(result_parts)

        return f"No FPF section found for '{pattern_id}'"

    async def reason_step(
        self,
        context: ConversationContext,
        previous_steps: list[dict] | None = None,
        fpf_context: str | None = None,
    ) -> FPFReasoningStep:
        """
        Execute one reasoning step.

        Follows SGR adaptive planning: generates full plan but only
        the next_action is executed.

        Args:
            context: Current conversation context
            previous_steps: Previously executed reasoning steps
            fpf_context: Additional FPF section content if read_fpf_section was used
        """
        prompt = self._build_step_prompt(context, previous_steps, fpf_context)
        result = await self.reasoning_agent.run(prompt)
        return result.data

    async def generate_response(
        self,
        context: ConversationContext,
        reasoning_steps: list[dict],
    ) -> FPFResponse:
        """Generate final response from reasoning trace."""
        prompt = self._build_response_prompt(context, reasoning_steps)
        result = await self.response_agent.run(prompt)
        return result.data

    async def run_reasoning_loop(
        self,
        context: ConversationContext,
        max_steps: int = 7,
    ) -> FPFResponse:
        """
        Run full reasoning loop until completion or max steps.

        Implements adaptive planning: each step replans from scratch
        with accumulated context.

        Handles special actions:
        - read_fpf_section: Fetches FPF content and continues reasoning
        - request_clarification: Returns clarification request to user
        """
        steps: list[dict] = []
        fpf_context: str | None = None

        for _ in range(max_steps):
            step = await self.reason_step(context, steps, fpf_context)
            steps.append(step.model_dump())

            # Clear FPF context after it's been used
            fpf_context = None

            if step.is_complete:
                break

            # Handle special action types
            action = step.next_action

            if action.action_type == "request_clarification":
                return FPFResponse(
                    response=action.question,
                    reasoning_trace=[
                        f"Identified ambiguity: {action.ambiguity}",
                        f"Options considered: {', '.join(action.options)}",
                    ],
                    confidence="low",
                    sources_used=[],
                )

            if action.action_type == "read_fpf_section":
                # Fetch FPF section and add to next step's context
                fpf_content = self.read_fpf_section(action.pattern_id)
                fpf_context = f"\n## FPF Section: {action.pattern_id}\n{fpf_content}"
                # Don't break - continue reasoning with the new context

        # Generate final response from reasoning trace
        return await self.generate_response(context, steps)

    def _build_step_prompt(
        self,
        context: ConversationContext,
        previous_steps: list[dict] | None,
        fpf_context: str | None = None,
    ) -> str:
        """Build prompt for a reasoning step."""
        parts = [
            f"## User Query\n{context.user_query}",
        ]

        if context.attached_files:
            parts.append("\n## Attached Files")
            for f in context.attached_files:
                content_preview = f.extracted_text[:3000]
                parts.append(f"\n### {f.filename}\n```\n{content_preview}\n```")

        if context.conversation_history:
            parts.append("\n## Conversation History")
            for msg in context.conversation_history[-5:]:
                parts.append(f"\n**{msg['role']}**: {msg['content'][:500]}")

        if previous_steps:
            parts.append("\n## Previous Reasoning Steps")
            for i, step in enumerate(previous_steps, 1):
                obj = step.get("object_of_talk", {}).get("category", "?")
                ctx = step.get("context", {}).get("context_id", "?")
                action = step.get("next_action", {}).get("action_type", "?")
                understanding = step.get("current_understanding", "")[:100]
                parts.append(f"\nStep {i}: [{obj}] in [{ctx}] → {action}")
                if understanding:
                    parts.append(f"  Understanding: {understanding}...")

        if fpf_context:
            parts.append(fpf_context)

        parts.append("\n## Task\nPerform the next reasoning step following the FPF Cascade pattern.")
        parts.append("If you need detailed FPF guidance, use the read_fpf_section action.")

        return "\n".join(parts)

    def _build_response_prompt(
        self,
        context: ConversationContext,
        reasoning_steps: list[dict],
    ) -> str:
        """Build prompt for final response generation."""
        parts = [
            f"## Original Query\n{context.user_query}",
            "\n## Reasoning Trace",
        ]

        for i, step in enumerate(reasoning_steps, 1):
            obj = step.get("object_of_talk", {})
            understanding = step.get("current_understanding", "")
            action = step.get("next_action", {})
            trust = step.get("trust_assessment")
            phase = step.get("reasoning_phase", {})

            parts.append(f"\n### Step {i}")
            parts.append(f"- Object: {obj.get('category', '?')} - {obj.get('description', '')[:100]}")
            parts.append(f"- Phase: {phase.get('phase', '?')}")
            parts.append(f"- Understanding: {understanding[:300]}")
            parts.append(f"- Action: {action.get('action_type', '?')}")

            if trust:
                parts.append(f"- Trust: F{trust.get('formality', '?')}, R={trust.get('reliability', '?')}, {trust.get('assurance_level', '?')}")

        # Collect sources from context and reasoning
        sources = extract_sources_from_context(context)
        fpf_sections_used = []
        for step in reasoning_steps:
            action = step.get("next_action", {})
            if action.get("action_type") == "read_fpf_section":
                fpf_sections_used.append(action.get("pattern_id", "unknown"))

        if sources or fpf_sections_used:
            parts.append("\n## Sources Used")
            if sources:
                parts.append(f"Files: {', '.join(sources)}")
            if fpf_sections_used:
                parts.append(f"FPF Sections: {', '.join(fpf_sections_used)}")

        parts.append("\n## Task\nSynthesize a final response based on the reasoning above.")
        parts.append("Include confidence level with rationale based on F-G-R if applicable.")

        return "\n".join(parts)


async def create_agent(
    config: ModelConfig,
    fpf_spec_path: Path | None = None,
) -> FPFReasoningAgent:
    """Factory function to create reasoning agent."""
    return FPFReasoningAgent(config, fpf_spec_path)
