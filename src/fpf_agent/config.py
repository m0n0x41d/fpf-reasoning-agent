"""
Configuration management for FPF Agent.

Loads from environment variables and .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """LLM model configuration"""
    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_key: str = ""
    # Lower temperature for rigorous reasoning (high variance hurts FPF)
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass
class Config:
    """Application configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    db_path: Path = field(default_factory=lambda: Path("data/chats.db"))
    # FPF specification file path - required for FPF reasoning
    fpf_spec_path: Path | None = None
    max_file_size_mb: int = 10
    max_context_chars: int = 50000
    server_host: str = "127.0.0.1"
    server_port: int = 7860


def load_config() -> Config:
    """Load configuration from environment."""
    load_dotenv()

    model = ModelConfig(
        provider=os.getenv("FPF_MODEL_PROVIDER", "openai"),
        model_name=os.getenv("FPF_MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("FPF_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        temperature=float(os.getenv("FPF_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("FPF_MAX_TOKENS", "4096")),
    )

    # Resolve FPF spec path - check env, then default locations
    fpf_spec_path = _resolve_fpf_spec_path()

    return Config(
        model=model,
        db_path=Path(os.getenv("FPF_DB_PATH", "data/chats.db")),
        fpf_spec_path=fpf_spec_path,
        max_file_size_mb=int(os.getenv("FPF_MAX_FILE_SIZE_MB", "10")),
        max_context_chars=int(os.getenv("FPF_MAX_CONTEXT_CHARS", "50000")),
        server_host=os.getenv("FPF_SERVER_HOST", "127.0.0.1"),
        server_port=int(os.getenv("FPF_SERVER_PORT", "7860")),
    )


def _resolve_fpf_spec_path() -> Path | None:
    """
    Resolve FPF specification path from environment or default locations.

    Search order:
    1. FPF_SPEC_PATH environment variable
    2. ./First Principles Framework*.md (current directory)
    3. ../First Principles Framework*.md (parent directory)
    """
    # Check environment variable first
    env_path = os.getenv("FPF_SPEC_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        print(f"[FPF Config] Warning: FPF_SPEC_PATH={env_path} does not exist")

    # Search default locations
    default_patterns = [
        "First Principles Framework*.md",
        "FPF*.md",
    ]
    search_dirs = [Path.cwd(), Path.cwd().parent]

    for search_dir in search_dirs:
        for pattern in default_patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                # Take the first match (usually there's only one)
                return matches[0]

    return None


def validate_fpf_config(config: Config) -> list[str]:
    """
    Validate FPF-specific configuration.
    Returns list of warning messages (empty if all OK).
    """
    warnings = []

    if config.fpf_spec_path is None:
        warnings.append(
            "FPF specification not found. Set FPF_SPEC_PATH or place "
            "'First Principles Framework*.md' in the project directory. "
            "Agent will work but cannot look up FPF sections."
        )
    elif not config.fpf_spec_path.exists():
        warnings.append(f"FPF specification path does not exist: {config.fpf_spec_path}")

    if config.model.temperature > 0.5:
        warnings.append(
            f"Temperature {config.model.temperature} is high for rigorous reasoning. "
            "Consider lowering to 0.2-0.3 for FPF work."
        )

    return warnings


# Supported models that have structured output capability
# Note: This is not exhaustive - most modern LLMs support structured output
SUPPORTED_MODELS: dict[str, list[str]] = {
    "openai": [
        # GPT-5 series (2025)
        "gpt-5",
        "gpt-5-mini",
        # GPT-4.1 series
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # GPT-4o series
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        # o-series reasoning models
        "o3",
        "o3-mini",
        "o4-mini",
        "o1",
        "o1-mini",
        "o1-preview",
        # GPT-4 legacy
        "gpt-4-turbo",
        "gpt-4",
    ],
    "anthropic": [
        # Claude 4 series
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        # Claude 3 series
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "google": [
        # Gemini 2.5 series (2025, full JSON Schema support)
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-04-17",
        # Gemini 2.0 series
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        # Gemini 1.5 series
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "xai": [
        # Grok 3 series (2025)
        "grok-3",
        "grok-3-mini",
        "grok-3-fast",
        # Grok 2 series
        "grok-2",
        "grok-2-mini",
        "grok-2-vision",
    ],
}


def get_supported_models() -> list[str]:
    """Get flat list of all supported models with provider prefix."""
    models = []
    for provider, model_list in SUPPORTED_MODELS.items():
        for model in model_list:
            models.append(f"{provider}:{model}")
    return models


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Parse 'provider:model' string into (provider, model)."""
    if ":" in model_str:
        provider, model = model_str.split(":", 1)
        return provider, model
    return "openai", model_str


def validate_model(provider: str, model: str) -> bool:
    """Check if model is in supported list."""
    if provider not in SUPPORTED_MODELS:
        return False
    return model in SUPPORTED_MODELS[provider]
