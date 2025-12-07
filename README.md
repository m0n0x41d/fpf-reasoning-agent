# FPF Reasoning Agent

[Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) agent based on the First Principles Framework created by Anatoly Levenchuk.
Current implementation is trying to follow version of FPF from September of 2025 - [Ref. Commit](https://github.com/m0n0x41d/FPF/commit/904119a8d33d7678d3016b90ad9ab1e793482be0)

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the agent
./run.sh
```

Open <http://127.0.0.1:7860> in your browser.

## Features

- **FPF-based reasoning**: Structured thinking using First Principles Framework
- **Schema-Guided Reasoning (SGR)**: Follows Rinat Abdullin's SGR patterns
- **On-demand FPF spec lookup**: Agent can read specific FPF sections during reasoning
- **Multiple LLM providers**: OpenAI, Anthropic, Google, xAI
- **File attachments**: PDF, Markdown, text, code files
- **Chat history**: Persistent SQLite storage
- **Configurable in UI**: Change models and API keys without restart

## Architecture

```
fpf_agent/
├── src/fpf_agent/
│   ├── core/                  # Pure functions (no side effects)
│   │   ├── schemas.py         # FPF-SGR Pydantic schemas
│   │   ├── fpf_schemas.py     # Extended FPF type coverage
│   │   ├── fpf_index.py       # FPF spec parser (167 sections)
│   │   ├── fpf_base_prompt.py # Static 10K base prompt
│   │   └── reasoning.py       # Reasoning logic
│   ├── shell/                 # Side effects (I/O, API calls)
│   │   ├── storage.py         # SQLite storage
│   │   ├── llm.py             # Pydantic-AI integration
│   │   └── files.py           # File extraction
│   ├── config.py              # Configuration
│   └── main.py                # Gradio UI
├── run.sh                     # Bootstrap script
└── pyproject.toml             # Dependencies
```

## SGR Patterns Used

### Cascade

Sequential reasoning from broad to specific:

1. Object of Talk (what are we reasoning about?)
2. Bounded Context (where does meaning live?)
3. Temporal Stance (design-time or run-time?)
4. Current Understanding
5. Knowledge Gaps
6. Plan
7. Next Action

### Routing

Discriminated unions for action selection:

- `analyze_query` — Break down user input
- `retrieve_knowledge` — Get info from files/context
- `read_fpf_section` — Consult FPF spec for guidance
- `generate_hypothesis` — Propose explanations (Abduction)
- `deduce_consequences` — Derive testable predictions
- `synthesize_response` — Formulate final answer
- `request_clarification` — Ask user for clarity

### Adaptive Planning

Plan 1-5 steps ahead, execute only the first, then replan with accumulated context.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FPF_API_KEY` | LLM API key | - |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | - |
| `XAI_API_KEY` | xAI API key | - |
| `FPF_MODEL_PROVIDER` | Provider: openai, anthropic, google, xai | openai |
| `FPF_MODEL_NAME` | Model name | gpt-4o |
| `FPF_SERVER_PORT` | Server port | 7860 |
| `FPF_DB_PATH` | SQLite database path | data/chats.db |

### .env File

```bash
OPENAI_API_KEY=sk-...
FPF_MODEL_PROVIDER=openai
FPF_MODEL_NAME=gpt-4o
FPF_SERVER_PORT=7860
```

## Supported Models

Models with structured output capability:

### OpenAI

- **GPT-5 series**: `gpt-5`, `gpt-5-mini`
- **GPT-4.1 series**: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- **GPT-4o series**: `gpt-4o`, `gpt-4o-mini`
- **o-series reasoning**: `o3`, `o3-mini`, `o4-mini`, `o1`, `o1-mini`
- **Legacy**: `gpt-4-turbo`, `gpt-4`

### Anthropic

- **Claude 4**: `claude-sonnet-4-20250514`, `claude-opus-4-20250514`
- **Claude 3.5**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- **Claude 3**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`

### Google

- **Gemini 2.5**: `gemini-2.5-pro`, `gemini-2.5-flash` (full JSON Schema support)
- **Gemini 2.0**: `gemini-2.0-flash`
- **Gemini 1.5**: `gemini-1.5-pro`, `gemini-1.5-flash`

### xAI

- **Grok 3**: `grok-3`, `grok-3-mini`, `grok-3-fast`
- **Grok 2**: `grok-2`, `grok-2-mini`

## Development

```bash
# Install dependencies only
./run.sh --install

# Run with uv directly
uv run python -m fpf_agent.main
```

## FPF Key Concepts

The agent is guided by these core FPF principles:

| Concept | Pattern | Description |
|---------|---------|-------------|
| **Strict Distinction** | A.7 | Always identify what you're reasoning about |
| **Bounded Context** | A.1.1 | Meaning is local to contexts |
| **Temporal Duality** | A.4 | Design-time vs run-time separation |
| **F-G-R Trust Model** | B.3 | Formality, Scope, Reliability tuple |
| **ADI Reasoning Cycle** | B.5 | Abduction → Deduction → Induction |
| **Artifact Lifecycle** | B.5.1 | Explore → Shape → Evidence → Operate |

## FPF Spec Access

The agent has access to the full 167-section FPF specification:

- **Static base prompt** (~8.7K chars) with core FPF fundamentals
- **On-demand lookup** via `read_fpf_section` action for detailed patterns
