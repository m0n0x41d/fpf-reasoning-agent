"""
FPF Reasoning Agent - Main entry point with Gradio UI.
"""

import asyncio
from datetime import datetime
from pathlib import Path

import gradio as gr

from .config import (
    load_config,
    get_supported_models,
    parse_model_string,
    validate_model,
    Config,
    ModelConfig,
)
from .core.schemas import ConversationContext, FileAttachment
from .shell.storage import ChatStorage
from .shell.llm import FPFReasoningAgent
from .shell.files import extract_from_bytes, validate_file


class AppState:
    """Application state container."""

    def __init__(self, config: Config):
        self.config = config
        self.storage = ChatStorage(config.db_path)
        self.agent: FPFReasoningAgent | None = None
        self.current_chat_id: str | None = None

    async def initialize(self):
        """Initialize storage and agent."""
        await self.storage.initialize()
        if self.config.model.api_key:
            self.agent = FPFReasoningAgent(self.config.model)

    def update_model(self, model_str: str, api_key: str) -> str:
        """Update model configuration."""
        if not api_key:
            return "API key required"

        provider, model = parse_model_string(model_str)
        if not validate_model(provider, model):
            return f"Invalid model: {model_str}"

        self.config.model = ModelConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )
        self.agent = FPFReasoningAgent(self.config.model)
        return f"Model updated: {model_str}"


# Global state
state: AppState | None = None


def get_state() -> AppState:
    """Get or create app state."""
    global state
    if state is None:
        config = load_config()
        state = AppState(config)
        asyncio.get_event_loop().run_until_complete(state.initialize())
    return state


async def process_message(
    message: str,
    history: list[list[str]],
    files: list[str] | None,
) -> tuple[list[list[str]], str]:
    """
    Process user message through FPF reasoning pipeline.

    Returns updated history and empty string (to clear input).
    """
    app = get_state()

    if not app.agent:
        history.append([message, "Please configure your API key in settings first."])
        return history, ""

    # Extract file attachments
    attachments: list[FileAttachment] = []
    if files:
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                content = path.read_bytes()
                is_valid, error = validate_file(
                    path.name,
                    len(content),
                    app.config.max_file_size_mb,
                )
                if is_valid:
                    attachment = extract_from_bytes(
                        path.name,
                        content,
                        app.config.max_context_chars,
                    )
                    attachments.append(attachment)

    # Build conversation history
    conv_history = []
    for user_msg, assistant_msg in history:
        if user_msg:
            conv_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conv_history.append({"role": "assistant", "content": assistant_msg})

    # Create context
    context = ConversationContext(
        chat_id=app.current_chat_id or "temp",
        user_query=message,
        conversation_history=conv_history,
        attached_files=attachments,
        created_at=datetime.now(),
    )

    # Run reasoning
    try:
        response = await app.agent.run_reasoning_loop(context)

        # Format response with reasoning trace
        response_text = response.response
        if response.reasoning_trace:
            trace = "\n".join(f"  {t}" for t in response.reasoning_trace)
            response_text += f"\n\n<details><summary>Reasoning trace ({response.confidence} confidence)</summary>\n\n{trace}\n</details>"

        history.append([message, response_text])

        # Save to storage if we have a chat
        if app.current_chat_id:
            await app.storage.add_message(app.current_chat_id, "user", message)
            await app.storage.add_message(
                app.current_chat_id,
                "assistant",
                response.response,
                {"confidence": response.confidence, "trace": response.reasoning_trace},
            )

    except Exception as e:
        error_msg = f"Error during reasoning: {str(e)}"
        history.append([message, error_msg])

    return history, ""


async def new_chat() -> tuple[list[list[str]], str]:
    """Create a new chat session."""
    app = get_state()
    chat = await app.storage.create_chat()
    app.current_chat_id = chat.id
    return [], f"New chat: {chat.id[:8]}"


async def load_chat_history(chat_id: str) -> list[list[str]]:
    """Load chat history from storage."""
    app = get_state()
    messages = await app.storage.get_message_history(chat_id)
    app.current_chat_id = chat_id

    history = []
    user_msg = None
    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg:
            history.append([user_msg, msg["content"]])
            user_msg = None

    return history


async def get_chat_list() -> list[tuple[str, str]]:
    """Get list of chats for dropdown."""
    app = get_state()
    chats = await app.storage.list_chats()
    return [(f"{c.title} ({c.id[:8]})", c.id) for c in chats]


def update_settings(model: str, api_key: str) -> str:
    """Update model settings."""
    app = get_state()
    return app.update_model(model, api_key)


def create_ui() -> gr.Blocks:
    """Create Gradio UI."""
    supported_models = get_supported_models()

    with gr.Blocks(
        title="FPF Reasoning Agent",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # FPF Reasoning Agent

            Schema-Guided Reasoning based on First Principles Framework.
            """
        )

        with gr.Tabs():
            # Chat Tab
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_dropdown = gr.Dropdown(
                            label="Select Chat",
                            choices=[],
                            interactive=True,
                        )
                        new_chat_btn = gr.Button("New Chat", variant="primary")
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=1,
                        )

                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            show_copy_button=True,
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Message",
                                placeholder="Ask anything...",
                                scale=4,
                                lines=2,
                            )
                            file_upload = gr.File(
                                label="Attach files",
                                file_count="multiple",
                                file_types=[".pdf", ".md", ".txt", ".py", ".js", ".json"],
                                scale=1,
                            )

                        send_btn = gr.Button("Send", variant="primary")

            # Settings Tab
            with gr.Tab("Settings"):
                gr.Markdown("### Model Configuration")

                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=supported_models,
                    value=supported_models[0] if supported_models else None,
                )

                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key...",
                )

                save_settings_btn = gr.Button("Save Settings", variant="primary")
                settings_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

                gr.Markdown(
                    """
                    ### Supported Models

                    **OpenAI:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
                    **Anthropic:** claude-sonnet-4, claude-3-5-sonnet, claude-3-5-haiku
                    **Google:** gemini-1.5-pro, gemini-1.5-flash

                    Note: Only models with structured output support are available.
                    """
                )

            # About Tab
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## FPF Reasoning Agent

                    This agent uses **Schema-Guided Reasoning (SGR)** based on the
                    **First Principles Framework (FPF)**.

                    ### Key Concepts

                    - **Strict Distinction (A.7)**: Always identify what you're reasoning about
                    - **Bounded Context (A.1.1)**: Meaning is local to contexts
                    - **F-G-R Trust Model (B.3)**: Formality, Scope, Reliability
                    - **Adaptive Planning**: Plan multiple steps, execute one, replan

                    ### SGR Patterns Used

                    - **Cascade**: Sequential reasoning (broad → specific)
                    - **Routing**: Discriminated unions for action selection
                    - **Adaptive Planning**: Generate plan each step, use first action only

                    ### Links

                    - [FPF Framework](https://github.com/m0n0x41d/FPF)
                    - [SGR by Rinat Abdullin](https://abdullin.com/schema-guided-reasoning/)
                    """
                )

        # Event handlers
        send_btn.click(
            fn=process_message,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input],
        )

        msg_input.submit(
            fn=process_message,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input],
        )

        new_chat_btn.click(
            fn=new_chat,
            outputs=[chatbot, status_text],
        )

        save_settings_btn.click(
            fn=update_settings,
            inputs=[model_dropdown, api_key_input],
            outputs=[settings_status],
        )

        # Load chat list on app start
        app.load(
            fn=get_chat_list,
            outputs=[chat_dropdown],
        )

    return app


def main():
    """Main entry point."""
    config = load_config()

    # Initialize state
    global state
    state = AppState(config)
    asyncio.get_event_loop().run_until_complete(state.initialize())

    # Create and launch UI
    app = create_ui()
    app.launch(
        server_name=config.server_host,
        server_port=config.server_port,
        share=False,
    )


if __name__ == "__main__":
    main()
