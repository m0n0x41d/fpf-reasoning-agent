#!/usr/bin/env bash
#
# FPF Reasoning Agent - Bootstrap and Run Script
#
# Usage:
#   ./run.sh              # Install deps and run
#   ./run.sh --install    # Only install dependencies
#   ./run.sh --help       # Show help
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
FPF Reasoning Agent

Usage:
    ./run.sh              Install dependencies and run the agent
    ./run.sh --install    Only install dependencies
    ./run.sh --help       Show this help message

Environment Variables:
    FPF_API_KEY           API key for the LLM provider (or set OPENAI_API_KEY)
    FPF_MODEL_PROVIDER    Model provider: openai, anthropic, google (default: openai)
    FPF_MODEL_NAME        Model name (default: gpt-4o)
    FPF_SERVER_PORT       Server port (default: 7860)

Example:
    export OPENAI_API_KEY="sk-..."
    ./run.sh

Or create a .env file:
    echo "OPENAI_API_KEY=sk-..." > .env
    ./run.sh

EOF
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH for this session
        export PATH="$HOME/.local/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            log_error "Failed to install uv. Please install manually:"
            log_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        log_info "uv installed successfully"
    else
        log_info "uv found: $(uv --version)"
    fi
}

install_deps() {
    log_info "Syncing dependencies..."
    uv sync
    log_info "Dependencies installed"
}

check_api_key() {
    if [[ -z "${FPF_API_KEY}" && -z "${OPENAI_API_KEY}" && -z "${ANTHROPIC_API_KEY}" ]]; then
        if [[ -f ".env" ]]; then
            log_info "Loading .env file..."
            set -a
            source .env
            set +a
        fi
    fi

    if [[ -z "${FPF_API_KEY}" && -z "${OPENAI_API_KEY}" && -z "${ANTHROPIC_API_KEY}" ]]; then
        log_warn "No API key found. You can:"
        log_warn "  1. Set environment variable: export OPENAI_API_KEY='sk-...'"
        log_warn "  2. Create .env file: echo 'OPENAI_API_KEY=sk-...' > .env"
        log_warn "  3. Configure in the UI settings"
        echo ""
    fi
}

run_agent() {
    log_info "Starting FPF Reasoning Agent..."
    log_info "Open http://127.0.0.1:${FPF_SERVER_PORT:-7860} in your browser"
    echo ""

    uv run python -m fpf_agent.main
}

# Main
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --install)
        check_uv
        install_deps
        log_info "Installation complete. Run './run.sh' to start the agent."
        exit 0
        ;;
    *)
        check_uv
        install_deps
        check_api_key
        run_agent
        ;;
esac
