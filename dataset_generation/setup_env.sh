#!/bin/bash
# Setup script for dataset generation environment
# Handles cache directory setup for shared systems

echo "=========================================="
echo "Dataset Generation Environment Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CACHE_DIR="$SCRIPT_DIR/.cache"

# Create cache directory
echo "Creating local cache directory: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Set HuggingFace environment variables
echo "Setting HuggingFace cache environment variables..."
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_TOKEN_PATH="$CACHE_DIR/token"

echo ""
echo "✓ Cache directory created: $CACHE_DIR"
echo "✓ Environment variables set:"
echo "  - HF_HOME=$HF_HOME"
echo "  - TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  - HF_TOKEN_PATH=$HF_TOKEN_PATH"
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ Warning: OPENAI_API_KEY not set!"
    echo "  Please set it with:"
    echo "    export OPENAI_API_KEY='your-api-key-here'"
    echo ""
else
    echo "✓ OPENAI_API_KEY is set"
    echo ""
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "If you need to login to HuggingFace (for Llama-3.1), run:"
echo "  huggingface-cli login"
echo "or"
echo "  hf auth login"
echo ""
echo "To persist these settings, add to your ~/.bashrc:"
echo "  export HF_HOME=$CACHE_DIR"
echo "  export TRANSFORMERS_CACHE=$CACHE_DIR"
echo ""

