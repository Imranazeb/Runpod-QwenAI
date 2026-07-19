#!/bin/bash
# Quick start script for Qwen AI on RunPod RTX PRO 6000

set -e

echo "=============================================="
echo "Qwen AI Startup"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+."
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install uv
uv sync

# Run diagnostics
echo ""
echo "🔍 Running diagnostics..."
uv run diagnose.py

# Check GPU memory
echo ""
echo "🔍 Checking GPU memory..."
gpu_vram=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3)) if torch.cuda.is_available() else 0" 2>/dev/null || echo "0")
gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'N/A'" 2>/dev/null || echo "N/A")

echo "GPU: $gpu_name"
echo "VRAM: ${gpu_vram}GB"
echo ""

# Check if GPU has enough memory for 72B model
if [ "$gpu_vram" -lt 40 ]; then
    echo "❌ Error: GPU needs at least 40GB VRAM for Qwen2.5-72B-Instruct (8-bit)"
    echo "   Current VRAM: ${gpu_vram}GB"
    exit 1
fi

echo "✓ GPU memory is sufficient for Qwen2.5-72B-Instruct"
echo ""

# Launch with 72B model
model="Qwen/Qwen2.5-72B-Instruct"
max_memory="70"

echo "🚀 Starting Gradio Web Interface..."
echo "   Model: $model"
echo "   Quantization: 8-bit"
echo ""

uv run gradio_app.py --model "$model" --max-memory "$max_memory" --load-8bit
