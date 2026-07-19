#!/bin/bash
# Quick start script for GPU deployment

set -e

echo "=============================================="
echo "AI Chat Setup Script"
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

# GPU Selection
echo ""
echo "=============================================="
echo "Select your GPU configuration:"
echo "=============================================="
echo ""
echo "1) RTX 4090 (24GB VRAM)"
echo "2) RTX 5090 (32GB VRAM)"
echo "3) High-end GPU (80GB+ VRAM) - e.g., A100, H100, RTX 6000"
echo "4) Custom/Auto-detect"
echo ""
read -p "Enter your choice (1-4): " gpu_choice

# Set default model and memory based on GPU
case $gpu_choice in
    1)
        echo ""
        echo "📊 RTX 4090 detected (24GB VRAM)"
        echo "   Best model: Qwen2.5-14B-Instruct (8-bit)"
        default_model="Qwen/Qwen2.5-14B-Instruct"
        max_memory="22"
        gpu_name="RTX 4090"
        auto_select=true
        ;;
    2)
        echo ""
        echo "📊 RTX 5090 detected (32GB VRAM)"
        echo "   Recommended: Qwen2.5-32B-Instruct or smaller"
        default_model="Qwen/Qwen2.5-32B-Instruct"
        max_memory="30"
        gpu_name="RTX 5090"
        auto_select=false
        ;;
    3)
        echo ""
        echo "📊 High-end GPU (80GB+ VRAM)"
        echo "   Best model: Qwen2.5-72B-Instruct (8-bit)"
        default_model="Qwen/Qwen2.5-72B-Instruct"
        max_memory="90"
        gpu_name="High-end GPU"
        auto_select=true
        ;;
    4)
        echo ""
        echo "� Auto-detecting GPU..."
        # Detect GPU and VRAM

        source /workspace/.venv/bin/activate
        
        gpu_vram=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3)) if torch.cuda.is_available() else 0" 2>/dev/null || echo "0")
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'N/A'" 2>/dev/null || echo "N/A")
        
        echo "   GPU: $gpu_name"
        echo "   VRAM: ${gpu_vram}GB"
        echo ""
        
        # Select best model based on VRAM with 8-bit quantization
        if [ "$gpu_vram" -ge 80 ]; then
            default_model="Qwen/Qwen2.5-72B-Instruct"
            max_memory="90"
            echo "   ✅ Recommended: Qwen2.5-72B-Instruct (8-bit for ${gpu_vram}GB VRAM)"
        elif [ "$gpu_vram" -ge 32 ]; then
            default_model="Qwen/Qwen2.5-32B-Instruct"
            max_memory="30"
            echo "   ✅ Recommended: Qwen2.5-32B-Instruct (8-bit for ${gpu_vram}GB VRAM)"
        elif [ "$gpu_vram" -ge 24 ]; then
            default_model="Qwen/Qwen2.5-14B-Instruct"
            max_memory="22"
            echo "   ✅ Recommended: Qwen2.5-14B-Instruct (8-bit for ${gpu_vram}GB VRAM)"
        elif [ "$gpu_vram" -ge 12 ]; then
            default_model="Qwen/Qwen2.5-7B-Instruct"
            max_memory="10"
            echo "   ✅ Recommended: Qwen2.5-7B-Instruct (8-bit for ${gpu_vram}GB VRAM)"
        else
            default_model="Qwen/Qwen2.5-3B-Instruct"
            max_memory="6"
            echo "   ✅ Recommended: Qwen2.5-3B-Instruct (8-bit for ${gpu_vram}GB VRAM)"
        fi
        
        auto_select=true
        auto_launch=true
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

# Model selection (skip for auto-select GPUs)
if [ "$auto_select" = true ]; then
    selected_model="$default_model"
    echo ""
    echo "✓ Auto-selected: $selected_model (8-bit quantization)"
else
    echo ""
    echo "=============================================="
    echo "Select model (or press Enter for recommended):"
    echo "=============================================="
    echo ""
    echo "Recommended for $gpu_name: $default_model"
    echo ""
    echo "Available models:"
    echo "  1) Qwen2.5-1.5B-Instruct  (Fast, ~3GB VRAM)"
    echo "  2) Qwen2.5-3B-Instruct    (Fast, ~6GB VRAM)"
    echo "  3) Qwen2.5-7B-Instruct    (Balanced, ~14GB VRAM in 8-bit)"
    echo "  4) Qwen2.5-14B-Instruct   (Good quality, ~28GB VRAM in 8-bit)"
    echo "  5) Qwen2.5-32B-Instruct   (High quality, ~64GB VRAM in 8-bit)"
    echo "  6) Qwen2.5-72B-Instruct   (Best quality, ~144GB VRAM in 8-bit)"
    echo "  7) Custom model ID"
    echo ""
    read -p "Enter choice (1-7, or Enter for recommended): " model_choice

    case $model_choice in
        1)
            selected_model="Qwen/Qwen2.5-1.5B-Instruct"
            ;;
        2)
            selected_model="Qwen/Qwen2.5-3B-Instruct"
            ;;
        3)
            selected_model="Qwen/Qwen2.5-7B-Instruct"
            ;;
        4)
            selected_model="Qwen/Qwen2.5-14B-Instruct"
            ;;
        5)
            selected_model="Qwen/Qwen2.5-32B-Instruct"
            ;;
        6)
            selected_model="Qwen/Qwen2.5-72B-Instruct"
            ;;
        7)
            read -p "Enter custom model ID (e.g., meta-llama/Llama-3.1-8B-Instruct): " selected_model
            ;;
        "")
            selected_model="$default_model"
            echo "Using recommended: $selected_model"
            ;;
        *)
            echo "❌ Invalid choice. Using recommended: $default_model"
            selected_model="$default_model"
            ;;
    esac
fi

# Auto-launch for auto-detect mode
if [ "${auto_launch:-false}" = true ]; then
    base_cmd="--model $selected_model"
    if [ "$max_memory" != "auto" ]; then
        base_cmd="$base_cmd --max-memory $max_memory"
    fi
    
    echo ""
    echo "🚀 Launching Gradio Web Interface (8-bit)..."
    echo "   Model: $selected_model"
    echo "   Quantization: 8-bit"
    echo ""
    uv run gradio_app.py $base_cmd --load-8bit
    exit 0
fi

# Interface and quantization selection
echo ""
echo "=============================================="
echo "What would you like to run?"
echo "=============================================="
echo ""
echo "1) Gradio Web Interface (8-bit) - Recommended"
echo "2) Gradio Web Interface (4-bit) - Lower memory"
echo "3) Gradio Web Interface (Full precision) - Highest quality"
echo "4) Command-line Interface (8-bit)"
echo "5) Command-line Interface (4-bit)"
echo ""
read -p "Enter your choice (1-5): " choice

# Build command
base_cmd="--model $selected_model"
if [ "$max_memory" != "auto" ]; then
    base_cmd="$base_cmd --max-memory $max_memory"
fi

case $choice in
    1)
        echo ""
        echo "🚀 Starting Gradio with 8-bit quantization..."
        echo "   Model: $selected_model"
        uv run gradio_app.py $base_cmd --load-8bit
        ;;
    2)
        echo ""
        echo "🚀 Starting Gradio with 4-bit quantization..."
        echo "   Model: $selected_model"
        uv run gradio_app.py $base_cmd --load-4bit
        ;;
    3)
        echo ""
        echo "🚀 Starting Gradio with full precision..."
        echo "   Model: $selected_model"
        uv run gradio_app.py $base_cmd
        ;;
    4)
        echo ""
        echo "🚀 Starting command-line with 8-bit quantization..."
        echo "   Model: $selected_model"
        uv run runpod.py $base_cmd --load-8bit
        ;;
    5)
        echo ""
        echo "🚀 Starting command-line with 4-bit quantization..."
        echo "   Model: $selected_model"
        uv run runpod.py $base_cmd --load-4bit
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac
