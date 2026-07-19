#!/bin/bash
# Kill Gradio app if running in background

echo "Checking for running Gradio processes..."

# Method 1: Kill by process name
if pkill -f "gradio_app" 2>/dev/null; then
    echo "✓ Killed Gradio process(es) by name"
else
    echo "⚠ No Gradio processes found by name"
fi

# Method 2: Also kill by port 7860 if still running
if command -v lsof &> /dev/null; then
    PID=$(lsof -i :7860 -t 2>/dev/null)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
        echo "✓ Killed process on port 7860 (PID: $PID)"
    fi
fi

echo "Done!"
