# Runpod QwenAI

A simple Qwen AI application running on RunPod.

## Template
Use the `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` template. RTX PRO 6000 GPU with 96GB VRAM or higher is recommended. Lower end GPUs will fail to run the model. 

## Quick Start

**First time:**
```bash
git clone https://github.com/Imranazeb/Runpod-QwenAI
cd /workspace/Runpod-QwenAI
chmod +x start_runpod.sh && bash start_runpod.sh
```

**Next time:**
```bash
cd /workspace/Runpod-QwenAI
bash start_runpod.sh
```