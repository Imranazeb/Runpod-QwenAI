#!/usr/bin/env python3
"""Quick diagnostic script for RunPod GPU and model performance."""

import torch

print("=" * 60)
print("RUNPOD DIAGNOSTICS")
print("=" * 60)

# Check PyTorch and CUDA
print(f"\n✓ PyTorch Version: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ CUDA Device Count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n--- GPU {i}: {torch.cuda.get_device_name(i)} ---")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
        print(
            f"  Free: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.1f} GB"
        )
else:
    print("⚠️  WARNING: CUDA not available! Model will run on CPU (very slow)")

# Check if bitsandbytes is available for quantization
try:
    import bitsandbytes

    print(f"\n✓ bitsandbytes available: {bitsandbytes.__version__}")
except ImportError:
    print("\n⚠️  WARNING: bitsandbytes not installed - quantization unavailable")
    print("   Install with: pip install bitsandbytes")

# Check transformers version
try:
    import transformers

    print(f"✓ transformers version: {transformers.__version__}")
except ImportError:
    print("⚠️  transformers not installed")

# Check accelerate
try:
    import accelerate

    print(f"✓ accelerate version: {accelerate.__version__}")
except ImportError:
    print("⚠️  accelerate not installed")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nYour GPU has {total_mem:.1f} GB VRAM")

    if total_mem >= 90:
        print("✓ Use: python runpod.py --load-8bit")
        print("  Expected: ~72GB usage, 15-30 tokens/sec")
    elif total_mem >= 40:
        print("✓ Use: python runpod.py --load-4bit")
        print("  Expected: ~36GB usage, 10-20 tokens/sec")
    else:
        print("⚠️  GPU too small for 72B model")
        print("  Consider using Qwen2.5-14B or smaller")
else:
    print("⚠️  Enable GPU access in RunPod!")
    print("  1. Check your pod configuration")
    print("  2. Verify CUDA is installed: nvidia-smi")

print("\n" + "=" * 60)
