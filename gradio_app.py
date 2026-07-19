from __future__ import annotations

import argparse
import os
import time
from threading import Thread

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# Set persistent cache directory
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/hf_cache")

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


try:
    from .prompt import DEFAULT_SYSTEM_PROMPT

    if (
        not isinstance(DEFAULT_SYSTEM_PROMPT, str)
        or DEFAULT_SYSTEM_PROMPT.strip() == ""
    ):
        DEFAULT_SYSTEM_PROMPT = (
            "You are a helpful and knowledgeable AI assistant. "
            "Provide clear, accurate, and thoughtful responses to user questions."
        )
except ImportError:
    # Fallback if prompt.py is not available
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful and knowledgeable AI assistant. "
        "Provide clear, accurate, and thoughtful responses to user questions."
    )
finally:
    print(f"Using system prompt: {DEFAULT_SYSTEM_PROMPT}")  # type: ignore


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful, and knowledgeable AI assistant. "
    "Provide clear, accurate, and thoughtful responses to user questions."
)

# Global variables for model and tokenizer
model = None
tokenizer = None
system_prompt = DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradio chat interface for large language models."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("HF_MODEL_ID", DEFAULT_MODEL),
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("HF_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="System prompt to use for the chat template.",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (uses ~50% memory).",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (uses ~25% memory).",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=90,
        help="Maximum GPU memory in GB to use.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio on (default: 7860).",
    )
    return parser.parse_args()


def load_model(
    model_id: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    max_memory_gb: int = 90,
):
    """Load the model and tokenizer."""
    print(f"\n{'=' * 60}")
    print(f"Loading {model_id}...")
    print(f"{'=' * 60}\n")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(
            f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB"
        )
    else:
        print("⚠️  WARNING: No GPU detected! Running on CPU will be very slow.")

    # Check for bitsandbytes if quantization is requested
    if load_8bit or load_4bit:
        try:
            import bitsandbytes as bnb

            print(f"✓ bitsandbytes {bnb.__version__} detected")
        except ImportError:
            print("\n⚠️  ERROR: bitsandbytes not installed!")
            print("Install with: pip install bitsandbytes")
            print("Falling back to full precision...\n")
            load_8bit = False
            load_4bit = False

    print("\nStep 1/3: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        raise

    # Build model loading kwargs
    model_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        model_kwargs["max_memory"] = {0: f"{max_memory_gb}GB"}

        if load_4bit:
            print("\nStep 2/3: Loading model with 4-bit quantization...")
            print("  This may take 2-5 minutes...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
        elif load_8bit:
            print("\nStep 2/3: Loading model with 8-bit quantization...")
            print("  This may take 2-5 minutes...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            print("\nStep 2/3: Loading model with full precision (BF16)...")
            print(
                "  ⚠️  WARNING: Large models require significant VRAM in full precision!"
            )
            print("  Consider using --load-8bit or --load-4bit to reduce memory usage.")
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        print("✓ Model loaded to device")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OUT OF MEMORY ERROR!")
        print(f"   The model is too large for your GPU.")
        print(f"   Try using --load-4bit instead of --load-8bit")
        print(f"   Or use a smaller model.")
        raise
    except Exception as e:
        print(f"\n❌ ERROR loading model: {type(e).__name__}")
        print(f"   {str(e)}")
        print(f"\n   If process was killed, check:")
        print(f"   1. dmesg | tail -20  (for OOM killer messages)")
        print(f"   2. Try --load-4bit instead of --load-8bit")
        print(f"   3. Reduce --max-memory value")
        raise

    print("\nStep 3/3: Setting model to eval mode...")
    model.eval()

    # Final memory check
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'=' * 60}")
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Total: {total:.1f} GB")
        print(f"  Free: {total - reserved:.1f} GB")
        print(f"{'=' * 60}")

    print("\n✅ Model loaded successfully!\n")
    return tokenizer, model


def chat_response(message, history, max_tokens, temperature):
    """Generate a streaming response for Gradio chat interface."""
    global model, tokenizer, system_prompt

    if model is None or tokenizer is None:
        yield history + [
            [message, "⚠️ Model not loaded. Please wait for initialization..."]
        ]
        return

    # Build conversation history
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Only add if not None
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Prepare inputs
    prompt_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt")  # type: ignore[operator]
    model_inputs = {
        name: tensor.to(model.device)  # type: ignore[attr-defined]
        for name, tensor in model_inputs.items()
    }

    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,  # type: ignore[attr-defined]
    }

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generate_kwargs["streamer"] = streamer

    # Start generation in separate thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)  # type: ignore[attr-defined]
    thread.start()

    # Stream tokens - yield history with accumulated response
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield history + [[message, generated_text]]

    thread.join()


def create_interface(model_name: str):
    """Create the Gradio interface."""

    with gr.Blocks(title="AI Chat Assistant") as interface:
        gr.Markdown(
            f"""
            # 🤖 AI Chat Assistant
            
            **Model:** {model_name}
            
            A powerful AI assistant ready to help with your questions!
            """
        )

        chatbot = gr.Chatbot(
            height=500,
            label="Chat",
            value=[],
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your question here...",
                scale=4,
            )
            submit = gr.Button("Send", variant="primary", scale=1)
            stop = gr.Button("⏹️ Stop", variant="stop", scale=1)

        with gr.Accordion("Advanced Settings", open=False):
            max_tokens = gr.Slider(
                minimum=50,
                maximum=2048,
                value=512,
                step=50,
                label="Max New Tokens",
                info="Maximum number of tokens to generate",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused. Set to 0 for greedy decoding.",
            )

        with gr.Row():
            clear = gr.Button("🗑️ Clear Chat")

        gr.Markdown(
            """
            ---
            ### Tips:
            - Responses stream in real-time
            - Click **Stop** to cancel the current response without clearing chat
            - Click **Clear Chat** to start a fresh conversation
            - Use advanced settings to control output length and creativity
            """
        )

        # Helper function to extract text from Gradio message format
        def extract_text_from_message(content):
            """Extract plain text from Gradio message format."""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Gradio message format: [{'text': '...', 'type': 'text'}, ...]
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                return "".join(text_parts) if text_parts else str(content)
            else:
                return str(content) if content else ""

        def filter_system_messages(history):
            """Remove system messages from history for display."""
            if not history:
                return []
            return [msg for msg in history if msg.get("role") != "system"]

        def clean_generated_text(text):
            """Remove any system message patterns from generated text."""
            import re

            # Remove system message headers and content that appears before actual assistant response
            # Pattern: "system\nYou are..." or "system\nYou are Qwen..."
            text = re.sub(
                r"system\s*\n.*?(?=\n(?:user|assistant)|$)",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Also remove standalone "system" or "user" labels
            text = re.sub(r"\b(system|user)\b\s*\n", "", text, flags=re.IGNORECASE)
            # Clean up any "You are Qwen" or similar patterns
            text = re.sub(
                r"You are Qwen.*?Alibaba[^.]*\.",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return text.strip()

        # Event handlers
        def user_submit(message, history):
            """Add user message to history and clear input."""
            if history is None:
                history = []

            # Always filter out system messages
            history = filter_system_messages(history)

            message_text = extract_text_from_message(message)
            if not message_text or not message_text.strip():
                return "", history
            return "", history + [{"role": "user", "content": message_text}]

        def bot_response(history, max_tokens, temperature):
            """Generate bot response for the last user message."""
            # Always filter out system messages for display
            history = filter_system_messages(history)

            if not history or history[-1].get("role") != "user":
                return history if history else []

            user_message = extract_text_from_message(history[-1]["content"])

            # Stream the response and update the last message
            global model, tokenizer, system_prompt

            if model is None or tokenizer is None:
                history.append({"role": "assistant", "content": "⚠️ Model not loaded."})
                yield filter_system_messages(history)
                return

            # Build conversation messages for the model (NO system role to avoid display issues)
            # Instead, we'll prepend the system prompt directly to the final prompt
            messages = []
            for msg in history[:-1]:  # Exclude the last message
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] != "system":  # Skip any system messages
                        content = extract_text_from_message(msg["content"])
                        messages.append({"role": msg["role"], "content": content})

            # Add the current user message
            messages.append({"role": "user", "content": user_message})

            # Prepare inputs - prepend system prompt to the template
            prompt_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages, tokenize=False, add_generation_prompt=True
            )
            # Manually prepend system prompt if the template doesn't include it
            if system_prompt not in prompt_text:
                prompt_text = f"{system_prompt}\n\n{prompt_text}"
            model_inputs = tokenizer(prompt_text, return_tensors="pt")  # type: ignore[operator]
            model_inputs = {
                name: tensor.to(model.device)  # type: ignore[attr-defined]
                for name, tensor in model_inputs.items()
            }

            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": tokenizer.eos_token_id,  # type: ignore[attr-defined]
            }

            # Setup streaming with skip_prompt to avoid outputting the input
            streamer = TextIteratorStreamer(
                tokenizer, skip_special_tokens=True, skip_prompt=True
            )
            generate_kwargs["streamer"] = streamer

            # Start generation
            thread = Thread(target=model.generate, kwargs=generate_kwargs)  # type: ignore[attr-defined]
            thread.start()

            # Start with an empty assistant message and update it while streaming
            history.append({"role": "assistant", "content": ""})
            yield filter_system_messages(history)

            # Stream tokens - only the newly generated text (skip_prompt handles this)
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                history[-1]["content"] = generated_text
                yield filter_system_messages(history)

            thread.join()

        # Event handlers with cancellation support
        submit_event = msg.submit(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            bot_response,
            inputs=[chatbot, max_tokens, temperature],
            outputs=[chatbot],
        )

        click_event = submit.click(
            user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            bot_response,
            inputs=[chatbot, max_tokens, temperature],
            outputs=[chatbot],
        )

        # Stop button cancels ongoing generation without clearing chat
        stop.click(
            None,
            cancels=[submit_event, click_event],
        )

        # Clear button cancels ongoing generation and clears chat
        clear.click(
            lambda: [],
            outputs=[chatbot],
            cancels=[submit_event, click_event],
        )

    return interface


def main():
    global model, tokenizer, system_prompt

    args = parse_args()
    system_prompt = args.system

    # Load model
    try:
        tokenizer, model = load_model(
            args.model,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            max_memory_gb=args.max_memory,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Loading interrupted by user. Exiting...")
        return
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR during model loading!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   Debugging steps:")
        print(f"   1. Check if you have enough GPU memory:")
        print(f"      nvidia-smi")
        print(f"   2. Check system logs for OOM killer:")
        print(f"      dmesg | grep -i 'killed process'")
        print(f"      dmesg | tail -30")
        print(f"   3. Try 4-bit quantization instead:")
        print(f"      python gradio_app.py --load-4bit")
        print(f"   4. Reduce memory limit:")
        print(f"      python gradio_app.py --load-8bit --max-memory 80")
        return

    # Create and launch interface
    interface = create_interface(args.model)

    print(f"\n{'=' * 60}")
    print(f"🚀 Launching Gradio interface on port {args.port}...")
    print(f"   Access at http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    """

    try:
        interface.launch(
            server_name="0.0.0.0",  # Listen on all interfaces
            server_port=args.port,
            share=args.share,
            show_error=True,
            theme=gr.themes.Soft(),
            css=custom_css,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Gradio server stopped by user.")
    except Exception as e:
        print(f"\n\n❌ ERROR launching Gradio:")
        print(f"   {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
