#!/usr/bin/env python3
"""Gradio GUI for noise cancellation."""

import asyncio
import tempfile
from pathlib import Path

import gradio as gr
from livekit import rtc
from livekit.plugins import noise_cancellation
from dotenv import load_dotenv

# Import the processor from the main module
import sys
sys.path.insert(0, str(Path(__file__).parent))

# We need to import after path setup
from importlib import import_module
nc_module = import_module("noise-canceller")
AudioFileProcessor = nc_module.AudioFileProcessor

load_dotenv()

MODELS = {
    "NC (Standard Noise Cancellation)": "NC",
    "BVC (Background Voice Cancellation)": "BVC",
    "BVC Telephony (Optimized for calls)": "BVCTelephony",
    "WebRTC (Local, faster)": "WebRTC",
}


def get_filter(model_key: str):
    """Get the appropriate noise filter based on selection."""
    model = MODELS[model_key]
    if model == "WebRTC":
        return None  # WebRTC uses a different path
    filter_map = {
        "NC": noise_cancellation.NC(),
        "BVC": noise_cancellation.BVC(),
        "BVCTelephony": noise_cancellation.BVCTelephony(),
    }
    return filter_map[model]


async def process_audio_async(input_path: str, model_key: str) -> str:
    """Process audio file with selected noise cancellation model."""
    use_webrtc = MODELS[model_key] == "WebRTC"
    noise_filter = get_filter(model_key) if not use_webrtc else noise_cancellation.NC()

    processor = AudioFileProcessor(
        noise_filter=noise_filter,
        use_webrtc=use_webrtc,
        silent=True
    )

    # Create output path
    input_file = Path(input_path)
    output_file = Path(tempfile.gettempdir()) / f"cleaned_{input_file.stem}.wav"

    await processor.process_file(input_file, output_file)

    return str(output_file)


def process_audio(audio_file: str, model: str) -> str:
    """Wrapper to run async processing."""
    if audio_file is None:
        raise gr.Error("Please upload an audio file")

    return asyncio.run(process_audio_async(audio_file, model))


# Build Gradio interface
with gr.Blocks(title="Noise Canceller") as demo:
    gr.Markdown("# Noise Canceller")
    gr.Markdown("Upload an audio file, choose a model, and get a cleaned version.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources=["upload"],
            )
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="NC (Standard Noise Cancellation)",
                label="Noise Cancellation Model",
            )
            submit_btn = gr.Button("Clean Audio", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(
                label="Cleaned Audio",
                type="filepath",
            )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_dropdown],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.launch()
