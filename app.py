import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import edge_tts

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.image_utils import load_image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = """
# QwQ Edge 💬
"""

css = '''
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: #fff;
  background: #1565c0;
  border-radius: 100vh;
}
'''

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load text-only model and tokenizer
model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    """Convert text to speech using Edge TTS and save as MP3"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def clean_chat_history(chat_history):
    """
    Filter out any chat entries whose "content" is not a string.
    This helps prevent errors when concatenating previous messages.
    """
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

# Environment variables and parameters for Stable Diffusion XL
# Use : SG161222/RealVisXL_V4.0_Lightning or SG161222/RealVisXL_V5.0_Lightning
MODEL_ID_SD = os.getenv("MODEL_VAL_PATH")  # SDXL Model repository path via env variable
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # For batched image generation

# Load the SDXL pipeline
sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID_SD,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    add_watermarker=False,
).to(device)
sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

# Ensure that the text encoder is in half-precision if using CUDA.
if torch.cuda.is_available():
    sd_pipe.text_encoder = sd_pipe.text_encoder.half()

# Optional: compile the model for speedup if enabled
if USE_TORCH_COMPILE:
    sd_pipe.compile()

# Optional: offload parts of the model to CPU if needed
if ENABLE_CPU_OFFLOAD:
    sd_pipe.enable_model_cpu_offload()

MAX_SEED = np.iinfo(np.int32).max

def save_image(img: Image.Image) -> str:
    """Save a PIL image with a unique filename and return the path."""
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def progress_bar_html(label: str) -> str:
    """
    Returns an HTML snippet for a thin progress bar with a label.
    The progress bar is styled as a dark red animated bar.
    """
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #FFF0F5; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #FF69B4; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

@spaces.GPU(duration=60, enable_queue=True)
def generate_image_fn(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    num_inference_steps: int = 25,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    num_images: int = 1,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate images using the SDXL pipeline."""
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    options = {
        "prompt": [prompt] * num_images,
        "negative_prompt": [negative_prompt] * num_images if use_negative_prompt else None,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "output_type": "pil",
    }
    if use_resolution_binning:
        options["use_resolution_binning"] = True

    images = []
    # Process in batches
    for i in range(0, num_images, BATCH_SIZE):
        batch_options = options.copy()
        batch_options["prompt"] = options["prompt"][i:i+BATCH_SIZE]
        if "negative_prompt" in batch_options and batch_options["negative_prompt"] is not None:
            batch_options["negative_prompt"] = options["negative_prompt"][i:i+BATCH_SIZE]
        # Wrap the pipeline call in autocast if using CUDA
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = sd_pipe(**batch_options)
        else:
            outputs = sd_pipe(**batch_options)
        images.extend(outputs.images)
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates chatbot responses with support for multimodal input, TTS, and image generation.
    Special commands:
      - "@tts1" or "@tts2": triggers text-to-speech.
      - "@image": triggers image generation using the SDXL pipeline.
    """
    text = input_dict["text"]
    files = input_dict.get("files", [])

    if text.strip().lower().startswith("@image"):
        # Remove the "@image" tag and use the rest as prompt
        prompt = text[len("@image"):].strip()
        # Show animated progress bar for image generation
        yield progress_bar_html("Generating Image")
        image_paths, used_seed = generate_image_fn(
            prompt=prompt,
            negative_prompt="",
            use_negative_prompt=False,
            seed=1,
            width=1024,
            height=1024,
            guidance_scale=3,
            num_inference_steps=25,
            randomize_seed=True,
            use_resolution_binning=True,
            num_images=1,
        )
        # Once done, yield the generated image
        yield gr.Image(image_paths[0])
        return  # Exit early

    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)
    
    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        # Clear previous chat history for a fresh TTS request.
        conversation = [{"role": "user", "content": text}]
    else:
        voice = None
        # Remove any stray @tts tags and build the conversation history.
        text = text.replace(tts_prefix, "").strip()
        conversation = clean_chat_history(chat_history)
        conversation.append({"role": "user", "content": text})

    if files:
        if len(files) > 1:
            images = [load_image(image) for image in files]
        elif len(files) == 1:
            images = [load_image(files[0])]
        else:
            images = []
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        # Show animated progress bar for multimodal generation
        yield progress_bar_html("Thinking...")
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
    else:
        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        outputs = []
        # Show animated progress bar for text generation
        yield progress_bar_html("Thinking...")
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)

        final_response = "".join(outputs)
        yield final_response

        # If TTS was requested, convert the final response to speech.
        if is_tts and voice:
            output_file = asyncio.run(text_to_speech(final_response, voice))
            yield gr.Audio(output_file, autoplay=True)

demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        ["@image Chocolate dripping from a donut against a yellow background, in the style of brocore, hyper-realistic"],
        ["Python Program for Array Rotation"],
        ["@tts1 Who is Nikola Tesla, and why did he die?"],
        [{"text": "Extract JSON from the image", "files": ["examples/document.jpg"]}],
        [{"text": "summarize the letter", "files": ["examples/1.png"]}],
        ["@tts2 What causes rainbows to form?"],
    ],
    cache_examples=False,
    type="messages",
    description=DESCRIPTION,
    css=css,
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple",  placeholder="‎ @tts1, @tts2-voices, @image-image gen, default [text, vision]"),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
