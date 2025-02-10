import os
import random
import uuid
import json
import time
import asyncio
import tempfile
from threading import Thread
import base64

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import edge_tts
import trimesh

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.image_utils import load_image

from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers import ShapEImg2ImgPipeline, ShapEPipeline
from diffusers.utils import export_to_ply

# -----------------------------------------------------------------------------
# Global constants and helper functions
# -----------------------------------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def glb_to_data_url(glb_path: str) -> str:
    """
    Reads a GLB file from disk and returns a data URL with a base64 encoded representation.
    This data URL can be used as the `src` for an HTML <model-viewer> tag.
    """
    with open(glb_path, "rb") as f:
        data = f.read()
    b64_data = base64.b64encode(data).decode("utf-8")
    return f"data:model/gltf-binary;base64,{b64_data}"

# -----------------------------------------------------------------------------
# Model class for Text-to-3D Generation (ShapE)
# -----------------------------------------------------------------------------

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)
        self.pipe.to(self.device)
        # Ensure the text encoder is in half precision to avoid dtype mismatches.
        if torch.cuda.is_available():
            try:
                self.pipe.text_encoder = self.pipe.text_encoder.half()
            except AttributeError:
                pass

        self.pipe_img = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16)
        self.pipe_img.to(self.device)
        # Use getattr with a default value to avoid AttributeError if text_encoder is missing.
        if torch.cuda.is_available():
            text_encoder_img = getattr(self.pipe_img, "text_encoder", None)
            if text_encoder_img is not None:
                self.pipe_img.text_encoder = text_encoder_img.half()

    def to_glb(self, ply_path: str) -> str:
        mesh = trimesh.load(ply_path)
        # Rotate the mesh for proper orientation
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        mesh.apply_transform(rot)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(mesh_path.name, file_type="glb")
        return mesh_path.name

    def run_text(self, prompt: str, seed: int = 0, guidance_scale: float = 15.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)

    def run_image(self, image: Image.Image, seed: int = 0, guidance_scale: float = 3.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe_img(
            image,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)

# -----------------------------------------------------------------------------
# Gradio UI configuration
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Load Models and Pipelines for Chat, Image, and Multimodal Processing
# -----------------------------------------------------------------------------

# Load the text-only model and tokenizer (for pure text chat)
model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

# Voices for text-to-speech
TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

# Load multimodal processor and model (e.g. for OCR and image processing)
MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

# -----------------------------------------------------------------------------
# Asynchronous text-to-speech
# -----------------------------------------------------------------------------

async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    """Convert text to speech using Edge TTS and save as MP3"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

# -----------------------------------------------------------------------------
# Utility function to clean conversation history
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Stable Diffusion XL Pipeline for Image Generation
# -----------------------------------------------------------------------------

MODEL_ID_SD = os.getenv("MODEL_VAL_PATH")  # SDXL Model repository path via env variable
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # For batched image generation

sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID_SD,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    add_watermarker=False,
).to(device)
sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

if torch.cuda.is_available():
    sd_pipe.text_encoder = sd_pipe.text_encoder.half()

if USE_TORCH_COMPILE:
    sd_pipe.compile()

if ENABLE_CPU_OFFLOAD:
    sd_pipe.enable_model_cpu_offload()

def save_image(img: Image.Image) -> str:
    """Save a PIL image with a unique filename and return the path."""
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

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
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = sd_pipe(**batch_options)
        else:
            outputs = sd_pipe(**batch_options)
        images.extend(outputs.images)
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

# -----------------------------------------------------------------------------
# Text-to-3D Generation using the ShapE Pipeline
# -----------------------------------------------------------------------------

@spaces.GPU(duration=120, enable_queue=True)
def generate_3d_fn(
    prompt: str,
    seed: int = 1,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
    randomize_seed: bool = False,
):
    """
    Generate a 3D model from text using the ShapE pipeline.
    Returns a tuple of (glb_file_path, used_seed).
    """
    seed = int(randomize_seed_fn(seed, randomize_seed))
    model3d = Model()
    glb_path = model3d.run_text(prompt, seed=seed, guidance_scale=guidance_scale, num_steps=num_steps)
    return glb_path, seed

# -----------------------------------------------------------------------------
# Chat Generation Function with support for @tts, @image, and @3d commands
# -----------------------------------------------------------------------------

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
    Generates chatbot responses with support for multimodal input, TTS, image generation,
    and 3D model generation.
    
    Special commands:
      - "@tts1" or "@tts2": triggers text-to-speech.
      - "@image": triggers image generation using the SDXL pipeline.
      - "@3d": triggers 3D model generation using the ShapE pipeline.
    """
    text = input_dict["text"]
    files = input_dict.get("files", [])

    # --- 3D Generation branch ---
    if text.strip().lower().startswith("@3d"):
        prompt = text[len("@3d"):].strip()
        yield "Generating 3D model..."
        glb_path, used_seed = generate_3d_fn(
            prompt=prompt,
            seed=1,
            guidance_scale=15.0,
            num_steps=64,
            randomize_seed=True,
        )
        # Convert the GLB file to a base64 data URL and embed it in an HTML <model-viewer> tag.
        data_url = glb_to_data_url(glb_path)
        html_output = f'''
        <model-viewer src="{data_url}" alt="3D Model" auto-rotate camera-controls style="width: 100%; height: 400px;"></model-viewer>
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        '''
        yield gr.HTML(html_output)
        return

    # --- Image Generation branch ---
    if text.strip().lower().startswith("@image"):
        prompt = text[len("@image"):].strip()
        yield "Generating image..."
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
        yield gr.Image(image_paths[0])
        return

    # --- Text and TTS branch ---
    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)
    
    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        conversation = [{"role": "user", "content": text}]
    else:
        voice = None
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
        yield "Thinking..."
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
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)

        final_response = "".join(outputs)
        yield final_response

        if is_tts and voice:
            output_file = asyncio.run(text_to_speech(final_response, voice))
            yield gr.Audio(output_file, autoplay=True)

# -----------------------------------------------------------------------------
# Gradio Chat Interface Setup and Launch
# -----------------------------------------------------------------------------

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
        ["@tts1 Who is Nikola Tesla, and why did he die?"],
        ["@3d A birthday cupcake with cherry"],
        [{"text": "summarize the letter", "files": ["examples/1.png"]}],
        ["@image Chocolate dripping from a donut against a yellow background, in the style of brocore, hyper-realistic"],
        ["Write a Python function to check if a number is prime."],
        ["@tts2 What causes rainbows to form?"],
    ],
    cache_examples=False,
    type="messages",
    description=DESCRIPTION,
    css=css,
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    # To create a public link, set share=True in launch().
    demo.queue(max_size=20).launch(share=True)
