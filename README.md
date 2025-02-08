---
title: QWQ EDGE
emoji: 💬
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: true
license: creativeml-openrail-m
about: demo space to try how multi model selection function works
short_description: 'Multimodality '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# QwQ Edge: Multimodal AI Assistant

![Demo](demo.gif)

QwQ Edge is an advanced multimodal AI assistant that combines text generation, image understanding, speech synthesis, and image creation capabilities in a single conversational interface. Built with cutting-edge AI models, it supports natural interactions through text, images, and voice commands.

## Key Features

💬 **Multimodal Chat**
- Text conversations with 0.5B parameter LLM
- Image analysis & OCR with Qwen2-VL 2B model
- Support for multiple image uploads
- Context-aware conversation history

🎨 **AI Image Generation**
- Create high-quality images from text prompts
- Powered by Stable Diffusion XL
- Customizable parameters:
  - Resolution up to 1024x1024
  - Guidance scale and inference steps
  - Seed control and randomization

🗣️ **Text-to-Speech**
- Natural sounding voice synthesis
- Multiple voice options:
  - Jenny Neural (Female)
  - Guy Neural (Male)
- Auto-play audio responses

⚡ **Special Commands**
- `@image [prompt]` - Generate images
- `@tts1`/`@tts2` - Switch voice responses
- Multi-file image analysis support

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/qwq-edge.git
cd qwq-edge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_VAL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
```

## Usage

```bash
python app.py
```

**Example Interactions:**
- `@tts1 Explain quantum computing in simple terms`
- `@image Cyberpunk cityscape at night, neon lights, rain reflections`
- Upload image + "What's written in this document?"
- "Write a Python script for face detection"

## Configuration

Environment Variables:
- `MAX_IMAGE_SIZE`: Max resolution for generated images (default: 4096)
- `BATCH_SIZE`: Number of images to generate simultaneously
- `USE_TORCH_COMPILE`: Enable model compilation for speed
- `ENABLE_CPU_OFFLOAD`: Enable memory optimization

Model Parameters:
- Temperature: 0.1-4.0 (response creativity)
- Top-p: 0.05-1.0 (response diversity)
- Max Tokens: Up to 2048 tokens
- Repetition Penalty: 1.0-2.0

## Technical Architecture

```mermaid
graph TD
    A[User Interface] --> B[Chat Logic]
    B --> C{Command Type}
    C -->|Text| D[FastThink-0.5B]
    C -->|Image| E[Qwen2-VL-OCR-2B]
    C -->|@image| F[Stable Diffusion XL]
    C -->|@tts| G[Edge TTS]
    D --> H[Response]
    E --> H
    F --> H
    G --> H
```
