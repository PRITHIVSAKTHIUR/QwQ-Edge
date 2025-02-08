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

# QwQ Edge 💬

QwQ Edge is an advanced multimodal chatbot that integrates AI-driven text-to-speech (TTS) generation, image generation using Stable Diffusion XL (SDXL), and a conversational interface powered by large language models. The app supports real-time conversations with multimedia output, including images and speech, and offers a flexible environment for creative and interactive communication.

### Key Features:
1. **Multimodal Conversational AI**: Supports text, image, and speech as input/output.
2. **Text-to-Speech (TTS)**: Convert text responses into speech with multiple voice options using Edge TTS.
3. **Image Generation**: Generate high-quality images based on prompts using the Stable Diffusion XL pipeline.
4. **Multimodal Inputs**: Handle text, image files, and spoken queries for generating appropriate outputs.
5. **Real-time Interaction**: Provides instant responses with streaming output and live updates for image generation and TTS processing.
6. **Customizable Parameters**: Fine-tune token generation parameters such as temperature, top-p, top-k, and repetition penalties for more controlled responses.

### Technologies Used:
- **Gradio**: For creating the user-friendly interface.
- **Transformers**: For NLP and text processing (Hugging Face models).
- **Stable Diffusion XL**: For high-quality image generation.
- **Edge TTS**: For converting text into speech.
- **Python**: For the backend logic and model integration.
- **PyTorch**: For deep learning model inference and acceleration.

### Supported Commands:
- **Text-to-Speech (TTS)**: 
  - `@tts1`: Use the "en-US-JennyNeural" voice.
  - `@tts2`: Use the "en-US-GuyNeural" voice.
- **Image Generation**:
  - `@image <description>`: Generate an image from the given description using Stable Diffusion XL.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/QwQ-Edge.git
   cd QwQ-Edge
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables for model paths and parameters:
   - `MODEL_VAL_PATH`: Path to the SDXL model.
   - `MAX_INPUT_TOKEN_LENGTH`: Maximum token length for model inputs.
   - Other environment variables for batch size, resolution, and CPU/GPU usage.

4. Launch the app:
   ```bash
   python app.py
   ```

### Example Usage:
1. **TTS Example**:
   - Type: `@tts1 What is quantum computing?`
   - The app will convert this text into speech using the JennyNeural voice.
   
2. **Image Generation Example**:
   - Type: `@image A futuristic city skyline at sunset with neon lights`
   - The app will generate an image based on the prompt using the SDXL model.

3. **Chatbot Conversation Example**:
   - Type: `What is the capital of France?`
   - The app will respond with a chatbot-generated text response.

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
