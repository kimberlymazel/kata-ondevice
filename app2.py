import gradio as gr
import numpy as np
import sounddevice as sd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
    VitsModel,
    AutoTokenizer as VitsTokenizer,
)
import whisper
import torch
import time
import asyncio

# --- Load Models ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="cpu"
)
whisper_model = whisper.load_model("small")
mms = VitsModel.from_pretrained("facebook/mms-tts-ind")
mms_token = VitsTokenizer.from_pretrained("facebook/mms-tts-ind")


# --- TTS ---
def ngomong(text):
    inputs = mms_token(text, return_tensors="pt")
    with torch.no_grad():
        output = mms(**inputs).waveform
        return output.squeeze().cpu().numpy()


# --- Helper Functions ---
def record_audio(duration=5, sample_rate=16000):
    """Records audio for a specified duration."""
    audio_data = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    return np.squeeze(audio_data)


async def transcribe_audio(audio_data):
    """Transcribes the audio data using Whisper."""
    start_time = time.time()
    result = whisper_model.transcribe(audio_data, fp16=False, language="id")
    transcription_time = time.time() - start_time
    return result["text"], transcription_time


async def generate_response(user_input):
    """Generates a response using the LLaMA model."""
    start_time = time.time()
    messages = [
        {"role": "system", "content": "Tolong jawab singkat."},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response_time = time.time() - start_time
    return response, response_time


# --- Gradio UI ---
async def process_audio():
    """Handles transcription, response generation, and TTS in parallel."""
    # Step 1: Record audio
    audio_data = record_audio()

    # Step 2: Start transcription and response generation in parallel
    transcription_task = asyncio.create_task(transcribe_audio(audio_data))
    transcription, transcription_time = await transcription_task

    # Step 3: Yield transcription immediately
    yield f"{transcription} (Time: {round(transcription_time, 2)}s)", None, None

    # Step 4: Generate response and convert it to speech
    response_task = asyncio.create_task(generate_response(transcription))
    response, response_time = await response_task
    speech_waveform = ngomong(response)

    # Step 5: Yield response and audio
    yield transcription, f"{response} (Time: {round(response_time, 2)}s)", speech_waveform


with gr.Blocks() as demo:
    gr.Markdown("### Voice Assistant with Parallel Updates")

    # UI Components
    with gr.Row():
        button = gr.Button("Record")
    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription", interactive=False)
    with gr.Row():
        response_output = gr.Textbox(label="Response", interactive=False)
    with gr.Row():
        audio_output = gr.Audio(label="Response Audio")

    # Single button click to process both transcription and response
    button.click(
        process_audio,
        inputs=None,
        outputs=[transcription_output, response_output, audio_output],
    )

# Launch the interface
demo.launch()
