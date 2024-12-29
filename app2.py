import gradio as gr
import numpy as np
import sounddevice as sd
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import whisper
import torch
import time
import pyttsx3

# --- Load Models ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="cpu"
)
whisper_model = whisper.load_model("small")

# Initialise TTS engine
tts_engine = pyttsx3.init()


# --- Helper Functions ---
def record_audio(duration=5, sample_rate=16000):
    """Records audio for a specified duration."""
    audio_data = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    return np.squeeze(audio_data)


def transcribe_audio(audio_data):
    """Transcribes the audio data using Whisper."""
    start_time = time.time()
    result = whisper_model.transcribe(audio_data, fp16=False, language="id")
    transcription_time = time.time() - start_time
    return result["text"], transcription_time


def generate_response(user_input):
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


def say_response(response):
    """Speaks the response using TTS."""
    tts_engine.say(response)
    tts_engine.runAndWait()


# --- Gradio UI ---
def process_audio():
    """Handles transcription and response generation sequentially."""
    # Record audio
    audio_data = record_audio()

    # Transcribe audio
    transcription, transcription_time = transcribe_audio(audio_data)
    yield transcription, None, f"Transcription Time: {round(transcription_time, 2)}s"

    # Generate response
    response, response_time = generate_response(transcription)
    yield transcription, response, f"Response Time: {round(response_time, 2)}s"

    # Speak the response after showing it
    say_response(response)


with gr.Blocks() as demo:
    gr.Markdown("### Voice Assistant with Written and Spoken Responses")

    with gr.Row():
        button = gr.Button("Record")
    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription", interactive=False)
    with gr.Row():
        response_output = gr.Textbox(label="Response", interactive=False)
    with gr.Row():
        timing_output = gr.Textbox(label="Processing Time", interactive=False)

    button.click(
        process_audio,
        inputs=None,
        outputs=[transcription_output, response_output, timing_output],
    )

# Launch the interface
demo.launch()
