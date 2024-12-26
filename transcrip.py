import gradio as gr
import sounddevice as sd
import numpy as np
import whisper

# --- Load Whisper Model ---
whisper_model = whisper.load_model("small")


# --- Helper Functions ---
def record_audio(duration=5, sample_rate=16000):
    """Records audio for a specified duration."""
    audio_data = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()  # Ensure recording is complete
    return np.squeeze(audio_data)


def transcribe_audio(audio_data):
    """Transcribes audio using Whisper."""
    result = whisper_model.transcribe(audio_data, fp16=False, language="id")
    return result["text"]


def handle_transcription():
    """Handles recording and transcription."""
    # Record audio
    audio_data = record_audio()
    # Transcribe audio
    transcription = transcribe_audio(audio_data)
    return transcription


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("### Basic Transcription App")
    record_button = gr.Button("Record and Transcribe")
    output_text = gr.Textbox(label="Transcription Output", lines=5, interactive=False)

    # Bind the button to the transcription handler
    record_button.click(handle_transcription, outputs=output_text)

# Launch the Gradio app
demo.launch()
