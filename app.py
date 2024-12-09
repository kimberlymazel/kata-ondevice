import streamlit as st
import speech_recognition as srec
from gtts import gTTS
import pyttsx3 as pyt
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper
import sounddevice as sd
import numpy as np
import torch
import time
import psutil

# --- Initialize TTS Engine ---
engine = pyt.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

# --- Load Models ---
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu"
)
whisper_model = whisper.load_model("small")

# --- Helper Functions ---
def record_audio():
    duration = 5
    sample_rate = 16000
    st.info("Recording for 5 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio_data = np.squeeze(audio_data)
    return audio_data

def transcribe_audio(audio_data):
    st.info("Transcribing audio...")
    result = whisper_model.transcribe(audio_data, fp16=False, language="id")
    return result["text"]

def generate_response(user_input):
    messages = [
        {"role": "system", "content": "Tolong jawab singkat."},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt") 

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512, 
        temperature=0.7,
        top_p=0.9,
        do_sample=True

    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def speak_text(text):
    voices = engine.getProperty("voices")
    for voice in voices:
        if "MSTTS_V110_idID_Andika" in voice.id:
            engine.setProperty("voice", voice.id)
            break
    engine.say(text)
    engine.runAndWait()

# --- Streamlit UI ---
st.title("Indonesian Voice Assistant")
st.write("This is a demo of the on-device voice assistant system.")

# Recording Section
if st.button("Record Audio"):
    audio_data = record_audio()
    st.audio(audio_data, format="audio/wav", sample_rate=16000)

    # Transcription
    transcription = transcribe_audio(audio_data)
    st.write("**Transcription:**", transcription)

    # Response Generation
    response = generate_response(transcription)
    st.write("**Response:**", response)

    # Text-to-Speech
    if st.button("Speak Response"):
        st.info("Speaking response...")
        speak_text(response)
