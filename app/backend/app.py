from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
import sounddevice as sd
import numpy as np
from transformers import AutoTokenizer, VitsModel
from openai import OpenAI
import torch
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import time

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize OpenAI client
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("small")

# Load MMS TTS model for text-to-speech
mms = VitsModel.from_pretrained("facebook/mms-tts-ind")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")

# In-memory conversation storage
conversation_history = {}


# Pydantic model for text input
class TextInput(BaseModel):
    text: str

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ConversationRequest(BaseModel):
    user_id: str
    message: str

class ConversationResponse(BaseModel):
    messages: List[Message]

@app.post("/record")
def record_audio():
    """Record audio from the microphone and transcribe it."""
    duration = 5  # Duration in seconds
    sample_rate = 16000  # Sampling rate

    try:
        start_time = time.perf_counter()
        print("Recording audio...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("Audio recorded.")

        # Process audio with Whisper
        audio_data = np.squeeze(audio_data)
        transcription = whisper_model.transcribe(audio_data, fp16=False, language="id")
        end_time = time.perf_counter()
        print(f"ASR Time: {(end_time-start_time):.4f} seconds")
        return {
            "text": transcription["text"],
            "time_taken_seconds": end_time - start_time
        }
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/conversation", response_model=ConversationResponse)
def manage_conversation(input: ConversationRequest):
    """Manage conversation with context."""
    user_id = input.user_id

    # Initialize user history if not present
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add user message to history
    conversation_history[user_id].append({"role": "user", "content": input.message})

    # Add system-level instruction for brevity if not already present
    if not any(msg["role"] == "system" for msg in conversation_history[user_id]):
        conversation_history[user_id].insert(0, {"role": "system", "content": "Tolong jawab kurang dari 30 kata."})

    # Generate assistant response
    try:
        start_time = time.perf_counter()

        completion = client.chat.completions.create(
            model="LLaMA_CPP",
            messages=conversation_history[user_id]
        )
        assistant_response = completion.choices[0].message.content.strip()
        # Remove special tokens (e.g., <|eot_id|>)
        cleaned_response = assistant_response.replace("<|eot_id|>", "").strip()

        # Add assistant response to history
        conversation_history[user_id].append({"role": "assistant", "content": cleaned_response})

        end_time = time.perf_counter()
        print(f"LLM Time: {(end_time-start_time):.4f} seconds")

        # Return updated conversation history
        return {
            "messages": conversation_history[user_id],
            "time_taken_seconds": end_time - start_time    
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/respond")
# def generate_response(input: TextInput):
#     """Generate a response using the LLM."""
#     try:
#         completion = client.chat.completions.create(
#             model="LLaMA_CPP",
#             messages=[
#                 {"role": "system", "content": "Tolong jawab dengan singkat"},
#                 {"role": "user", "content": input.text}
#             ]
#         )
#         response = completion.choices[0].message.content
#         # Remove special tokens (e.g., <|eot_id|>)
#         cleaned_response = response.replace("<|eot_id|>", "").strip()
#         return {"response": cleaned_response}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
def text_to_speech(input: TextInput):
    """Convert text to speech and play the audio."""
    try:
        start_time = time.perf_counter()
        inputs = mms_tokenizer(input.text, return_tensors="pt")
        with torch.no_grad():
            output = mms(**inputs).waveform

        # Convert waveform to audio and play
        audio_array = output.squeeze().cpu().numpy()
        sample_rate = 16000
        sd.play(audio_array, sample_rate)
        sd.wait()

        end_time = time.perf_counter()
        print(f"TTS: {(end_time-start_time):.4f} seconds")

        return {
            "message": "Speech played successfully",
            "time_taken_seconds": end_time - start_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
