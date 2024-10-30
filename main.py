from gtts import gTTS
import pyttsx3 as pyt
from transformers import AutoTokenizer, AutoModelForCausalLM, VitsModel
import torch

import whisper
import sounddevice as sd
import numpy as np
from IPython.display import Audio


model = AutoModelForCausalLM.from_pretrained(
    'sail/Sailor-0.5B-Chat',
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained('sail/Sailor-0.5B-Chat')

# --- STT ---

whisper_model = whisper.load_model("small")

def perintah():
    duration = 5
    sample_rate = 16000  

    print("Mendengarkan......")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    print("Diterima.....")

    audio_data = np.squeeze(audio_data)  
    dengar = whisper_model.transcribe(audio_data, fp16=False, language="id")
    print(dengar["text"])

    return dengar["text"]

# --- TTS ---

mms = VitsModel.from_pretrained("facebook/mms-tts-ind")
mms_token = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")

def ngomong(text):
    inputs = mms_token(text, return_tensors="pt")

    with torch.no_grad():
        output = mms(**inputs).waveform
    
    return Audio(output.squeeze().cpu().numpy(), rate=16000) # Can change rate to make it faster/slower


# --- RUNNING VOICE ASSISTANT ---
def run_va():
    # prompt from user
    Layanan = perintah()

    system_prompt= 'Jawab dalam bahasa indonesia.'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "question", "content": Layanan}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt")  # Keep it on CPU
    input_ids = model_inputs.input_ids  # No need to transfer to device (GPU)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=512,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    ngomong(response)

run_va()