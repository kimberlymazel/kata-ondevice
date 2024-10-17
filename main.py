import speech_recognition as srec
from gtts import gTTS
import pyttsx3 as pyt
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import requests
import json

engine = pyt.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

device = "auto"

# --- INDO MINSTRAL 7B MODEL ---

# model = "indischepartij/MiaLatte-Indo-Mistral-7b"
# tokenizer = AutoTokenizer.from_pretrained(model)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# --- AZURE SAILOR 0.5B MODEL ---

model = AutoModelForCausalLM.from_pretrained(
    'sail/Sailor-0.5B-Chat',
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained('sail/Sailor-0.5B-Chat')

# --- STT ---

def perintah():
    mendengar = srec.Recognizer()
    with srec.Microphone() as source:
        print('Mendengarkan......')
        suara = mendengar.listen(source, phrase_time_limit=5)
        try:
            print('Diterima.....')
            dengar = mendengar.recognize_google(suara, language='id-ID')
            print(dengar)
        except:
            pass
        return dengar

# --- TTS ---

# def ngomong(self):
#     teks = (self)
#     bahasa = 'id'
#     namafile = 'Ngomong.mp3'
#     def reading():
#         suara = gTTS(text=teks, lang=bahasa, slow=False)
#         suara.save(namafile)
#         os.system(f'start {namafile}')
#     reading()

def ngomong(text):
    voices = engine.getProperty('voices')

    for voice in voices:
        if "MSTTS_V110_idID_Andika" in voice.id:
            engine.setProperty('voice', voice.id)
            break

    # Speak the text
    engine.say(text)
    
    # Wait until speaking is finished
    engine.runAndWait()


# --- RUNNING VOICE ASSISTANT ---
def run_va():
    # prompt from user
    Layanan = perintah()

    # messages = [{"role": "user", "content": Layanan}]
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # print(outputs[0]["generated_text"])

    # ngomong(Layanan)

    # print(Layanan)

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

# --- CHECK IF YOUR SYSTEM HAS INDONESIAN TTS ---
# REFER TO https://stackoverflow.com/questions/56730889/pyttsx-isn-t-showing-installed-languages-on-windows-10

# voices = engine.getProperty('voices')
# for voice in voices:
#     print(voice.id)
