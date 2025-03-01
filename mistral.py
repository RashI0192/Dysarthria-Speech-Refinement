import streamlit as st
import os
import torch
import librosa
import numpy as np
import sounddevice as sd
import base64
import openai
from transformers import pipeline
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
HF_TOKEN = os.getenv("huggingface")

# Load Mistral Model
mistral_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

def generate_transcription_with_mistral(audio_text):
    """Use Mistral to generate a refined transcription."""
    prompt = f"Correct and enhance this transcription for better clarity and grammar: {audio_text}"
    response = mistral_pipeline(prompt, max_length=512, do_sample=True, top_k=50, temperature=0.7)
    return response[0]['generated_text']

# Function to process and transcribe speech
def speech_to_text(audio_input):
    """Convert speech to text using Mistral."""
    text = librosa.feature.mfcc(y=audio_input, sr=16000, n_mfcc=13)  # Extract basic features
    raw_transcription = " ".join(map(str, text.flatten()[:50]))  # Simulated basic transcription
    return generate_transcription_with_mistral(raw_transcription)

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    st.info(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()

# Streamlit App UI
def main():
    st.title("Speech-to-Text with Mistral AI")
    
    if st.button("Start Recording"):
        audio_data = record_audio(duration=5)
        st.success("Recording complete! Processing now...")
        
        # Transcription using Mistral
        transcription = speech_to_text(audio_data)
        
        st.subheader("Transcription:")
        st.write(transcription)
        
if __name__ == "__main__":
    main()
